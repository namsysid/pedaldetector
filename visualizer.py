import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from collections import deque

# === CONFIGURATION ===
fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
fft_window = 2048
cooldown_time = 0.25
decay_model_A = 39.77
decay_model_B = -41.52
WIDTH, HEIGHT = 800, 600
FPS = 30
MAX_CIRCLES = 100
BEAT_HISTORY = 20
BUFFER_SECONDS = 2.0
GLOBAL_RMS_WINDOW_SEC = 10.0               # NEW: rolling window for global RMS max
GLOBAL_RMS_BLEED_THRESH = 0.25            # NEW: threshold on normalized global RMS to allow bleed
buffer_samples = int(BUFFER_SECONDS * fs)
onset_window = int(0.5 * fs)
loudest_note_interval = 0.1  # seconds

# === STATE ===
recent_global_rms = deque(maxlen=int(GLOBAL_RMS_WINDOW_SEC / frame_duration))
normalized_global_rms = 0.0
audio_buffer = np.zeros(buffer_samples)
last_onset_time = {}
recent_amplitudes = {}
circles_by_midi = {}
color_by_midi = {}
last_onset_amp_by_midi = {}
last_onset_rms = 1e-6
background_color = np.array([255, 255, 255], dtype=np.float32)
last_loudest_time = 0

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RMS Visualizer (Global-RMS Bleed)")
clock = pygame.time.Clock()

# === UTILS ===
def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def compute_decay_multiplier(freq):
    decay_rate = decay_model_A + decay_model_B * np.log10(freq)
    return 10 ** (decay_rate * frame_duration / 20)

def find_note_peaks(signal, sr):
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(signal), 1 / sr)
    peaks, _ = find_peaks(spectrum, height=0.3 * np.max(spectrum) if np.max(spectrum) > 0 else 0)
    return [(freqs[i], spectrum[i]) for i in peaks if 30 < freqs[i] < 4200]

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def random_color():
    return [random.randint(0, 100) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

# === CIRCLE ===
class Circle:
    def __init__(self, x, y, r, color, freq, amp, midi):
        self.x = x
        self.y = y
        self.r = r
        self.color = np.array(color, dtype=np.float32)
        self.alpha = 255
        self.freq = freq
        self.amp = amp
        self.midi = midi
        self.decay = 1

    def update(self, fft_peaks):
        global background_color, normalized_global_rms, max_recent

        # Track live amp for this MIDI (fallback slight decay)
        self.amp = next((amp for freq, amp in fft_peaks if freq_to_midi(freq) == self.midi), self.amp * 0.98)

        # Keep radius tied to per-note normalized amp (against that note's last onset amp)
        normalized_note = self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6)
        self.r = max(1.0, float(normalized_note) * 58.0)
        glob_normalized_note = self.amp / (max_recent + 1e-6)
        if self.midi == 69:
            print(max_recent)
        # NEW: bleed based ONLY on global normalized RMS vs threshold
        if glob_normalized_note >= GLOBAL_RMS_BLEED_THRESH:
            background_color += 0.007 * (self.color - background_color)

        return self.r > 1 and self.alpha > 5

    def draw(self, surface):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(
            s,
            (*self.color.astype(int), int(self.alpha)),
            (int(self.x), int(self.y)),
            max(1, int(self.r))
        )
        surface.blit(s, (0, 0))

# === PROCESS FRAME ===
def process_frame():
    global last_onset_rms, last_loudest_time, normalized_global_rms, max_recent

    now = time.time()

    # Global RMS for this frame; update rolling window & normalized metric
    current_rms = compute_rms(audio_buffer[-onset_window:])
    recent_global_rms.append(current_rms)
    max_recent = max(recent_global_rms) if recent_global_rms else (current_rms + 1e-6)
    normalized_global_rms = current_rms / (max_recent + 1e-6)

    peaks = find_note_peaks(audio_buffer[-fft_window:], fs)
    if not peaks:
        return

    # Only trigger the loudest note every 100ms
    if now - last_loudest_time >= loudest_note_interval:
        freq, amp = max(peaks, key=lambda p: p[1])
        if amp < 5:
            return
        midi = freq_to_midi(freq)
        last_time = last_onset_time.get(midi, 0)

        if now - last_time > cooldown_time:
            if midi not in color_by_midi:
                color_by_midi[midi] = random_color()
            last_onset_amp_by_midi[midi] = amp

            if midi in circles_by_midi:
                circle = circles_by_midi[midi]
                circle.r = amp
                circle.amp = amp
            else:
                circles_by_midi[midi] = Circle(*random_position(), amp, color_by_midi[midi], freq, amp, midi)

            last_onset_time[midi] = now
            last_onset_rms = current_rms
            last_loudest_time = now

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# === MAIN LOOP ===
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=fs, blocksize=frame_samples)
with stream:
    running = True
    while running:
        if isinstance(background_color, np.ndarray):
            screen.fill(background_color)
        else:
            screen.fill([255, 255, 255])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        process_frame()

        if normalized_global_rms > 0.07:
            background_color += 0.085 * (255 - background_color)
        else:
            background_color = 255.0

        fft_peaks = find_note_peaks(audio_buffer[-fft_window:], fs)
        expired = []
        for midi, circle in list(circles_by_midi.items()):
            if not circle.update(fft_peaks):
                expired.append(midi)
        for midi in expired:
            del circles_by_midi[midi]

        for circle in circles_by_midi.values():
            circle.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)