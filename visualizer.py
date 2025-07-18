import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import librosa
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
buffer_samples = int(BUFFER_SECONDS * fs)
onset_window = int(0.5 * fs)

# === STATE ===
audio_buffer = np.zeros(buffer_samples)
last_onset_time = {}
recent_amplitudes = {}
circles_by_midi = {}
color_by_midi = {}
last_onset_amp_by_midi = {}
last_onset_rms = 1e-6
background_color = np.array([255, 255, 255], dtype=np.float32)

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Frequency-Aware RMS Decay Visualizer")
clock = pygame.time.Clock()

# === UTILITIES ===
def freq_to_midi(freq):
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def compute_decay_multiplier(freq):
    decay_rate = decay_model_A + decay_model_B * np.log10(freq)
    return 10 ** (decay_rate * frame_duration / 20)

def find_note_peaks(signal, sr):
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(rfft(windowed))
    freqs = rfftfreq(len(signal), 1 / sr)
    peaks, _ = find_peaks(spectrum, height=0.3 * np.max(spectrum))
    return [(freqs[i], spectrum[i]) for i in peaks if 30 < freqs[i] < 4200]

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def is_beat_modulated(amps, last_onset_time, now, midi, tau=0.75, threshold=0.0):
    if len(amps) < 2:
        return False
    ts, ys = zip(*amps)
    ts = np.array(ts)
    ys = np.array(ys)
    delta = ys[-1] - ys[-2]
    slope = (ys[-1] - ys[0]) / (ts[-1] - ts[0] + 1e-6)
    mean_amp = np.mean(ys)
    if mean_amp == 0:
        return False
    mi = (np.max(ys) - np.min(ys)) / mean_amp
    age = now - last_onset_time
    cooldown_penalty = np.exp(-age / tau)
    score = (
        + 0.0559 * delta +
        + 4.966 * slope +
        - 0.959 * mi +
        - 0.0 * cooldown_penalty
    )
    return score < threshold

def random_color():
    return [random.randint(50, 255) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

# === CIRCLE CLASS ===
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
        self.amp = next((amp for freq, amp in fft_peaks if freq_to_midi(freq) == self.midi), self.amp * 0.98)
        normalized_rms = self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6)
        decay_rate = self.decay * normalized_rms
        decay_rate = min(decay_rate, 1)
        # self.r *= decay_rate
        self.r = normalized_rms * 58
        # self.alpha *= 0.97
        global background_color
        self.color = np.clip(self.color, 0, 255)
        if self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6) < 0.5 and self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6) > 0.05:
            background_color += 0.00375 * (self.color - background_color)
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

#INTERNALS
prev_spectrum = np.zeros(fft_window // 2 + 1)  # init at global scope

def compute_spectral_flux(signal):
    global prev_spectrum
    windowed = signal * np.hanning(len(signal))
    current_spectrum = np.abs(rfft(windowed))
    
    flux = np.sum((np.maximum(0, current_spectrum - prev_spectrum)) ** 2)
    prev_spectrum = current_spectrum
    return flux

# === PROCESS FRAME ===
def process_frame():
    global last_onset_rms

    current_rms = compute_rms(audio_buffer[-onset_window:])
    flux = compute_spectral_flux(audio_buffer[-fft_window:])
    normalized_rms = current_rms / (last_onset_rms + 1e-6)
    now = time.time()
    # print(str(normalized_rms) + " Flux " + str(flux))
    if flux != 0.0:
        print(flux)

    peaks = find_note_peaks(audio_buffer[-fft_window:], fs)

    for freq, amp in peaks:
        if amp < 5:
            continue
        midi = freq_to_midi(freq)
        last_time = last_onset_time.get(midi, 0)

        if midi not in recent_amplitudes:
            recent_amplitudes[midi] = deque(maxlen=BEAT_HISTORY)
        recent_amplitudes[midi].append((now, amp))

        if is_beat_modulated(recent_amplitudes[midi], last_time, now, midi):
            continue

        if now - last_time > cooldown_time:
            # print(f"🎵 {freq:.1f} Hz (MIDI {midi}) | amp: {amp:.3f}")
            max_radius = 40
            size = min(max_radius, int(30 + 200 * min(1.0, amp / 100)))
            last_onset_amp_by_midi[midi] = amp
            if midi not in color_by_midi:
                color_by_midi[midi] = random_color()

            if midi in circles_by_midi:
                circle = circles_by_midi[midi]
                circle.r = size
                circle.amp = amp
            else:
                circles_by_midi[midi] = Circle(*random_position(), size, color_by_midi[midi], freq, amp, midi)

            last_onset_time[midi] = now
            last_onset_rms = current_rms

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
        background_color += 0.05 * (255 - background_color)
        screen.fill(background_color.astype(int))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        process_frame()

        fft_peaks = find_note_peaks(audio_buffer[-fft_window:], fs)
        expired = []
        for midi, circle in circles_by_midi.items():
            if not circle.update(fft_peaks):
                expired.append(midi)
        for midi in expired:
            del circles_by_midi[midi]

        for circle in circles_by_midi.values():
            circle.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)