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
MAX_CIRCLES = 50
BEAT_HISTORY = 20
BUFFER_SECONDS = 2.0
buffer_samples = int(BUFFER_SECONDS * fs)
onset_window = int(0.5 * fs)

# === STATE ===
audio_buffer = np.zeros(buffer_samples)
last_onset_time = {}
recent_amplitudes = {}
circles = []
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
    def __init__(self, x, y, r, color, freq, amp):
        self.x = x
        self.y = y
        self.r = r
        self.color = np.array(color, dtype=np.float32)
        self.alpha = 255
        # self.decay = compute_decay_multiplier(freq)
        self.decay = 1

    def update(self, normalized_rms):
        decay_rate = self.decay * normalized_rms
        decay_rate = min(decay_rate, 1)
        self.r *= decay_rate
        self.alpha *= 0.97
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
    global last_onset_rms

    # Compute current RMS from last 0.5s of buffer
    current_rms = compute_rms(audio_buffer[-onset_window:])
    normalized_rms = current_rms / (last_onset_rms + 1e-6)
    now = time.time()

    # Run FFT peak detection
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
            print(f"🎵 {freq:.1f} Hz (MIDI {midi}) | amp: {amp:.3f}")
            max_radius = 40
            size = min(max_radius, int(30 + 200 * min(1.0, amp / 100)))
            circles.append(Circle(*random_position(), size, random_color(), freq, amp))
            if len(circles) > MAX_CIRCLES:
                circles.pop(0)
            last_onset_time[midi] = now
            last_onset_rms = current_rms
    background_color[:] = background_color * 0.975 + np.array([255, 255, 255]) * 0.025

    # Update circles with shared normalized RMS
    new_circles = []
    for c in circles:
        if c.update(normalized_rms):
            new_circles.append(c)
            background_color[:] = background_color * 0.995 + 0.005 * c.color
    circles[:] = new_circles

# === MAIN LOOP ===
try:
    with sd.InputStream(channels=1, samplerate=fs, blocksize=frame_samples, callback=lambda indata, frames, time_info, status: audio_buffer.__setitem__(slice(None), np.roll(audio_buffer, -len(indata[:, 0]))) or audio_buffer.__setitem__(slice(-len(indata[:, 0]), None), indata[:, 0])):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            process_frame()
            screen.fill(background_color.astype(int))
            for c in circles:
                c.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)
except KeyboardInterrupt:
    pygame.quit()
    print("🛑 Visualizer stopped.")