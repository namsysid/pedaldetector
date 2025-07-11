import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from collections import deque

# === CONFIGURATION ===
fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
fft_window = 2048
peak_threshold = 0.3
cooldown_time = 0.5
decay_model_A = 39.77
decay_model_B = -41.52

WIDTH, HEIGHT = 800, 600
FPS = 30
MAX_CIRCLES = 50

BEAT_HISTORY = 20
BEAT_SIMILARITY_THRESHOLD = 0.95

# === STATE ===
audio_buffer = np.zeros(fft_window)
last_onset_time = {}
recent_amplitudes = {}  # midi_note: deque[(time, amp)]
circles = []

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Beat-Suppressed Piano Visualizer")
clock = pygame.time.Clock()
background_color = np.array([255, 255, 255], dtype=np.float32)

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
    peaks, _ = find_peaks(spectrum, height=peak_threshold * np.max(spectrum))
    return [(freqs[i], spectrum[i]) for i in peaks if 30 < freqs[i] < 4200]

def is_beat_modulated(amps, last_onset_time, now, tau=0.2, threshold=1.0):
    if len(amps) < 4:
        return False

    ts, ys = zip(*amps)
    ts = np.array(ts)
    ys = np.array(ys)
    delta = ys[-1] - ys[-2]

    # Slope: trend over last few frames
    slope = (ys[-1] - ys[0]) / (ts[-1] - ts[0] + 1e-6)

    # Modulation Index
    mean_amp = np.mean(ys)
    if mean_amp == 0:
        return False
    mi = (np.max(ys) - np.min(ys)) / mean_amp

    # Cooldown penalty (age since last onset for this MIDI note)
    age = now - last_onset_time
    cooldown_penalty = np.exp(-age / tau)

    # Final suppression score
    score = (
        + 1.0 * delta +
        + 1.0 * slope +
        - 3.0 * mi +
        - 2.5 * cooldown_penalty
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
        self.decay = compute_decay_multiplier(freq)

    def update(self):
        self.r *= self.decay
        self.alpha *= 0.97
        return self.r > 1 and self.alpha > 5

    def draw(self, surface):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color.astype(int), int(self.alpha)), (int(self.x), int(self.y)), max(1, int(self.r)))
        surface.blit(s, (0, 0))

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_chunk = indata[:, 0]
    audio_buffer = np.concatenate([audio_buffer[len(audio_chunk):], audio_chunk])

# === PROCESS FRAME ===
def process_frame():
    global circles, background_color
    now = time.time()
    peaks = find_note_peaks(audio_buffer, fs)

    for freq, amp in peaks:
        if amp < 5:
            continue
        midi = freq_to_midi(freq)
        last_time = last_onset_time.get(midi, 0)
        # if now - last_time < cooldown_time:
        #     continue
        if midi not in recent_amplitudes:
            recent_amplitudes[midi] = deque(maxlen=BEAT_HISTORY)
        recent_amplitudes[midi].append((now, amp))
        # Beat suppression check
        if is_beat_modulated(recent_amplitudes[midi], now - last_onset_time.get(midi, 0), now):
            continue

        if now - last_time > cooldown_time:
            print(f"🎵 {freq:.1f} Hz (MIDI {midi}) | amp: {amp:.3f}")
            max_area = (WIDTH * HEIGHT) / 2
            max_radius = int(np.sqrt(max_area / np.pi))
            size = min(max_radius, int(30 + 200 * min(1.0, amp)))
            circles.append(Circle(*random_position(), size, random_color(), freq, amp))
            if len(circles) > MAX_CIRCLES:
                circles.pop(0)
            last_onset_time[midi] = now

    background_color[:] = background_color * 0.99 + np.array([255, 255, 255]) * 0.01

    new_circles = []
    for c in circles:
        if c.update():
            new_circles.append(c)
            background_color[:] = background_color * 0.995 + 0.005 * c.color
    circles[:] = new_circles
    # if 69 in recent_amplitudes.keys(): #Debugging log, to see how the curve is
    #     print(recent_amplitudes[69])

# === MAIN LOOP ===
try:
    with sd.InputStream(channels=1, samplerate=fs, blocksize=frame_samples, callback=audio_callback):
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
    print(recent_amplitudes.get(69, 0))
    print("🛑 Visualizer stopped.")