import numpy as np
import sounddevice as sd
import librosa
import time
import pygame
import random
from collections import deque

# === AUDIO CONFIGURATION ===
fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
buffer_duration = 1.5
buffer_samples = int(buffer_duration * fs)
rms_window_duration = 0.2
rms_window_samples = int(rms_window_duration * fs)
rms_threshold = 0.01
onset_cooldown_sec = 0.5
onset_slice_duration = 0.5

# === VISUAL CONFIGURATION ===
WIDTH, HEIGHT = 800, 600
MAX_CIRCLES = 50
FPS = 30

# === STATE ===
audio_buffer = deque(maxlen=buffer_samples)
rms_history = deque(maxlen=3)
last_onset_rms = 1e-6
last_onset_time = 0
circles = []

# === LOG-LINEAR DECAY MODEL (from fitted values) ===
A, B = 39.77, -41.52  # From digitized Figure 8

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Frequency-Aware Blurriness Visualizer")
clock = pygame.time.Clock()
background_color = np.array([255, 255, 255], dtype=np.float32)

# === HELPER FUNCTIONS ===
def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

def estimate_pitch(signal, sr):
    f0 = librosa.yin(signal, fmin=30, fmax=2000, sr=sr)
    valid_f0 = f0[f0 > 0]
    return float(np.median(valid_f0)) if len(valid_f0) > 0 else 440.0

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def decay_multiplier_from_frequency(freq, frame_dt):
    decay_rate = A + B * np.log10(freq)
    return 10 ** (decay_rate * frame_dt / 20)

def random_color():
    return [random.randint(50, 255) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

# === CIRCLE CLASS ===
class Circle:
    def __init__(self, x, y, r, color, freq, rms):
        self.x = x
        self.y = y
        self.r = r
        self.color = np.array(color, dtype=np.float32)
        self.alpha = 255
        self.freq = freq
        self.decay = decay_multiplier_from_frequency(freq, frame_duration)
        self.initial_rms = rms

    def update(self, normalized_rms):
        decay_ratio = self.decay * normalized_rms
        self.r *= decay_ratio
        self.r = min(self.r, 40)
        self.alpha *= 0.97
        return self.r > 1 and self.alpha > 5

    def draw(self, surface):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        color_with_alpha = (*self.color.astype(int), int(self.alpha))
        pygame.draw.circle(s, color_with_alpha, (int(self.x), int(self.y)), max(1, int(self.r)))
        surface.blit(s, (0, 0))

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    global last_onset_rms, last_onset_time, circles, background_color
    if status:
        print("Stream status:", status)
    audio_chunk = indata[:, 0]
    audio_buffer.extend(audio_chunk)

    if len(audio_buffer) >= buffer_samples:
        buffer_array = np.array(audio_buffer)
        recent_rms_slice = buffer_array[-rms_window_samples:]
        current_rms = compute_rms(recent_rms_slice)
        rms_history.append(current_rms)
        smoothed_rms = np.mean(rms_history)
        now = time.time()

        normalized_rms = smoothed_rms / (last_onset_rms + 1e-6)
        normalized_rms = np.clip(normalized_rms, 0.0, 2.0)

        print(f"🔊 Current RMS: {smoothed_rms:.5f} | Normalized RMS: {normalized_rms:.2f}")

        # Soft fade to white
        if smoothed_rms < 0.002:
            background_color[:] = background_color * 0.95 + np.array([255, 255, 255]) * 0.05

        if smoothed_rms > rms_threshold:
            onset_slice = buffer_array[-int(onset_slice_duration * fs):]
            onsets = librosa.onset.onset_detect(y=onset_slice, sr=fs)
            if len(onsets) > 0 and (now - last_onset_time > onset_cooldown_sec):
                last_onset_rms = smoothed_rms
                last_onset_time = now
                print(f"🎵 Onset detected! RMS: {smoothed_rms:.5f}")
                freq = estimate_pitch(onset_slice, fs)
                print(f"🎼 Estimated frequency: {freq:.2f} Hz")
                # max_area = (WIDTH * HEIGHT) / 2
                # max_radius = int(np.sqrt(max_area / np.pi))
                max_radius = 40
                size = min(max_radius, int(30 + 200 * min(1.0, smoothed_rms / 0.3)))
                print(f"Size {size}")
                color = random_color()
                x, y = random_position()
                circles.append(Circle(x, y, size, color, freq, smoothed_rms))
                if len(circles) > MAX_CIRCLES:
                    circles.pop(0)

        updated_circles = []
        for c in circles:
            if c.update(normalized_rms):
                updated_circles.append(c)
                background_color = background_color * 0.995 + 0.005 * c.color
        circles[:] = updated_circles

# === MAIN LOOP ===
try:
    with sd.InputStream(channels=1, samplerate=fs, blocksize=frame_samples, callback=audio_callback):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            screen.fill(background_color.astype(int))

            for c in circles:
                c.draw(screen)

            pygame.display.flip()
            clock.tick(FPS)

except KeyboardInterrupt:
    print("\n🛑 Stopped visualizer.")
    pygame.quit()