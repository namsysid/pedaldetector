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
rms_threshold = 0.002
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

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Blurriness Visualizer")
clock = pygame.time.Clock()
background_color = np.array([255, 255, 255], dtype=np.float32)  # start white

# === CIRCLE CLASS ===
class Circle:
    def __init__(self, x, y, r, color, rms):
        self.x = x
        self.y = y
        self.r = r
        self.color = np.array(color, dtype=np.float32)
        self.alpha = 255
        self.initial_rms = rms

    def update(self, normalized_rms):
        decay_ratio = normalized_rms
        self.r *= (0.98 * decay_ratio + 0.02)
        self.alpha *= 0.97
        return self.r > 1 and self.alpha > 5

    def draw(self, surface):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        color_with_alpha = (*self.color.astype(int), int(self.alpha))
        pygame.draw.circle(s, color_with_alpha, (int(self.x), int(self.y)), max(1, int(self.r)))
        surface.blit(s, (0, 0))

# === AUDIO UTILITIES ===
def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def detect_new_onset(signal, sr):
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    return onsets

def random_color():
    return [random.randint(50, 255) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

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
        normalized_rms = min(max(normalized_rms, 0.0), 1.0)

        print(f"🔊 Current RMS: {smoothed_rms:.5f} | Normalized RMS: {normalized_rms:.2f}")

        if smoothed_rms < 0.05:
            background_color = background_color * 0.95 + np.array([255, 255, 255]) * 0.05

        if smoothed_rms > rms_threshold:
            onset_slice = buffer_array[-int(onset_slice_duration * fs):]
            onsets = detect_new_onset(onset_slice, fs)
            if len(onsets) > 0 and (now - last_onset_time > onset_cooldown_sec):
                last_onset_rms = smoothed_rms
                last_onset_time = now
                print(f"🎵 Onset detected! Onset RMS: {smoothed_rms:.5f}")
                target_area = (WIDTH * HEIGHT) / 2
                max_radius = np.sqrt(target_area / np.pi)
                size = normalized_rms * max_radius
                size = int(30 + 200 * min(1.0, smoothed_rms / 0.3))
                color = random_color()
                x, y = random_position()
                circles.append(Circle(x, y, size, color, smoothed_rms))
                if len(circles) > MAX_CIRCLES:
                    circles.pop(0)

        updated_circles = []
        for c in circles:
            if c.update(normalized_rms):
                updated_circles.append(c)
                background_color = background_color * 0.995 + 0.005 * c.color * c.r/100
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