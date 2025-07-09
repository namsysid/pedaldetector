import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

# === CONFIGURATION ===
fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
fft_window = 2048
peak_threshold = 0.3
cooldown_time = 0.5  # seconds between onsets per note
decay_model_A = 39.77
decay_model_B = -41.52

WIDTH, HEIGHT = 800, 600
FPS = 30
MAX_CIRCLES = 50

# === STATE ===
audio_buffer = np.zeros(fft_window)
last_onset_time = {}
circles = []

# === INIT PYGAME ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Piano Visualizer (FFT-based)")
clock = pygame.time.Clock()
background_color = np.array([255, 255, 255], dtype=np.float32)

# === UTILITY FUNCTIONS ===
def midi_to_freq(midi):
    return 440 * 2 ** ((midi - 69) / 12)

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
    detected = []

    for i in peaks:
        f = freqs[i]
        amp = spectrum[i]
        if 30 < f < 4200:  # piano range
            detected.append((f, amp))

    return detected

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
        self.freq = freq
        self.decay = compute_decay_multiplier(freq)
        self.initial_amp = amp

    def update(self):
        self.r *= self.decay
        self.alpha *= 0.97
        return self.r > 1 and self.alpha > 5

    def draw(self, surface):
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        color_with_alpha = (*self.color.astype(int), int(self.alpha))
        pygame.draw.circle(s, color_with_alpha, (int(self.x), int(self.y)), max(1, int(self.r)))
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
    freqs_amplitudes = find_note_peaks(audio_buffer, fs)

    for freq, amp in freqs_amplitudes:
        midi = freq_to_midi(freq)
        last_time = last_onset_time.get(midi, 0)

        if now - last_time > cooldown_time:
            print(f"🎵 Note: {freq:.1f} Hz (MIDI {midi}) | Amp: {amp:.3f}")
            max_area = (WIDTH * HEIGHT) / 2
            max_radius = int(np.sqrt(max_area / np.pi))
            size = min(max_radius, int(30 + 200 * min(1.0, amp)))
            color = random_color()
            x, y = random_position()
            circles.append(Circle(x, y, size, color, freq, amp))
            if len(circles) > MAX_CIRCLES:
                circles.pop(0)
            last_onset_time[midi] = now

    updated = []
    for c in circles:
        if c.update():
            updated.append(c)
            background_color[:] = background_color * 0.995 + 0.005 * c.color
    circles[:] = updated

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
    print("\n🛑 Visualizer stopped.")
    pygame.quit()