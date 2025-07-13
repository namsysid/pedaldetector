import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CONFIGURATION ===
fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
fft_window = 2048
peak_threshold = 0.3
cooldown_time = 0.25
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
detected_list = []

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

def is_beat_modulated(amps, last_onset_time, now, midi, tau=0.75, threshold=0.0):
    if len(amps) < 2:
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
        + 1.5931 * delta +
        + 0.36687 * slope +
        - 2.20309 * mi +
        - 0.8063 * cooldown_penalty
    )
    if (score < threshold):
        print("Rejected: " + str(now) + " " + str(midi) + " " + str(ys[-1]) + " --> " + str(score))
    else:
        print("Accepted: " + str(now) + " " + str(midi) + " " + str(ys[-1]) + " --> " + str(score))

    return score < threshold

def random_color():
    return [random.randint(50, 255) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

#INTERNAL FUNCTIONS
def visualize_modulation(midi_note=69):
    midi_note = 69  # A4 or choose any note you're tracking
    window = 30     # Number of points to display in the plot
    rms_window = 5  # For RMS smoothing

    fig, ax = plt.subplots()
    line_amp, = ax.plot([], [], label="Amplitude (ys)", marker='o')
    line_rms, = ax.plot([], [], label=f"{rms_window}-pt RMS", linestyle='--')
    ax.set_title(f"Amplitude and RMS for MIDI {midi_note}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(0, 1)  # Adjust based on your signal range

    def init():
        line_amp.set_data([], [])
        line_rms.set_data([], [])
        return line_amp, line_rms

    def update(frame):
        if midi_note not in recent_amplitudes:
            return line_amp, line_rms

        ts_ys = recent_amplitudes[midi_note][-window:]
        if len(ts_ys) < 2:
            return line_amp, line_rms

        ts, ys = zip(*ts_ys)
        ts = np.array(ts)
        ys = np.array(ys)

        line_amp.set_data(ts, ys)

        if len(ys) >= rms_window:
            rms = np.sqrt(np.convolve(np.square(ys), np.ones(rms_window)/rms_window, mode='valid'))
            rms_ts = ts[rms_window//2 : -(rms_window//2) or None]
            line_rms.set_data(rms_ts, rms)
        else:
            line_rms.set_data([], [])

        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(0, max(1.1 * max(ys), 0.1))  # dynamic range
        return line_amp, line_rms

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=False, interval=100)
    plt.tight_layout()
    plt.show()

# === CIRCLE CLASS ===
class Circle:
    def __init__(self, x, y, r, color, freq, amp):
        self.x = x
        self.y = y
        self.r = r
        self.color = np.array(color, dtype=np.float32)
        self.alpha = 255
        self.decay = compute_decay_multiplier(freq)
        self.freq = freq
        self.midi = freq_to_midi(freq)

    def update(self):

        # RMS-based alpha decay
        if self.midi in recent_amplitudes and len(recent_amplitudes[self.midi]) >= 3:
            _, amps = zip(*recent_amplitudes[self.midi])
            amps = np.array(amps[-5:])  # take up to 5 recent points
            rms = np.sqrt(np.mean(np.square(amps)))
            decay_rate = np.clip(0.9 + (1 - rms) * 0.1, 0.90, 0.999)  # slower fade if RMS is high
        else:
            decay_rate = 1  # default if not enough data

        # self.alpha *= decay_rate
        self.r *= self.decay * decay_rate
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
    global circles, background_color, detected_list
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
        if is_beat_modulated(recent_amplitudes[midi], now - last_onset_time.get(midi, 0), now, midi):
            continue

        detected_list.append(now)

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
            # visualize_modulation(69)
            screen.fill(background_color.astype(int))
            for c in circles:
                c.draw(screen)
            pygame.display.flip()
            clock.tick(FPS)
            # print(detected_list)
except KeyboardInterrupt:
    pygame.quit()
    print(recent_amplitudes.get(69, 0))
    print("🛑 Visualizer stopped.")