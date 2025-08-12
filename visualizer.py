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
bleed_rate = 0.007
alpha = 0.2               # smoothing factor for EMA (~0.2 = ~200 ms smoothing at 25 Hz updates)
# th_on = -28.0              # dB threshold to turn ON
# th_off = -32.0             # dB threshold to turn OFF
# min_time_in_state = 0.15   # seconds to require before switching
pedal_threshold = -30

# Persistent variables
pedal_state = 0
smoothed_level = None

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

def find_note_peaks(
    signal: np.ndarray,
    sample_rate: float,
    min_freq: float = 30.0,
    max_freq: float = 4200.0,
    min_prominence_db: float = 18.0,
    max_peaks: int = 24,
):
    """
    Return a list of (frequency_Hz, amp_rms) peaks for the current frame.

    amp_rms is a window- and FFT-scaled RMS-like amplitude per bin, directly
    comparable in scale to time-domain RMS over the same frame.

    Key normalization (NumPy FFT):
        power[k] = |X[k]|^2 / (N * sum(hann^2))
        amp_rms[k] = sqrt(power[k])

    Peak picking is band-limited and uses a median-noise-floor prominence test.
    """
    x = np.asarray(signal)
    N = int(x.shape[0])
    if N < 4:
        return []

    # Windowed RFFT
    w = np.hanning(N)
    X = np.fft.rfft(x * w)
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)

    # Correct power normalization for NumPy's FFT scaling:
    # Parseval -> sum|y[n]|^2 = (1/N) sum|Y[k]|^2
    # Using window y = x*w: normalize by N * sum(w^2)
    W = float(np.sum(w * w)) + 1e-20
    power = (np.abs(X) ** 2) / (N * W)

    # Per-bin RMS-like amplitude
    amp_rms = np.sqrt(power + 1e-30)

    # Band-limit
    band_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(band_mask):
        return []
    f = freqs[band_mask]
    a = amp_rms[band_mask]
    p = power[band_mask]
    if a.size < 3:
        return []

    # Local-maximum peak detection (no extra deps)
    left = a[1:-1] > a[:-2]
    right = a[1:-1] >= a[2:]
    idx = np.nonzero(left & right)[0] + 1
    if idx.size == 0:
        return []

    # Prominence vs median noise floor (in dB, using power)
    noise_floor = np.median(p) + 1e-30
    prom_db = 10.0 * np.log10(p[idx] / noise_floor + 1e-30)
    keep = prom_db >= float(min_prominence_db)
    if not np.any(keep):
        return []
    idx = idx[keep]

    # Top-N by power, then sort by amplitude desc
    if idx.size > max_peaks:
        top = np.argpartition(-p[idx], max_peaks - 1)[:max_peaks]
        idx = idx[top]
    order = np.argsort(-a[idx])
    idx = idx[order]

    # Return (freq_hz, amp_rms)
    return [(float(f[i]), float(a[i])) for i in idx]

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def random_color():
    return [random.randint(0, 100) for _ in range(3)]

def random_position():
    return random.randint(0, WIDTH), random.randint(0, HEIGHT)

# Time tracking
last_switch_time = time.time()

def update_pedal_state(current_level_db, global_db):
    global pedal_state, smoothed_level, pedal_threshold

    now = time.time()

    # Initialize smoother
    if smoothed_level is None:
        smoothed_level = current_level_db

    # Exponential moving average smoothing
    smoothed_level = (1 - alpha) * smoothed_level + alpha * current_level_db
    relative_level = smoothed_level - global_db
    if (global_db < -26):
        relative_level = global_db - 26
    # if smoothed_level > -40:
    #     print(relative_level)
    if smoothed_level > -35:
        pedal_state = max(0, relative_level - pedal_threshold)
    else:
        pedal_state = 0
    
    print(smoothed_level, global_db)

    # State machine with hysteresis
    # if not pedal_state:
    #     # Currently OFF, check if we should turn ON
    #     if smoothed_level > th_on and (now - last_switch_time) > min_time_in_state:
    #         pedal_state = True
    #         last_switch_time = now
    #         print("PEDAL: ON")
    # else:
    #     # Currently ON, check if we should turn OFF
    #     if smoothed_level < th_off and (now - last_switch_time) > min_time_in_state:
    #         pedal_state = False
    #         last_switch_time = now
    #         print("PEDAL: OFF")

    # return pedal_state, smoothed_level

def measure_interharmonic_broadband_rms(audio_frame, sr, peaks=None,
                                        peak_mask_hz=30.0, highband_hz=5000.0,
                                        print_prefix="PEDAL"):
    """
    Compute & print inter-harmonic (peak-masked) RMS and high-band RMS from a single audio frame.
    - Causal (uses only the given frame).
    - Inter-harmonic floor: masks ±peak_mask_hz around detected spectral peaks.
    - High-band RMS: RMS above highband_hz (e.g., >5 kHz for undamped-string hiss).
    """
    if audio_frame is None or len(audio_frame) == 0:
        return

    # Window + spectrum
    window = np.hanning(len(audio_frame))
    frame = audio_frame * window
    spec = np.abs(rfft(frame))
    freqs = rfftfreq(len(audio_frame), 1.0 / sr)
    global_rms = float(np.sqrt(np.mean(spec ** 2)))
    global_rms_db = 20.0 * np.log10(global_rms + 1e-12)

    # Get peaks (use existing function if peaks not supplied)
    if peaks is None:
        try:
            peak_list = find_note_peaks(audio_frame, sr)  # expects list of (freq, mag)
        except Exception:
            peak_list = []
    else:
        peak_list = peaks

    # Build mask for "inter-harmonic" regions (exclude bins near peaks)
    mask = np.ones_like(spec, dtype=bool)

    # avoid DC/ultra-low
    mask &= freqs >= 30.0

    # exclude ±peak_mask_hz around each detected peak frequency
    for (pf, _pmag) in peak_list:
        left = pf - peak_mask_hz
        right = pf + peak_mask_hz
        mask &= ~((freqs >= left) & (freqs <= right))

    # Inter-harmonic RMS (masked bins)
    inter_bins = spec[mask]
    inter_rms = float(np.sqrt(np.mean(inter_bins ** 2))) if inter_bins.size else 0.0

    # High-band RMS (e.g., >5 kHz) — broadband cue for undamped strings
    hb_mask = freqs >= highband_hz
    hb_bins = spec[hb_mask]
    highband_rms = float(np.sqrt(np.mean(hb_bins ** 2))) if hb_bins.size else 0.0

    # Print in dB for easy thresholding (relative scale)
    eps = 1e-12
    inter_db = 20.0 * np.log10(inter_rms + eps)
    highband_db = 20.0 * np.log10(highband_rms + eps)
    update_pedal_state(inter_db, global_rms_db)
    # print(f"{print_prefix} interharmonic_rms_db={inter_db:.2f}  highband_rms_db={highband_db:.2f}")

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
        global background_color, normalized_global_rms, max_recent, pedal_state

        # Track live amp for this MIDI (fallback slight decay)
        self.amp = next((amp for freq, amp in fft_peaks if freq_to_midi(freq) == self.midi), self.amp * 0.98)

        # Keep radius tied to per-note normalized amp (against that note's last onset amp)
        normalized_note = self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6)
        self.r = min(250, float(self.amp) * 5000.0)
        glob_normalized_note = self.amp / (max_recent + 1e-6)
        if max_recent < 0.01:
            glob_normalized_note = 0.0
        # if self.midi == 69:
        #     print(glob_normalized_note)
        # NEW: bleed on pedal state
        background_color += 0.007 * pedal_state * (self.color - background_color)

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
    if (max_recent < 0.01):
        normalized_global_rms = 0.0
    peaks = find_note_peaks(audio_buffer[-fft_window:], fs)
    if not peaks:
        return

    # Only trigger the loudest note every 100ms
    if now - last_loudest_time >= loudest_note_interval:
        freq, amp = max(peaks, key=lambda p: p[1])
        if amp < 0.001:
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

        # print(normalized_global_rms)
        if normalized_global_rms > 0.007:
            background_color += 0.085 * (255 - background_color)
        else:
            background_color = 255.0

        fft_peaks = find_note_peaks(audio_buffer[-fft_window:], fs)
        measure_interharmonic_broadband_rms(
            audio_frame=audio_buffer[-fft_window:], 
            sr=fs, 
            peaks=fft_peaks,              # optional; pass None to let the function call find_note_peaks itself
            peak_mask_hz=30.0,            # tweak: 20–40 Hz works well
            highband_hz=5000.0,           # tweak: 5–6 kHz at 22.05 kHz SR
            print_prefix="PEDAL"
        )
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