import numpy as np
import sounddevice as sd
import pygame
import random
import time
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from collections import deque
import math

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
pedal_threshold = -28

# Persistent variables
pedal_state = False
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
PEDAL_HYST_ON_DB   = -30.0
PEDAL_HYST_ON_DUR  = 4.0    # seconds
PEDAL_HYST_OFF_DUR = 0.050  # seconds
interharmonic_rms = 0.0

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

def interharmonic_plot_update(inter_db, *, title="Inter-harmonic RMS (dB)", window_seconds=10.0):
    """
    Real-time scrolling plot for inter-harmonic RMS in dB.
    Call this once per main-loop iteration with your latest 'inter_db' value.
    - No lookahead, non-blocking (uses interactive mode).
    - Keeps the last 'window_seconds' of data visible.
    - Auto-initializes on first call; no separate init required.
    """
    import time
    import numpy as np

    st = interharmonic_plot_update.__dict__
    now = time.time()

    # Lazy init
    if "init" not in st:
        st["init"] = True
        st["t0"] = now
        st["times"] = []   # seconds since first call
        st["vals"] = []    # inter_db values

        # Set up matplotlib (interactive)
        import matplotlib
        matplotlib.use("MacOSX")  # fall back to a common interactive backend; change if you prefer
        import matplotlib.pyplot as plt
        st["plt"] = plt
        plt.ion()
        st["fig"], st["ax"] = plt.subplots(figsize=(9, 3))
        st["line"], = st["ax"].plot([], [], lw=1.8)
        st["ax"].set_title(title)
        st["ax"].set_xlabel("Time (s)")
        st["ax"].set_ylabel("dB")
        st["ax"].grid(True, alpha=0.3)
        st["fig"].canvas.mpl_connect("close_event", lambda ev: st.__setitem__("closed", True))
        st["closed"] = False

    if st.get("closed"):
        return  # window was closed by the user; do nothing

    # Append new sample
    t_rel = now - st["t0"]
    st["times"].append(t_rel)
    st["vals"].append(float(inter_db))

    # Drop old samples outside the window
    t_min = t_rel - float(window_seconds)
    i0 = 0
    # Find first index with time >= t_min (linear scan is fine for typical loop rates)
    while i0 < len(st["times"]) and st["times"][i0] < t_min:
        i0 += 1
    if i0 > 0:
        st["times"] = st["times"][i0:]
        st["vals"] = st["vals"][i0:]

    # Update plot data
    st["line"].set_data(st["times"], st["vals"])

    # X limits: keep last window_seconds visible
    if st["times"]:
        st["ax"].set_xlim(max(0.0, st["times"][0]), st["times"][-1] if st["times"][-1] > window_seconds else window_seconds)
    else:
        st["ax"].set_xlim(0, window_seconds)

    # Y limits: pad around current min/max to avoid cramped view
    if st["vals"]:
        vmin = float(np.min(st["vals"]))
        vmax = float(np.max(st["vals"]))
        pad = max(1.0, 0.1 * max(1.0, vmax - vmin))
        st["ax"].set_ylim(vmin - pad, vmax + pad)

    # Draw without blocking your loop
    st["fig"].canvas.draw_idle()
    st["fig"].canvas.flush_events()

def update_pedal_state_with_timers(
    rel_db,
    pedal_state,
    *,
    # Use YOUR real thresholds here (from your code)
    th_on_db,
    th_off_db,

    # Time constants (ms)
    dwell_ms=500,        # need this long continuously beyond threshold to flip
    gray_hold_ms=500,    # freeze while in gray zone for this long
    attack_ms=80,        # easing toward ON target
    release_ms=160,      # easing toward OFF target

    # Smoothing of input metric (set 0.0 to disable)
    alpha=0.2,

    # IMPORTANT: your float targets (not limited to 0..1)
    # Set these to what your renderer expects for "no pedal" vs "full pedal"
    target_on_value=1.0,
    target_off_value=0.0
):
    """
    Timer/hysteresis pedal updater that returns a FLOAT pedal_state (not clamped to 0..1).

    Inputs:
      rel_db:      your relative cue in dB
      pedal_state: your current float pedal_state

    Behavior:
      - Hysteresis (th_on_db > th_off_db)
      - Dwell: must remain beyond threshold for dwell_ms to flip internal latch
      - Gray-hold: if rel_db is in (th_off_db, th_on_db), freeze state for gray_hold_ms
      - Attack/release easing toward arbitrary float targets you set
    """

    st = update_pedal_state_with_timers.__dict__
    now = time.time()

    # --- init persistent state ---
    if "init" not in st:
        st["init"] = True
        # internal discrete latch (0/1), derived from where the current pedal_state is closer
        st["latched"] = 1 if abs(pedal_state - target_on_value) < abs(pedal_state - target_off_value) else 0
        st["rel_s"] = float(rel_db)   # smoothed metric
        st["cand_on_t"] = None
        st["cand_off_t"] = None
        st["gray_since"] = None
        st["prev_t"] = now

    # --- smooth the metric (optional) ---
    st["rel_s"] = (1 - alpha) * st["rel_s"] + alpha * float(rel_db)
    x = st["rel_s"]

    # --- gray zone hold ---
    in_gray = (x > th_off_db) and (x < th_on_db)
    now = time.time()
    if in_gray:
        if st["gray_since"] is None:
            st["gray_since"] = now
        gray_hold_active = (now - st["gray_since"]) < (gray_hold_ms / 1000.0)
    else:
        st["gray_since"] = None
        gray_hold_active = False

    # --- dwell‑based latch updates (only if not in active gray hold) ---
    if not gray_hold_active:
        # candidate ON
        if st["latched"] == 0 and x >= th_on_db:
            st["cand_on_t"] = st["cand_on_t"] or now
            if (now - st["cand_on_t"]) >= (dwell_ms / 1000.0):
                st["latched"] = 1
                st["cand_on_t"] = None
                st["cand_off_t"] = None
        else:
            st["cand_on_t"] = None

        # candidate OFF
        if st["latched"] == 1 and x <= th_off_db:
            st["cand_off_t"] = st["cand_off_t"] or now
            if (now - st["cand_off_t"]) >= (dwell_ms / 1000.0):
                st["latched"] = 0
                st["cand_off_t"] = None
                st["cand_on_t"] = None
        else:
            st["cand_off_t"] = None

    # --- ease your FLOAT pedal_state toward targets (not clamped 0..1) ---
    dt = max(1e-3, now - st["prev_t"])
    st["prev_t"] = now

    target = target_on_value if st["latched"] == 1 else target_off_value
    # separate attack/release time constants
    tau = (attack_ms / 1000.0) if (target - pedal_state) > 0 else (release_ms / 1000.0)
    k = 1.0 - math.exp(-dt / max(1e-3, tau))
    new_pedal_state = pedal_state + (target - pedal_state) * k

    # Optional debug if you want to log behavior
    dbg = {
        "rel_db_smooth": x,
        "latched": st["latched"],
        "in_gray": in_gray,
        "th_on_db": th_on_db,
        "th_off_db": th_off_db,
        "target": target,
    }

    return new_pedal_state, dbg

def update_pedal_state(current_level_db, global_db):
    """
    Update the global pedal_state using hysteresis with dwell timers.

    Rules:
      • Start with pedal OFF (False / 0.0).
      • If inter-harmonic RMS in dB stays > PEDAL_HYST_ON_DB for PEDAL_HYST_ON_DUR seconds, set pedal True.
      • If pedal is True and inter-harmonic RMS in dB stays <= PEDAL_HYST_ON_DB for PEDAL_HYST_OFF_DUR seconds, set pedal False.
    """
    global pedal_state

    # Plot for debugging (non-blocking)
    try:
        interharmonic_plot_update(current_level_db)
    except Exception:
        pass

    now = time.monotonic()
    st = update_pedal_state.__dict__

    if "state" not in st:
        st["state"] = False  # default OFF
        st["on_start"] = None
        st["off_start"] = None

    above = current_level_db > PEDAL_HYST_ON_DB

    if not st["state"]:
        # Currently OFF → consider turning ON
        if above:
            if st["on_start"] is None:
                st["on_start"] = now
            if (now - st["on_start"]) >= PEDAL_HYST_ON_DUR:
                st["state"] = True
                st["off_start"] = None
        else:
            st["on_start"] = None
    else:
        # Currently ON → consider turning OFF
        if not above:  # ≤ threshold
            if st["off_start"] is None:
                st["off_start"] = now
            if (now - st["off_start"]) >= PEDAL_HYST_OFF_DUR:
                st["state"] = False
                st["on_start"] = None
        else:
            st["off_start"] = None

    # Expose as float for rest of pipeline
    pedal_state = 1.0 if st["state"] else 0.0
    print(pedal_state)
    return pedal_state

def measure_interharmonic_broadband_rms(
    audio_frame,
    sr,
    peaks=None,
    peak_mask_hz=30.0,
    highband_hz=5000.0,
    memory_ms=300.0,
    print_prefix="PEDAL"
):
    """
    Drop-in Level-2 version: same output as before, but the inter-harmonic mask includes
    peaks from the current frame *and* all peaks observed in the last `memory_ms`.
    No other code changes required.

    - Uses only past data (causal), zero-lookahead.
    - Keeps a tiny in-function memory of recent peaks as (freq, tstamp).
    - If `peaks` is provided, it will be merged into the memory; otherwise it calls your
      existing `find_note_peaks(audio_frame, sr)` without modifying it.
    """
    if audio_frame is None or len(audio_frame) == 0:
        return

    # Local imports so you don't have to edit global imports
    import time
    import numpy as np
    from numpy.fft import rfft, rfftfreq

    global interharmonic_rms

    # Initialize a tiny ring buffer on the function itself (no global changes)
    if not hasattr(measure_interharmonic_broadband_rms, "_recent_peaks"):
        from collections import deque
        # store tuples: (freq_hz, timestamp_seconds)
        measure_interharmonic_broadband_rms._recent_peaks = deque()
    recent_peaks = measure_interharmonic_broadband_rms._recent_peaks

    # === FFT ===
    window = np.hanning(len(audio_frame))
    frame = audio_frame * window
    spec = np.abs(rfft(frame))
    freqs = rfftfreq(len(audio_frame), 1.0 / sr)

    # === Global RMS (optional, handy if you want normalization later) ===
    global_rms = float(np.sqrt(np.mean(spec ** 2)))
    global_rms_db = 20.0 * np.log10(global_rms + 1e-12)

    # === Current-frame peaks ===
    if peaks is None:
        try:
            peak_list = find_note_peaks(audio_frame, sr)  # expects list of (freq, mag) or (freq, *)
        except Exception:
            peak_list = []
    else:
        peak_list = peaks

    now = time.time()

    # Append current peaks to memory with timestamps
    for p in peak_list:
        try:
            pf = float(p[0])  # frequency in Hz
        except Exception:
            continue
        recent_peaks.append((pf, now))

    # Prune old peaks outside memory window
    cutoff = now - (memory_ms / 1000.0)
    while recent_peaks and recent_peaks[0][1] < cutoff:
        recent_peaks.popleft()

    # === Build mask for inter-harmonic floor ===
    mask = np.ones_like(spec, dtype=bool)
    mask &= freqs >= 30.0  # drop DC/infra

    # Mask current-frame peaks (defensive if caller passes peaks but you still want both)
    for p in peak_list:
        try:
            pf = float(p[0])
        except Exception:
            continue
        left = pf - peak_mask_hz
        right = pf + peak_mask_hz
        mask &= ~((freqs >= left) & (freqs <= right))

    # Mask all peaks seen in the recent memory window
    # (this is the Level-2 addition over the original function)
    for pf, _t in recent_peaks:
        left = pf - peak_mask_hz
        right = pf + peak_mask_hz
        mask &= ~((freqs >= left) & (freqs <= right))

    # === Inter-harmonic RMS (masked bins) ===
    inter_bins = spec[mask]
    inter_rms = float(np.sqrt(np.mean(inter_bins ** 2))) if inter_bins.size else 0.0

    # === High-band RMS (> highband_hz) ===
    hb_mask = freqs >= highband_hz
    hb_bins = spec[hb_mask]
    highband_rms = float(np.sqrt(np.mean(hb_bins ** 2))) if hb_bins.size else 0.0

    # === Convert to dB ===
    eps = 1e-12
    inter_db = 20.0 * np.log10(inter_rms + eps)
    highband_db = 20.0 * np.log10(highband_rms + eps)
    interharmonic_rms = inter_db
    update_pedal_state(inter_db, global_rms_db)
    # Print (same style as before)
    # print(
    #     f"{print_prefix} interharmonic_rms_db={inter_db:.2f}  "
    #     f"highband_rms_db={highband_db:.2f}  "
    #     f"global_rms_db={global_rms_db:.2f}"
    # )

    # Optionally return values for downstream use
    # return {
    #     "inter_db": inter_db,
    #     "highband_db": highband_db,
    #     "global_rms_db": global_rms_db,
    #     "num_recent_peaks": len(recent_peaks),
    # }


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
        global background_color, normalized_global_rms, max_recent, pedal_state, interharmonic_rms

        # Track live amp for this MIDI (fallback slight decay)
        self.amp = next((amp for freq, amp in fft_peaks if freq_to_midi(freq) == self.midi), self.amp * 0.98)

        # Keep radius tied to per-note normalized amp (against that note's last onset amp)
        normalized_note = self.amp / (last_onset_amp_by_midi.get(self.midi, self.amp) + 1e-6)
        self.r = min(250, float(self.amp) * 3000.0)
        glob_normalized_note = self.amp / (max_recent + 1e-6)
        if max_recent < 0.01:
            glob_normalized_note = 0.0
        # if self.midi == 69:
        #     print(glob_normalized_note)
        # NEW: bleed on pedal state
        if pedal_state:
            background_color += 0.00000009 * (interharmonic_rms + 35.0) * (self.color - background_color)

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
            memory_ms=300.0,              # Level-2 addition: keep masking peaks seen in last 300 ms
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

        # pygame.display.flip()
        clock.tick(FPS)