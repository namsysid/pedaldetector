import numpy as np
from scipy.signal import spectrogram, hilbert
import sounddevice as sd

fs = 44100
duration = 5.0

print("🎙️ Recording... play a note (with or without pedal)")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()

def compute_envelope(signal):
    return np.abs(hilbert(signal))

def estimate_t60(envelope, fs):
    env_db = 20 * np.log10(envelope / np.max(envelope) + 1e-10)
    time = np.arange(len(envelope)) / fs
    valid = env_db > -60
    return time[valid][-1] - time[valid][0] if np.sum(valid) > 2 else 0

def normalized_residual_energy(signal, fs):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, nperseg=2048, noverlap=1024)
    total_energy = np.mean(Sxx)
    band_mask = (f >= 100) & (f <= 2500)
    return np.mean(Sxx[band_mask]) / (total_energy + 1e-10)

def detect_pedal(signal, fs):
    env = compute_envelope(signal)
    t60 = estimate_t60(env, fs)
    res_ratio = normalized_residual_energy(signal, fs)

    print(f"\n📊 T60: {t60:.2f}s | Residual (Normalized): {res_ratio:.2%}")
    return "✅ Pedal likely used" if t60 > 2.5 and res_ratio > 0.25 else "❌ No pedal detected"

print(detect_pedal(audio, fs))