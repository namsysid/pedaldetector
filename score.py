import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt

# === Configuration ===
fs = 22050
duration = 10.0
fft_window = 2048
hop_size = 512
target_freq = 110
bandwidth = 10

# === Derivative Score Function ===
def compute_positive_derivative_sum(rms_curve):
    peak = np.max(rms_curve)
    if peak == 0:
        return 0.0, np.zeros_like(rms_curve)
    norm_rms = rms_curve / peak
    deriv = np.diff(norm_rms, prepend=norm_rms[0])
    positive_deriv = np.where(deriv > 0, deriv, 0)
    score = np.sum(positive_deriv)
    return score, norm_rms

# === Record Audio ===
print(f"🎙️ Recording {duration:.1f} seconds of audio...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = audio.flatten()
print("✅ Recording complete.")

# === Extract RMS ===
rms_trace = []
freqs = rfftfreq(fft_window, 1 / fs)
target_bins = np.where((freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth))[0]

for i in range(0, len(audio) - fft_window, hop_size):
    frame = audio[i:i + fft_window] * np.hanning(fft_window)
    spectrum = np.abs(rfft(frame))
    band_rms = np.sqrt(np.mean(spectrum[target_bins] ** 2))
    rms_trace.append(band_rms)

# === Compute Score and Plot ===
rms_trace = np.array(rms_trace)
score, norm_rms = compute_positive_derivative_sum(rms_trace)
time_axis = np.arange(len(norm_rms)) * hop_size / fs

print(f"\n📈 Total Positive Derivative Sum: {score:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(time_axis, norm_rms, label="Normalized RMS (440 Hz)", color='blue')

# Highlight rising segments
deriv = np.diff(norm_rms, prepend=norm_rms[0])
plt.fill_between(time_axis, norm_rms, where=(deriv > 0), color='orange', alpha=0.4, label="Positive Slope")

plt.title(f"RMS Decay and Positive Derivative Score = {score:.4f}")
plt.xlabel("Time (s)")
plt.ylabel("Normalized RMS")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()