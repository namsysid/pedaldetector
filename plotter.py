import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Configuration
fs = 22050
duration = 10.0  # seconds
fft_window = 2048
hop_size = 512
track_freq = 110  # Hz
freq_band_width = 10  # Hz range around the frequency

# Record audio
print("🎙️  Recording for 10 seconds...")
audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("✅ Done recording.")

# Flatten audio and prepare analysis
audio = audio.flatten()
freqs = rfftfreq(fft_window, 1 / fs)
target_bins = np.where((freqs >= track_freq - freq_band_width) & (freqs <= track_freq + freq_band_width))[0]

rms_trace = []
for i in range(0, len(audio) - fft_window, hop_size):
    frame = audio[i:i + fft_window] * np.hanning(fft_window)
    spectrum = np.abs(rfft(frame))
    band_rms = np.sqrt(np.mean(spectrum[target_bins] ** 2))
    rms_trace.append(band_rms)

# Plot
times = np.arange(len(rms_trace)) * (hop_size / fs)
plt.figure(figsize=(10, 4))
plt.plot(times, rms_trace, label=f'RMS near {track_freq} Hz')
plt.xlabel("Time (s)")
plt.ylabel("RMS Amplitude")
plt.title(f"RMS Decay at ~{track_freq} Hz")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()