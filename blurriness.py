import numpy as np
import sounddevice as sd
import librosa
import time
from collections import deque

fs = 22050
frame_duration = 0.05
frame_samples = int(frame_duration * fs)
buffer_duration = 1.5
buffer_samples = int(buffer_duration * fs)
rms_window_duration = 0.2
rms_window_samples = int(rms_window_duration * fs)
rms_threshold = 0.01
onset_cooldown_sec = 0.5
recent_onset_window = 0.5

audio_buffer = deque(maxlen=buffer_samples)
last_onset_rms = 1e-6
last_onset_time = 0
rms_history = deque(maxlen=3)

print("🎧 Starting responsive live monitor... (Press Ctrl+C to stop)")

def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2)) if len(signal) > 0 else 0.0

def detect_new_onset(signal, sr):
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    return onsets

def audio_callback(indata, frames, time_info, status):
    global last_onset_rms, last_onset_time
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

        if smoothed_rms > rms_threshold:
            onset_slice = buffer_array[-int(recent_onset_window * fs):]
            onsets = detect_new_onset(onset_slice, fs)
            if len(onsets) > 0 and (now - last_onset_time > onset_cooldown_sec):
                last_onset_rms = smoothed_rms
                last_onset_time = now
                print("🎵 Onset detected!")

        normalized_rms = smoothed_rms / (last_onset_rms + 1e-10)
        print(f"🔊 RMS: {smoothed_rms:.5f} | Normalized: {normalized_rms:.2f}")

try:
    with sd.InputStream(channels=1, samplerate=fs, blocksize=frame_samples, callback=audio_callback):
        while True:
            time.sleep(frame_duration)
except KeyboardInterrupt:
    print("\n🛑 Stopped live monitor.")