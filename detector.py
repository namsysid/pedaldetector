from collections import deque
import time

import numpy as np


class PedalDetector:
    def __init__(
        self,
        sample_rate=48000,
        frame_duration=0.05,
        fft_window=2048,
        buffer_seconds=2.0,
        global_rms_window_sec=10.0,
        cooldown_time=0.25,
        loudest_note_interval=0.1,
        pedal_hyst_on_db=-30.0,
        pedal_hyst_on_dur=2.0,
        pedal_hyst_off_dur=0.01,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_duration = float(frame_duration)
        self.fft_window = int(fft_window)
        self.buffer_samples = max(self.fft_window, int(buffer_seconds * self.sample_rate))
        self.onset_window = max(self.fft_window, int(0.5 * self.sample_rate))
        self.cooldown_time = float(cooldown_time)
        self.loudest_note_interval = float(loudest_note_interval)
        self.pedal_hyst_on_db = float(pedal_hyst_on_db)
        self.pedal_hyst_on_dur = float(pedal_hyst_on_dur)
        self.pedal_hyst_off_dur = float(pedal_hyst_off_dur)

        rms_history_size = max(1, int(global_rms_window_sec / frame_duration))
        self.recent_global_rms = deque(maxlen=rms_history_size)
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self.recent_peaks = deque()

        self.last_onset_time = {}
        self.color_by_midi = {}
        self.last_onset_amp_by_midi = {}
        self.last_loudest_time = 0.0
        self.pedal_state = 0.0
        self.on_start = None
        self.off_start = None

    def process_samples(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return self._empty_result()

        if samples.size >= self.buffer_samples:
            self.audio_buffer = samples[-self.buffer_samples :].copy()
        else:
            self.audio_buffer = np.roll(self.audio_buffer, -samples.size)
            self.audio_buffer[-samples.size :] = samples

        frame = self.audio_buffer[-self.fft_window :]
        peaks = self.find_note_peaks(frame, self.sample_rate)
        current_rms = self.compute_rms(self.audio_buffer[-self.onset_window :])
        self.recent_global_rms.append(current_rms)
        max_recent = max(self.recent_global_rms) if self.recent_global_rms else current_rms + 1e-6
        normalized_global_rms = current_rms / (max_recent + 1e-6)
        if max_recent < 0.01:
            normalized_global_rms = 0.0

        onset = self._detect_loudest_onset(peaks, current_rms)
        interharmonic = self.measure_interharmonic_broadband_rms(frame, self.sample_rate, peaks)

        return {
            "pedal_state": self.pedal_state,
            "pedal_on": bool(self.pedal_state >= 0.5),
            "inter_db": interharmonic["inter_db"],
            "highband_db": interharmonic["highband_db"],
            "global_rms_db": interharmonic["global_rms_db"],
            "global_rms": current_rms,
            "normalized_global_rms": normalized_global_rms,
            "peaks": [
                {"freq": freq, "amp": amp, "midi": self.freq_to_midi(freq)}
                for freq, amp in peaks[:12]
            ],
            "onset": onset,
            "sample_rate": self.sample_rate,
        }

    def _empty_result(self):
        return {
            "pedal_state": self.pedal_state,
            "pedal_on": bool(self.pedal_state >= 0.5),
            "inter_db": -120.0,
            "highband_db": -120.0,
            "global_rms_db": -120.0,
            "global_rms": 0.0,
            "normalized_global_rms": 0.0,
            "peaks": [],
            "onset": None,
            "sample_rate": self.sample_rate,
        }

    def _detect_loudest_onset(self, peaks, current_rms):
        if not peaks:
            return None

        now = time.time()
        if now - self.last_loudest_time < self.loudest_note_interval:
            return None

        freq, amp = max(peaks, key=lambda peak: peak[1])
        if amp < 0.001:
            return None

        midi = self.freq_to_midi(freq)
        last_time = self.last_onset_time.get(midi, 0.0)
        if now - last_time <= self.cooldown_time:
            return None

        if midi not in self.color_by_midi:
            self.color_by_midi[midi] = self._color_for_midi(midi)

        self.last_onset_amp_by_midi[midi] = amp
        self.last_onset_time[midi] = now
        self.last_loudest_time = now

        return {
            "freq": freq,
            "amp": amp,
            "midi": midi,
            "color": self.color_by_midi[midi],
            "r": min(250.0, float(amp) * 3000.0),
        }

    def measure_interharmonic_broadband_rms(
        self,
        audio_frame,
        sample_rate,
        peaks,
        peak_mask_hz=30.0,
        highband_hz=5000.0,
        memory_ms=300.0,
    ):
        if audio_frame is None or len(audio_frame) == 0:
            self._update_pedal_state(-120.0)
            return {"inter_db": -120.0, "highband_db": -120.0, "global_rms_db": -120.0}

        window = np.hanning(len(audio_frame))
        frame = audio_frame * window
        spec = np.abs(np.fft.rfft(frame))
        freqs = np.fft.rfftfreq(len(audio_frame), 1.0 / sample_rate)

        global_rms = float(np.sqrt(np.mean(spec**2)))
        global_rms_db = 20.0 * np.log10(global_rms + 1e-12)

        now = time.time()
        for peak in peaks:
            try:
                self.recent_peaks.append((float(peak[0]), now))
            except (TypeError, ValueError):
                continue

        cutoff = now - (memory_ms / 1000.0)
        while self.recent_peaks and self.recent_peaks[0][1] < cutoff:
            self.recent_peaks.popleft()

        mask = np.ones_like(spec, dtype=bool)
        mask &= freqs >= 30.0

        for peak_freq, _timestamp in self.recent_peaks:
            for harmonic in range(1, 5):
                harmonic_freq = harmonic * peak_freq
                if harmonic_freq >= highband_hz:
                    break
                left = harmonic_freq - peak_mask_hz
                right = harmonic_freq + peak_mask_hz
                mask &= ~((freqs >= left) & (freqs <= right))

        inter_bins = spec[mask]
        inter_rms = float(np.sqrt(np.mean(inter_bins**2))) if inter_bins.size else 0.0

        hb_mask = freqs >= highband_hz
        hb_bins = spec[hb_mask]
        highband_rms = float(np.sqrt(np.mean(hb_bins**2))) if hb_bins.size else 0.0

        inter_db = 20.0 * np.log10(inter_rms + 1e-12)
        highband_db = 20.0 * np.log10(highband_rms + 1e-12)
        self._update_pedal_state(inter_db)

        return {
            "inter_db": inter_db,
            "highband_db": highband_db,
            "global_rms_db": global_rms_db,
        }

    def _update_pedal_state(self, current_level_db):
        now = time.monotonic()
        above = current_level_db > self.pedal_hyst_on_db

        if self.pedal_state < 0.5:
            if above:
                if self.on_start is None:
                    self.on_start = now
                if now - self.on_start >= self.pedal_hyst_on_dur:
                    self.pedal_state = 1.0
                    self.off_start = None
            else:
                self.on_start = None
        else:
            if not above:
                if self.off_start is None:
                    self.off_start = now
                if now - self.off_start >= self.pedal_hyst_off_dur:
                    self.pedal_state = 0.0
                    self.on_start = None
            else:
                self.off_start = None

    @staticmethod
    def find_note_peaks(
        signal,
        sample_rate,
        min_freq=30.0,
        max_freq=4200.0,
        min_prominence_db=18.0,
        max_peaks=24,
    ):
        x = np.asarray(signal)
        sample_count = int(x.shape[0])
        if sample_count < 4:
            return []

        window = np.hanning(sample_count)
        spectrum = np.fft.rfft(x * window)
        freqs = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate)
        window_power = float(np.sum(window * window)) + 1e-20
        power = (np.abs(spectrum) ** 2) / (sample_count * window_power)
        amp_rms = np.sqrt(power + 1e-30)

        band_mask = (freqs >= min_freq) & (freqs <= max_freq)
        if not np.any(band_mask):
            return []

        band_freqs = freqs[band_mask]
        band_amp = amp_rms[band_mask]
        band_power = power[band_mask]
        if band_amp.size < 3:
            return []

        left = band_amp[1:-1] > band_amp[:-2]
        right = band_amp[1:-1] >= band_amp[2:]
        peak_idx = np.nonzero(left & right)[0] + 1
        if peak_idx.size == 0:
            return []

        noise_floor = np.median(band_power) + 1e-30
        prom_db = 10.0 * np.log10(band_power[peak_idx] / noise_floor + 1e-30)
        peak_idx = peak_idx[prom_db >= float(min_prominence_db)]
        if peak_idx.size == 0:
            return []

        if peak_idx.size > max_peaks:
            top = np.argpartition(-band_power[peak_idx], max_peaks - 1)[:max_peaks]
            peak_idx = peak_idx[top]

        order = np.argsort(-band_amp[peak_idx])
        peak_idx = peak_idx[order]
        return [(float(band_freqs[i]), float(band_amp[i])) for i in peak_idx]

    @staticmethod
    def compute_rms(signal):
        return float(np.sqrt(np.mean(signal**2))) if len(signal) > 0 else 0.0

    @staticmethod
    def freq_to_midi(freq):
        return int(round(69 + 12 * np.log2(freq / 440.0)))

    @staticmethod
    def _color_for_midi(midi):
        hue = (midi * 37) % 360
        return _hsl_to_rgb(hue, 0.62, 0.42)


def _hsl_to_rgb(hue, saturation, lightness):
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    x_val = chroma * (1 - abs((hue / 60) % 2 - 1))
    match = lightness - chroma / 2

    if hue < 60:
        red, green, blue = chroma, x_val, 0
    elif hue < 120:
        red, green, blue = x_val, chroma, 0
    elif hue < 180:
        red, green, blue = 0, chroma, x_val
    elif hue < 240:
        red, green, blue = 0, x_val, chroma
    elif hue < 300:
        red, green, blue = x_val, 0, chroma
    else:
        red, green, blue = chroma, 0, x_val

    return [
        int((red + match) * 255),
        int((green + match) * 255),
        int((blue + match) * 255),
    ]
