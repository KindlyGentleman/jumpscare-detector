"""Physics-based acoustic anomaly, speech rejection, and transient shock detection."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt
from jumpscare_detector.audio_transient import compute_tkeo


def compute_a_weighting(frequencies: np.ndarray) -> np.ndarray:
    """Calculate IEC 61672:2003 A-weighting gain curve for human ear sensitivity.

    A-weighting emphasizes frequencies in the 1 kHz to 5 kHz range (human ear
    canal acoustic resonance) where screams, glass shatters, and percussive stingers occur.
    """
    f = np.maximum(frequencies, 1e-4)
    f2 = f ** 2
    r_a = (12194.0 ** 2 * f ** 4) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    # Normalized so 1000 Hz has unity gain (0 dB)
    r_1000 = 0.7943411775965269
    gain = r_a / r_1000
    return np.clip(gain, 0.0, 2.0)


@dataclass
class PhysicsConfig:
    """Hyperparameters for acoustic anomaly, dual-mode shock, and speech rejection."""
    sample_rate: int = 16000
    frame_duration_ms: float = 25.0
    hop_duration_ms: float = 10.0
    baseline_window_sec: float = 2.5
    baseline_lag_sec: float = 0.15
    min_contrast_db: float = 16.0
    min_attack_jerk: float = 6.0
    min_jerk_db_per_sec: float = 350.0
    min_peak_rms: float = 0.20
    min_audible_dbfs: float = -42.0
    normalize_headroom: bool = True
    shock_threshold: float = 0.52
    high_freq_cutoff_hz: float = 2200.0
    use_a_weighting: bool = True
    speech_suppression_weight: float = 0.95
    speech_hold_time_sec: float = 0.35
    cooldown_sec: float = 1.0
    warning_lead_time_sec: float = 2.0


@dataclass
class AudioPhysicsFeatures:
    """Physics-derived acoustic descriptors across discrete time frames."""
    timestamps: np.ndarray
    rms_energy: np.ndarray
    acoustic_jerk: np.ndarray
    baseline_energy: np.ndarray
    dynamic_ratio_db: np.ndarray
    spectral_centroid: np.ndarray
    spectral_flux: np.ndarray
    spectral_flatness: np.ndarray
    pitch_periodicity: np.ndarray
    speech_confidence: np.ndarray
    high_freq_ratio: np.ndarray
    crest_factor: np.ndarray
    shock_score: np.ndarray
    tkeo_peak: np.ndarray
    roughness_score: np.ndarray


@dataclass
class AudioShockEvent:
    """Detected acoustic anomaly event with timestamp and physical measurements."""
    timestamp: float
    duration: float
    peak_score: float
    max_rms: float
    jerk_value: float
    dynamic_ratio_db: float
    spectral_centroid_hz: float
    spectral_flatness: float
    pitch_periodicity: float
    speech_confidence: float
    crest_factor: float
    tkeo_peak: float
    roughness_score: float
    severity: str


class AudioPhysicsDetector:
    """Detects jumpscares and acoustic shocks while rejecting conversational speech.

    Physical Principles:
    1. Instantaneous Acoustic Energy (RMS & dBFS): Sound pressure level over 25 ms.
    2. Linear Acoustic Jerk (dE/dt) and Attack Memory: Leading attack acceleration
       over a causal 70 ms window. Shocks exhibit rapid energy rise (dE/dt >= 6 s^-1).
    3. Lagged Ambient Quiescence Contrast: Dynamic range ratio comparing transient
       peak volume against a lagged baseline of pre-shock tension (excluding onset).
    4. Psychoacoustic Spectral Centroid: Center of mass of the frequency spectrum
       shifting into human ear canal resonance and scream bands (2.3 kHz to 5 kHz).
    5. Half-wave Rectified Spectral Flux: Novelty detection tracking sudden
       spectral energy dispersion across frequency bins.
    6. Crest Factor: Peak-to-RMS ratio identifying impulsive acoustic waves.
    7. Conversational Speech Invariant Filter: Normalized harmonic product of
       pitch periodicity, low centroid, and Wiener spectral flatness with a 350 ms
       VAD leaky integrator hangover to reject human conversation while bypassing
       screams and colossal explosive stingers.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        baseline_window_sec: float = 2.5,
        baseline_lag_sec: float = 0.15,
        min_contrast_db: float = 16.0,
        min_peak_rms: float = 0.22,
        shock_threshold: float = 0.52,
        high_freq_cutoff_hz: float = 2200.0,
        use_a_weighting: bool = True,
        speech_suppression_weight: float = 0.95,
        config: Optional[PhysicsConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = PhysicsConfig(
                sample_rate=sample_rate,
                frame_duration_ms=frame_duration_ms,
                hop_duration_ms=hop_duration_ms,
                baseline_window_sec=baseline_window_sec,
                baseline_lag_sec=baseline_lag_sec,
                min_contrast_db=min_contrast_db,
                min_peak_rms=min_peak_rms,
                shock_threshold=shock_threshold,
                high_freq_cutoff_hz=high_freq_cutoff_hz,
                use_a_weighting=use_a_weighting,
                speech_suppression_weight=speech_suppression_weight,
            )

        self.sample_rate = self.config.sample_rate
        self.frame_length = int(self.sample_rate * (self.config.frame_duration_ms / 1000.0))
        self.hop_length = int(self.sample_rate * (self.config.hop_duration_ms / 1000.0))
        self.baseline_frames = max(1, int(self.config.baseline_window_sec / (self.config.hop_duration_ms / 1000.0)))
        self.lag_frames = max(1, int(self.config.baseline_lag_sec / (self.config.hop_duration_ms / 1000.0)))
        self.min_contrast_db = self.config.min_contrast_db
        self.min_jerk_db_per_sec = self.config.min_jerk_db_per_sec
        self.shock_threshold = self.config.shock_threshold
        self.high_freq_cutoff_hz = self.config.high_freq_cutoff_hz
        self.use_a_weighting = self.config.use_a_weighting
        self.speech_suppression_weight = self.config.speech_suppression_weight
        self.window = np.hanning(self.frame_length)

        # Precalculate frequency bins and A-weighting response
        self.freq_bins = np.fft.rfftfreq(self.frame_length, d=1.0 / self.sample_rate)
        if self.use_a_weighting:
            self.a_weights = compute_a_weighting(self.freq_bins)
        else:
            self.a_weights = np.ones_like(self.freq_bins)

        # Human speech pitch fundamental range: 80 Hz to 350 Hz
        self.pitch_lag_min = max(1, int(self.sample_rate / 350.0))
        self.pitch_lag_max = min(self.frame_length - 1, int(self.sample_rate / 80.0))

    def extract_features(self, audio: np.ndarray) -> AudioPhysicsFeatures:
        """Extract physical acoustic descriptors from raw mono PCM audio."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float64)
        num_samples = len(audio)

        if num_samples < self.frame_length:
            empty = np.array([], dtype=np.float64)
            return AudioPhysicsFeatures(
                timestamps=empty,
                rms_energy=empty,
                acoustic_jerk=empty,
                baseline_energy=empty,
                dynamic_ratio_db=empty,
                spectral_centroid=empty,
                spectral_flux=empty,
                spectral_flatness=empty,
                pitch_periodicity=empty,
                speech_confidence=empty,
                high_freq_ratio=empty,
                crest_factor=empty,
                shock_score=empty,
                tkeo_peak=empty,
                roughness_score=empty,
            )

        num_frames = 1 + (num_samples - self.frame_length) // self.hop_length
        shape = (num_frames, self.frame_length)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)

        dt = self.hop_length / float(self.sample_rate)
        timestamps = np.arange(num_frames, dtype=np.float64) * dt

        # Instantaneous RMS acoustic energy and decibel level
        squared = frames ** 2
        rms = np.sqrt(np.mean(squared, axis=1) + 1e-12)
        eps = 1e-6
        decibel_level = 20.0 * np.log10(rms + eps)

        # Handle pure silence or digital zeros
        if np.max(rms) < 1e-4:
            zero_arr = np.zeros(num_frames, dtype=np.float64)
            return AudioPhysicsFeatures(
                timestamps=timestamps,
                rms_energy=rms,
                acoustic_jerk=zero_arr,
                baseline_energy=rms,
                dynamic_ratio_db=zero_arr,
                spectral_centroid=zero_arr,
                spectral_flux=zero_arr,
                spectral_flatness=zero_arr,
                pitch_periodicity=zero_arr,
                speech_confidence=zero_arr,
                high_freq_ratio=zero_arr,
                crest_factor=np.ones(num_frames, dtype=np.float64),
                shock_score=zero_arr,
                tkeo_peak=zero_arr,
                roughness_score=zero_arr,
            )

        # Crest factor: peak amplitude divided by RMS (impulsiveness metric)
        peak_amp = np.max(np.abs(frames), axis=1)
        crest_factor = peak_amp / (rms + eps)

        # Teager-Kaiser Energy Operator (TKEO)
        nyq = 0.5 * self.sample_rate
        b, a = butter(2, min(8000.0, nyq * 0.95) / nyq, btype='low')
        audio_tkeo_filt = filtfilt(b, a, audio)
        tkeo_signal = compute_tkeo(audio_tkeo_filt)
        tkeo_frames = np.lib.stride_tricks.as_strided(tkeo_signal, shape=shape, strides=strides)
        tkeo_peak = np.max(tkeo_frames, axis=1)

        # Linear acoustic jerk in s^-1: rate of rise in sound energy dE/dt
        linear_jerk = np.zeros(num_frames, dtype=np.float64)
        linear_jerk[1:] = np.maximum(0.0, (rms[1:] - rms[:-1]) / dt)

        # Causal attack transient memory over recent 70 ms (7 frames)
        attack_jerk = maximum_filter1d(linear_jerk, size=7, origin=3, mode="constant", cval=0.0)

        # Lagged ambient baseline tracking in decibels
        # Evaluates pre-shock tension window [i - win - lag : i - lag]
        win = self.baseline_frames
        lag = self.lag_frames
        padded_db = np.pad(decibel_level, (win + lag, 0), mode="edge")
        cumsum_db = np.cumsum(padded_db)
        baseline_db = (cumsum_db[win:] - cumsum_db[:-win]) / win
        baseline_db = baseline_db[-num_frames:]

        # Dynamic contrast ratio in decibels
        contrast_ratio_db = np.clip(decibel_level - baseline_db, 0.0, 60.0)

        # Short-Time Fourier Transform
        windowed_frames = frames * self.window
        fft_complex = np.fft.rfft(windowed_frames, axis=1)
        mag_spec = np.abs(fft_complex)
        power_spec = mag_spec ** 2

        # Fast autocorrelation via Wiener-Khinchin theorem for pitch periodicity
        autocorr = np.fft.irfft(power_spec, n=self.frame_length, axis=1)
        r0 = autocorr[:, 0:1] + eps
        norm_autocorr = autocorr / r0
        pitch_periodicity = np.max(
            norm_autocorr[:, self.pitch_lag_min : self.pitch_lag_max], axis=1
        )
        pitch_periodicity = np.clip(pitch_periodicity, 0.0, 1.0)

        # Wiener spectral flatness measure (tonal vs broadband noise)
        log_power = np.log(power_spec + 1e-12)
        geom_mean = np.exp(np.mean(log_power, axis=1))
        arith_mean = np.mean(power_spec + 1e-12, axis=1)
        spectral_flatness = geom_mean / (arith_mean + eps)

        # Apply psychoacoustic frequency weighting
        weighted_spec = mag_spec * self.a_weights

        # Spectral centroid: spectral center of mass
        spec_sum = np.sum(weighted_spec, axis=1) + eps
        centroid = np.sum(weighted_spec * self.freq_bins, axis=1) / spec_sum

        # Spectral flux: positive half-wave rectified spectral novelty
        spec_diff = np.zeros_like(weighted_spec)
        spec_diff[1:] = np.maximum(0.0, weighted_spec[1:] - weighted_spec[:-1])
        spectral_flux = np.sum(spec_diff, axis=1) / spec_sum

        # High-frequency scream and impact concentration
        high_mask = self.freq_bins >= self.high_freq_cutoff_hz
        high_energy = np.sum(weighted_spec[:, high_mask] ** 2, axis=1)
        total_spec_energy = np.sum(weighted_spec ** 2, axis=1) + eps
        high_freq_ratio = high_energy / total_spec_energy

        # Harmonic Voiced Speech Invariant Filter:
        # Human speech vocal tract phonation: high pitch periodicity, formant centroid < 1950 Hz,
        # very low spectral flatness (< 0.015), and low energy above 2.2 kHz.
        s_p = np.clip((pitch_periodicity - 0.50) / 0.25, 0.0, 1.0)
        s_c = np.clip(1.0 - (centroid - 1400.0) / 700.0, 0.0, 1.0)
        s_f = np.clip(1.0 - (spectral_flatness - 0.005) / 0.015, 0.0, 1.0)
        s_hf = np.clip(1.0 - (high_freq_ratio - 0.03) / 0.06, 0.0, 1.0)
        raw_speech = (s_p ** 0.35) * (s_f ** 0.35) * (s_c ** 0.20) * (s_hf ** 0.10)

        # Voice Activity Detector (VAD) leaky integrator with 350 ms hold time
        decay = np.exp(-dt / self.config.speech_hold_time_sec)
        speech_confidence = np.zeros_like(raw_speech)
        cur_sp = 0.0
        for i in range(len(raw_speech)):
            cur_sp = max(raw_speech[i], cur_sp * decay)
            speech_confidence[i] = cur_sp

        # Non-speech shock bypass conditions:
        # Colossal explosive pressure blast (RMS >= 0.65, attack jerk >= 20.0)
        is_colossal = (rms >= 0.65) & (attack_jerk >= 20.0)
        # High-frequency terror scream / stinger (Centroid >= 2350 Hz, Flatness >= 0.030, RMS >= 0.24, attack jerk >= 6.0)
        is_scream = (centroid >= 2350.0) & (spectral_flatness >= 0.030) & (rms >= 0.24) & (attack_jerk >= 6.0)
        speech_penalty = np.where(is_colossal | is_scream, 0.0, speech_confidence * self.speech_suppression_weight)

        # Mode 1: Ambush Shock (Impulsive contrast explosion from quiet tension)
        n_jerk = np.clip(attack_jerk / 16.0, 0.0, 1.0)
        n_contrast = np.clip(contrast_ratio_db / 25.0, 0.0, 1.0)
        n_rms = np.clip((rms - 0.20) / 0.40, 0.0, 1.0)
        n_crest = np.clip((crest_factor - 1.5) / 5.0, 0.0, 1.0)
        s_ambush = 0.35 * n_jerk + 0.30 * n_contrast + 0.20 * n_rms + 0.15 * n_crest
        s_ambush *= (contrast_ratio_db >= self.config.min_contrast_db).astype(np.float64)
        s_ambush *= (rms >= self.config.min_peak_rms).astype(np.float64)
        s_ambush *= (attack_jerk >= 8.0).astype(np.float64)

        # Mode 2: Screech / Stinger / Compound Shock (High-frequency panic during ongoing scene)
        n_cent = np.clip((centroid - 2000.0) / 1200.0, 0.0, 1.0)
        n_hf = np.clip(high_freq_ratio / 0.30, 0.0, 1.0)
        n_flat = np.clip((spectral_flatness - 0.02) / 0.08, 0.0, 1.0)
        n_jerk_st = np.clip(attack_jerk / 12.0, 0.0, 1.0)
        s_screech = 0.30 * n_cent + 0.25 * n_hf + 0.25 * n_flat + 0.20 * n_jerk_st
        s_screech *= (centroid >= 2300.0).astype(np.float64)
        s_screech *= (spectral_flatness >= 0.028).astype(np.float64)
        s_screech *= (rms >= self.config.min_peak_rms).astype(np.float64)
        s_screech *= (attack_jerk >= 5.0).astype(np.float64)
        s_screech *= (contrast_ratio_db >= 5.5).astype(np.float64)

        # Mode 3: Colossal Sonic Blast (Explosive stinger climax)
        s_blast = np.where(is_colossal, 0.90, 0.0)

        raw_score = np.maximum(np.maximum(s_ambush, s_screech), s_blast)
        shock_score = raw_score * (1.0 - speech_penalty)
        shock_score *= (rms >= self.config.min_peak_rms).astype(np.float64)

        from jumpscare_detector.audio_transient import compute_psychoacoustic_roughness
        # Calculate psychoacoustic roughness
        roughness_score = compute_psychoacoustic_roughness(audio, self.sample_rate, self.hop_length)

        return AudioPhysicsFeatures(
            timestamps=timestamps,
            rms_energy=rms,
            acoustic_jerk=attack_jerk,
            baseline_energy=baseline_db,
            dynamic_ratio_db=contrast_ratio_db,
            spectral_centroid=centroid,
            spectral_flux=spectral_flux,
            spectral_flatness=spectral_flatness,
            pitch_periodicity=pitch_periodicity,
            speech_confidence=speech_confidence,
            high_freq_ratio=high_freq_ratio,
            crest_factor=crest_factor,
            shock_score=shock_score,
            tkeo_peak=tkeo_peak,
            roughness_score=roughness_score,
        )

    def _determine_severity(self, score: float, contrast_db: float) -> str:
        """Map score and contrast to danger level."""
        if score >= 0.75 or contrast_db >= 28.0:
            return "Extreme"
        if score >= 0.65 or contrast_db >= 22.0:
            return "High"
        if score >= 0.52 or contrast_db >= 16.0:
            return "Moderate"
        return "Low"

    def detect_events(
        self,
        features: AudioPhysicsFeatures,
        cooldown_sec: Optional[float] = None,
    ) -> List[AudioShockEvent]:
        """Cluster threshold-exceeding frames into discrete events with sustain duration."""
        if cooldown_sec is None:
            cooldown_sec = self.config.cooldown_sec

        events: List[AudioShockEvent] = []
        scores = features.shock_score
        num_frames = len(scores)
        if num_frames == 0:
            return events

        i = 0
        min_frames_cooldown = max(1, int(cooldown_sec / (self.hop_length / self.sample_rate)))
        dt = self.hop_length / float(self.sample_rate)

        while i < num_frames:
            if scores[i] >= self.shock_threshold and features.rms_energy[i] >= self.config.min_peak_rms:
                start_idx = i
                peak_idx = i
                peak_score = scores[i]
                peak_rms = features.rms_energy[i]

                # Group subsequent frames that maintain shock energy
                decay_threshold_rms = peak_rms * 0.35
                while i < num_frames and (
                    scores[i] >= (self.shock_threshold * 0.50)
                    or features.rms_energy[i] >= decay_threshold_rms
                ):
                    if scores[i] > peak_score:
                        peak_score = scores[i]
                        peak_idx = i
                    if features.rms_energy[i] > peak_rms:
                        peak_rms = features.rms_energy[i]
                        decay_threshold_rms = peak_rms * 0.35
                    i += 1
                    if (i - start_idx) * dt >= 3.0:
                        break

                end_idx = max(start_idx, i - 1)
                t_start = float(features.timestamps[start_idx])
                duration = float(max(0.35, features.timestamps[end_idx] - t_start))
                peak_contrast = float(features.dynamic_ratio_db[peak_idx])

                events.append(
                    AudioShockEvent(
                        timestamp=t_start,
                        duration=duration,
                        peak_score=float(peak_score),
                        max_rms=float(features.rms_energy[peak_idx]),
                        jerk_value=float(features.acoustic_jerk[peak_idx]),
                        dynamic_ratio_db=peak_contrast,
                        spectral_centroid_hz=float(features.spectral_centroid[peak_idx]),
                        spectral_flatness=float(features.spectral_flatness[peak_idx]),
                        pitch_periodicity=float(features.pitch_periodicity[peak_idx]),
                        speech_confidence=float(features.speech_confidence[peak_idx]),
                        crest_factor=float(features.crest_factor[peak_idx]),
                        tkeo_peak=float(features.tkeo_peak[peak_idx]),
                        roughness_score=float(features.roughness_score[peak_idx]),
                        severity=self._determine_severity(peak_score, peak_contrast),
                    )
                )
                i += min_frames_cooldown
            else:
                i += 1

        return events


class StreamingAudioPhysicsDetector:
    """Stateful real-time chunked stream processor with speech rejection filter."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 25.0,
        hop_duration_ms: float = 10.0,
        baseline_window_sec: float = 2.5,
        min_contrast_db: float = 14.0,
        shock_threshold: float = 0.52,
        config: Optional[PhysicsConfig] = None,
    ) -> None:
        if config is not None:
            self.detector = AudioPhysicsDetector(config=config)
        else:
            self.detector = AudioPhysicsDetector(
                sample_rate=sample_rate,
                frame_duration_ms=frame_duration_ms,
                hop_duration_ms=hop_duration_ms,
                baseline_window_sec=baseline_window_sec,
                min_contrast_db=min_contrast_db,
                shock_threshold=shock_threshold,
            )

        self.buffer = np.zeros(0, dtype=np.float64)
        self.prev_frame_spectrum: Optional[np.ndarray] = None
        self.prev_db: float = -60.0
        self.baseline_ema_db: float = -45.0
        self.alpha_baseline = 1.0 / (self.detector.config.baseline_window_sec * (1000.0 / self.detector.config.hop_duration_ms))
        self.time_offset = 0.0

    def process_chunk(self, chunk: np.ndarray) -> List[Tuple[float, float, bool]]:
        """Process incoming audio chunk and return list of (timestamp, score, is_alert)."""
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)

        chunk = chunk.astype(np.float64)
        dt = self.detector.hop_length / float(self.detector.sample_rate)
        results: List[Tuple[float, float, bool]] = []

        combined = np.concatenate([self.buffer, chunk])
        frame_len = self.detector.frame_length
        hop_len = self.detector.hop_length

        idx = 0
        while idx + frame_len <= len(combined):
            frame = combined[idx : idx + frame_len]
            rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
            db = 20.0 * np.log10(rms + 1e-6)
            jerk_db = max(0.0, (db - self.prev_db) / dt)

            # Update lagged ambient baseline in decibels
            self.baseline_ema_db = (1.0 - self.alpha_baseline) * self.baseline_ema_db + self.alpha_baseline * db
            contrast_db = max(0.0, min(60.0, db - self.baseline_ema_db))

            windowed = frame * self.detector.window
            fft_c = np.fft.rfft(windowed)
            mag_spec = np.abs(fft_c)
            power_spec = mag_spec ** 2

            # Periodicity via autocorrelation
            autocorr = np.fft.irfft(power_spec, n=frame_len)
            r0 = autocorr[0] + 1e-9
            pitch_periodicity = float(
                np.max(autocorr[self.detector.pitch_lag_min : self.detector.pitch_lag_max]) / r0
            )
            pitch_periodicity = max(0.0, min(1.0, pitch_periodicity))

            # Spectral Flatness
            log_p = np.log(power_spec + 1e-12)
            flatness = float(np.exp(np.mean(log_p)) / (np.mean(power_spec) + 1e-12))

            # Spectral Centroid
            weighted_spec = mag_spec * self.detector.a_weights
            centroid = float(np.sum(weighted_spec * self.detector.freq_bins) / (np.sum(weighted_spec) + 1e-9))

            # Speech confidence
            p_t = max(0.0, min(1.0, (pitch_periodicity - 0.35) / 0.35))
            c_t = max(0.0, min(1.0, 1.0 - (centroid - 800.0) / 1200.0))
            f_t = max(0.0, min(1.0, 1.0 - flatness / 0.04))
            speech_conf = p_t * c_t * f_t

            is_conversational = (db < -14.0) and (jerk_db < 600.0) and (centroid < 2000.0)
            speech_penalty = (
                speech_conf * self.detector.speech_suppression_weight
                if is_conversational
                else speech_conf * 0.10
            )

            spec_flux = 0.0
            if self.prev_frame_spectrum is not None:
                diff = np.maximum(0.0, weighted_spec - self.prev_frame_spectrum)
                spec_flux = float(np.sum(diff) / (np.sum(weighted_spec) + 1e-6))
            self.prev_frame_spectrum = weighted_spec

            peak_amp = float(np.max(np.abs(frame)))
            crest = peak_amp / (rms + 1e-6)

            norm_jerk = min(1.0, jerk_db / 1200.0)
            norm_contrast = min(1.0, contrast_db / 28.0)
            norm_flux = min(1.0, spec_flux / 4.0)
            norm_crest = min(1.0, max(0.0, (crest - 1.5) / 5.0))
            norm_centroid = min(1.0, max(0.0, (centroid - 1200.0) / 2400.0))

            raw_score = (
                0.30 * norm_jerk
                + 0.30 * norm_contrast
                + 0.15 * norm_flux
                + 0.15 * norm_crest
                + 0.10 * norm_centroid
            )

            score = raw_score * (1.0 - speech_penalty)
            if db < self.detector.config.min_audible_dbfs:
                score = 0.0

            is_alert = (
                (score >= self.detector.shock_threshold)
                and (contrast_db >= self.detector.min_contrast_db)
                and (jerk_db >= self.detector.min_jerk_db_per_sec)
            )
            results.append((self.time_offset, float(score), is_alert))

            self.prev_db = db
            self.time_offset += dt
            idx += hop_len

        leftover_idx = idx
        self.buffer = combined[leftover_idx:].copy()
        return results
