"""High-resolution audio transient, envelope rise-time, and spectral feature extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import kurtosis


@dataclass
class AudioTransientConfig:
    """Configuration for high-resolution audio transient analysis."""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    baseline_window_sec: float = 2.0
    min_dynamic_range_linear: float = 1e-4
    envelope_smooth_sigma: float = 2.0
    clipping_threshold: float = 0.99


def robust_zscore(
    x: np.ndarray,
    min_scale: float = 1e-4,
    clip_value: float = 8.0,
) -> np.ndarray:
    """Compute robust Z-score using Median Absolute Deviation (MAD).

    Formula: Z = (x - median(x)) / max(1.4826 * MAD(x), min_scale)
    The 1.4826 factor scales MAD to equal standard deviation for Gaussian distributions.
    The min_scale parameter prevents variance collapse in silence or synthetic signals.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    med = float(np.nanmedian(x_arr))
    mad = float(np.nanmedian(np.abs(x_arr - med)))
    scale = max(1.4826 * mad, min_scale)
    z = (x_arr - med) / scale
    return np.clip(z, 0.0, clip_value)


def trailing_median(x: np.ndarray, size: int) -> np.ndarray:
    """Compute strictly causal trailing rolling median over past window."""
    x_arr = np.asarray(x, dtype=np.float64)
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    padded = np.pad(x_arr, (size - 1, 0), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=-1)


def safe_smooth(x: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply 1D Gaussian smoothing with nearest-edge padding."""
    x_arr = np.asarray(x, dtype=np.float64)
    if len(x_arr) < 5:
        return x_arr
    return gaussian_filter1d(x_arr, sigma=sigma, mode="nearest")


def compute_tkeo(x: np.ndarray) -> np.ndarray:
    """Compute Teager-Kaiser Energy Operator: Psi(x[n]) = x^2[n] - x[n-1]*x[n+1]."""
    x_arr = np.asarray(x, dtype=np.float64)
    if len(x_arr) < 3:
        return np.zeros_like(x_arr)
    
    tkeo = np.zeros_like(x_arr)
    tkeo[1:-1] = x_arr[1:-1]**2 - x_arr[:-2] * x_arr[2:]
    tkeo[0] = tkeo[1]
    tkeo[-1] = tkeo[-2]
    return np.maximum(0.0, tkeo)


def compute_psychoacoustic_roughness(y: np.ndarray, sr: int, target_hop: int) -> np.ndarray:
    """Compute Psychoacoustic Roughness (Sensory Dissonance).
    
    Extracts amplitude modulation depth within 30-150 Hz across Mel bands,
    which strongly correlates with the biological perception of harshness
    found in screams or mechanical tearing.
    """
    mod_hop = max(32, sr // 344) # roughly ~344 Hz frame rate
    frame_rate = sr / mod_hop
    
    # Use short window to preserve fast amplitude modulations (150Hz = ~6.6ms period)
    n_fft_roughness = 256
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft_roughness, hop_length=mod_hop, n_mels=32)
    
    nyq = frame_rate / 2.0
    low = 30.0 / nyq
    high = 150.0 / nyq
    high = min(high, 0.99)
    if low >= high:
        return np.zeros(int(np.ceil(len(y) / target_hop)))
        
    b, a = butter(2, [low, high], btype='band')
    roughness = np.zeros(S.shape[1])
    global_max = float(np.max(S)) + 1e-9
    
    for i in range(S.shape[0]):
        band_env = S[i, :]
        if np.mean(band_env) < 0.01 * global_max:
            continue
            
        mod = filtfilt(b, a, band_env)
        smooth_env = np.convolve(band_env, np.ones(10)/10, mode='same')
        local_depth = np.abs(mod) / (smooth_env + 0.01 * global_max)
        
        # Weight by relative energy in this band
        roughness += local_depth * (band_env / global_max)
        
    smooth_frames = max(1, int(0.05 * frame_rate))
    window = np.ones(smooth_frames) / smooth_frames
    roughness = np.convolve(roughness, window, mode='same')
    
    # Resample to match the main feature hop_length
    target_frames = 1 + len(y) // target_hop
    orig_times = librosa.frames_to_time(np.arange(len(roughness)), sr=sr, hop_length=mod_hop)
    target_times = librosa.frames_to_time(np.arange(target_frames), sr=sr, hop_length=target_hop)
    
    roughness_resampled = np.interp(target_times, orig_times, roughness)
    return roughness_resampled


def compute_audio_transients(
    audio_input: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    config: Optional[AudioTransientConfig] = None,
) -> Tuple[pd.DataFrame, int]:
    """Extract fine-grained audio transients, Hilbert envelope, and multi-band spectral flux.

    Features Extracted:
    - RMS & decibels (relative digital level)
    - Trailing baseline median & relative RMS spike (C_R)
    - Analytic Hilbert amplitude envelope a[n] = |x[n] + j H{x[n]}|
    - 10-90% envelope rise time T_r = t_90 - t_10
    - Peak attack slope S_attack = max(da/dt)
    - Dimensionless rise score S_rise = C_R / (T_r + eps)
    - Multi-band positive spectral flux (Low: 20-250 Hz, Mid: 250-2000 Hz, High: 2000-10000 Hz)
    - Spectral centroid, bandwidth, contrast, and Wiener spectral flatness
    - Zero-crossing rate (ZCR), crest factor, excess kurtosis, and clipping ratio
    """
    if config is None:
        config = AudioTransientConfig()

    sr = config.sample_rate if sample_rate is None else sample_rate

    # Load audio array
    if isinstance(audio_input, (str, Path)):
        y, sr = librosa.load(str(audio_input), sr=sr, mono=True)
    else:
        y = np.asarray(audio_input, dtype=np.float64)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

    n_fft = config.n_fft
    hop = config.hop_length
    eps = 1e-10

    # Short-Time Fourier Transform
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann", center=True)
    magnitude = np.abs(stft_complex)
    power = magnitude ** 2
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop)
    frame_count = len(times)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 1. RMS Energy & Decibels
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms + eps, ref=1.0)

    # 2. Local RMS Spike over Trailing Baseline
    baseline_frames = max(5, int(config.baseline_window_sec * sr / hop))
    rms_baseline = trailing_median(rms, baseline_frames)
    rms_spike_db = 20.0 * np.log10((rms + eps) / (rms_baseline + eps))

    # 3. Log-Power Spectral Flux
    log_power = np.log(power + eps)
    spectral_flux = np.mean(np.maximum(0.0, log_power[:, 1:] - log_power[:, :-1]), axis=0)
    spectral_flux = np.concatenate([[0.0], spectral_flux])

    # 4. Multi-Band Spectral Flux (Low: 20-250 Hz, Mid: 250-2000 Hz, High: 2000-10000 Hz)
    low_mask = (freqs >= 20.0) & (freqs < 250.0)
    mid_mask = (freqs >= 250.0) & (freqs < 2000.0)
    high_mask = (freqs >= 2000.0) & (freqs <= 10000.0)

    diff_mag = np.maximum(0.0, magnitude[:, 1:] - magnitude[:, :-1])
    diff_mag = np.concatenate([np.zeros((magnitude.shape[0], 1)), diff_mag], axis=1)

    total_mag = np.sum(magnitude, axis=0) + eps
    flux_low = np.sum(diff_mag[low_mask, :], axis=0) / total_mag
    flux_mid = np.sum(diff_mag[mid_mask, :], axis=0) / total_mag
    flux_high = np.sum(diff_mag[high_mask, :], axis=0) / total_mag

    # 5. Spectral Centroid, Bandwidth, Contrast, and Flatness
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr, n_bands=6)
    spectral_contrast_mean = np.mean(contrast, axis=0)
    flatness = librosa.feature.spectral_flatness(S=power)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop, center=True)[0]
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean)

    # 6. Analytic Hilbert Envelope & Rise-Time Analysis
    analytic = hilbert(y)
    envelope = np.abs(analytic)
    envelope = safe_smooth(envelope, sigma=max(1.0, float(sr / hop / 100.0)))

    crest_factor = np.zeros(frame_count, dtype=np.float64)
    kurtosis_values = np.zeros(frame_count, dtype=np.float64)
    clipping_ratio = np.zeros(frame_count, dtype=np.float64)
    envelope_peak = np.zeros(frame_count, dtype=np.float64)
    rise_time = np.full(frame_count, np.nan, dtype=np.float64)
    attack_slope = np.zeros(frame_count, dtype=np.float64)
    impulsiveness = np.zeros(frame_count, dtype=np.float64)
    
    # 7. Teager-Kaiser Energy Operator (TKEO)
    nyq = 0.5 * sr
    b, a = butter(2, min(8000.0, nyq * 0.95) / nyq, btype='low')
    y_tkeo_filt = filtfilt(b, a, y)
    tkeo_signal = compute_tkeo(y_tkeo_filt)
    tkeo_peak = np.zeros(frame_count, dtype=np.float64)

    dt_sample = 1.0 / float(sr)

    for k in range(frame_count):
        start = k * hop
        stop = min(start + n_fft, len(y))
        segment = y[start:stop]
        if len(segment) < 8:
            continue

        seg_abs = np.abs(segment)
        seg_rms = float(np.sqrt(np.mean(segment ** 2) + eps))
        seg_peak = float(np.max(seg_abs))

        crest_factor[k] = seg_peak / (seg_rms + eps)
        kurtosis_values[k] = float(kurtosis(segment, fisher=True, bias=False))
        clipping_ratio[k] = float(np.mean(seg_abs >= config.clipping_threshold))
        impulsiveness[k] = seg_rms / (float(np.mean(seg_abs)) + eps)
        tkeo_peak[k] = float(np.max(tkeo_signal[start:stop]))

        # Frame envelope slice
        e_start = min(start, len(envelope) - 1)
        e_stop = min(start + n_fft, len(envelope))
        e = envelope[e_start:e_stop]
        if len(e) < 8:
            continue

        e_smooth = safe_smooth(e, sigma=config.envelope_smooth_sigma)
        e_min = float(np.percentile(e_smooth, 10))
        e_max = float(np.percentile(e_smooth, 99))
        envelope_peak[k] = e_max

        dyn_range = e_max - e_min
        if dyn_range <= config.min_dynamic_range_linear:
            continue

        a10 = e_min + 0.10 * dyn_range
        a90 = e_min + 0.90 * dyn_range
        idx10 = np.flatnonzero(e_smooth >= a10)
        idx90 = np.flatnonzero(e_smooth >= a90)

        if len(idx10) > 0 and len(idx90) > 0:
            i10 = idx10[0]
            i90_candidates = idx90[idx90 >= i10]
            if len(i90_candidates) > 0:
                i90 = i90_candidates[0]
                rise_time[k] = float(i90 - i10) * dt_sample

        deriv = np.diff(e_smooth) / dt_sample
        if len(deriv) > 0:
            attack_slope[k] = float(np.max(deriv))

    # 7. Peak-to-Baseline Contrast in Envelope
    peak_baseline = trailing_median(envelope_peak + eps, baseline_frames)
    peak_to_baseline_db = 20.0 * np.log10((envelope_peak + eps) / (peak_baseline + eps))

    # 8. Dimensionless Rise Score S_rise = max(0, peak_to_baseline_db) / (rise_time + 1ms)
    rise_score = np.zeros_like(rise_time)
    valid_rise = np.isfinite(rise_time) & (rise_time > 0.0)
    rise_score[valid_rise] = np.maximum(0.0, peak_to_baseline_db[valid_rise]) / (rise_time[valid_rise] + 1e-3)

    def fit_length(values: np.ndarray) -> np.ndarray:
        val_arr = np.asarray(values, dtype=np.float64)
        if len(val_arr) == frame_count:
            return val_arr
        source_t = np.linspace(0.0, times[-1], len(val_arr))
        return np.interp(times, source_t, val_arr)

    # 9. Psychoacoustic Roughness (30-150Hz Modulation)
    roughness = compute_psychoacoustic_roughness(y, sr, hop)

    df_data: Dict[str, np.ndarray] = {
        "time": times,
        "timestamp": times,
        "rms": rms,
        "rms_db": rms_db,
        "rms_spike_db": rms_spike_db,
        "spectral_flux": fit_length(spectral_flux),
        "flux_low": fit_length(flux_low),
        "flux_mid": fit_length(flux_mid),
        "flux_high": fit_length(flux_high),
        "onset_strength": fit_length(onset_strength),
        "spectral_centroid_hz": fit_length(centroid),
        "spectral_bandwidth_hz": fit_length(bandwidth),
        "spectral_contrast": fit_length(spectral_contrast_mean),
        "spectral_flatness": fit_length(flatness),
        "zcr": fit_length(zcr),
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_values,
        "impulsiveness": impulsiveness,
        "clipping_ratio": clipping_ratio,
        "envelope_peak": envelope_peak,
        "rise_time_s": rise_time,
        "rise_time_ms": rise_time * 1000.0,
        "attack_slope": attack_slope,
        "peak_to_baseline_db": peak_to_baseline_db,
        "rise_score": rise_score,
        "tkeo_peak": tkeo_peak,
        "roughness_score": fit_length(roughness),
    }

    return pd.DataFrame(df_data), sr


extract_audio_transients = compute_audio_transients
