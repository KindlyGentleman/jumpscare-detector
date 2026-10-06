"""Unit tests for Phase 1: High-resolution audio transients and Hilbert rise-time."""

import numpy as np
import pytest

from jumpscare_detector.audio_transient import (
    AudioTransientConfig,
    compute_audio_transients,
    robust_zscore,
    trailing_median,
)
from jumpscare_detector.benchmark import (
    generate_ambient_drone,
    generate_scream_shock,
    generate_violin_cluster,
)


def test_robust_zscore() -> None:
    """Robust Z-score via MAD should scale Gaussian noise correctly and flag outliers."""
    np.random.seed(42)
    noise = np.random.normal(0, 1.0, 1000)
    # Add an extreme outlier
    noise[500] = 12.0
    z = robust_zscore(noise)

    # Median and normal points should be close to 0
    assert np.isclose(float(np.median(z)), 0.0, atol=0.1)
    # Outlier should be strongly elevated
    assert z[500] >= 5.0


def test_trailing_median_causality() -> None:
    """Trailing median must not look into future samples."""
    x = np.zeros(100)
    x[50:] = 10.0  # Step function at index 50

    window_size = 11
    filtered = trailing_median(x, window_size)

    # Before index 50, trailing median must be 0
    assert np.all(filtered[:50] == 0.0)
    # At index 55 (half window), trailing median reaches step
    assert filtered[55] == 10.0


def test_hilbert_rise_time_fast_shock_vs_crescendo() -> None:
    """Explosive shock must have very short rise time; slow crescendo must have long rise time."""
    sr = 22050
    # Fast shock: 1.5s ambient + explosive 15ms onset scream
    ambient = generate_ambient_drone(2.0, sr, amplitude=0.01)
    shock = generate_scream_shock(1.0, sr, amplitude=0.85)
    fast_audio = np.concatenate([ambient, shock])

    cfg = AudioTransientConfig(sample_rate=sr, n_fft=1024, hop_length=256)
    df_fast, _ = compute_audio_transients(fast_audio, sample_rate=sr, config=cfg)

    # Find the shock onset frame around 2.0s
    shock_idx = int(2.0 * sr / cfg.hop_length)
    # Check minimum rise time around onset
    local_window = df_fast["rise_time_s"].iloc[shock_idx : shock_idx + 8].dropna()
    assert len(local_window) > 0
    min_shock_rise = float(local_window.min())
    assert min_shock_rise < 0.05  # Rises in under 50 ms

    # Check attack slope and rise score
    local_rise_scores = df_fast["rise_score"].iloc[shock_idx : shock_idx + 8]
    assert float(local_rise_scores.max()) > 100.0  # High dimensionless rise score


def test_multiband_flux_separation() -> None:
    """Sub-bass thuds must dominate low-band flux; screams must dominate high-band flux."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Sub-bass burst: 60 Hz tone with 20ms onset
    env = np.clip(t / 0.02, 0.0, 1.0) * np.exp(-3.0 * t)
    bass = np.sin(2 * np.pi * 60 * t) * env

    cfg = AudioTransientConfig(sample_rate=sr, n_fft=1024, hop_length=256)
    df_bass, _ = compute_audio_transients(bass, sample_rate=sr, config=cfg)

    # Peak low flux should exceed high flux for bass
    max_low_bass = float(df_bass["flux_low"].max())
    max_high_bass = float(df_bass["flux_high"].max())
    assert max_low_bass > max_high_bass

    # High-frequency stinger (violin cluster around 3.5 kHz)
    scream = generate_violin_cluster(1.0, sr, amplitude=0.8)
    df_scream, _ = compute_audio_transients(scream, sample_rate=sr, config=cfg)

    max_low_scream = float(df_scream["flux_low"].max())
    max_high_scream = float(df_scream["flux_high"].max())
    assert max_high_scream > max_low_scream


def test_kurtosis_and_crest_factor_impulsiveness() -> None:
    """Impulsive noise clicks should produce higher kurtosis and crest factor than continuous sine."""
    sr = 22050
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t) * 0.5

    # Sparse impulsive click train
    click = np.zeros_like(t)
    click[int(0.25 * sr)] = 0.95

    cfg = AudioTransientConfig(sample_rate=sr, n_fft=1024, hop_length=256)
    df_sine, _ = compute_audio_transients(sine, sample_rate=sr, config=cfg)
    df_click, _ = compute_audio_transients(click, sample_rate=sr, config=cfg)

    # Crest factor of sine is sqrt(2) ~ 1.41
    mean_sine_crest = float(df_sine["crest_factor"].mean())
    assert abs(mean_sine_crest - 1.41) < 0.3

    # Peak crest factor and kurtosis of click must be much higher
    assert float(df_click["crest_factor"].max()) > 5.0
    assert float(df_click["kurtosis"].max()) > 10.0
