"""Unit tests for Phase 3 multimodal fusion and synchrony engine."""

import numpy as np
import pandas as pd
import pytest
from jumpscare_detector.multimodal import (
    MultimodalConfig,
    compute_audio_visual_synchrony,
    detect_multimodal_jumpscares,
    extract_candidate_events,
    fuse_multimodal_timeline,
)


def _generate_synthetic_audio_df(n_points: int = 100, dt: float = 0.05) -> pd.DataFrame:
    """Generate baseline audio features DataFrame."""
    timestamps = np.arange(n_points) * dt
    data = {
        "timestamp": timestamps,
        "rms_db": np.full(n_points, -40.0),
        "rise_time_ms": np.full(n_points, 500.0),
        "rise_score": np.full(n_points, 0.1),
        "attack_slope": np.full(n_points, 0.05),
        "flux_low": np.full(n_points, 0.01),
        "flux_mid": np.full(n_points, 0.01),
        "flux_high": np.full(n_points, 0.01),
        "spectral_centroid": np.full(n_points, 1000.0),
        "spectral_bandwidth": np.full(n_points, 500.0),
        "spectral_contrast": np.full(n_points, 1.0),
        "spectral_flatness": np.full(n_points, 0.1),
        "onset_strength": np.full(n_points, 0.05),
        "crest_factor": np.full(n_points, 2.0),
        "kurtosis": np.full(n_points, 0.0),
        "clipping_ratio": np.full(n_points, 0.0),
        "tkeo_peak": np.full(n_points, 0.0),
        "roughness_score": np.full(n_points, 0.0),
    }
    return pd.DataFrame(data)


def _generate_synthetic_video_df(n_points: int = 100, dt: float = 0.05) -> pd.DataFrame:
    """Generate baseline video features DataFrame."""
    timestamps = np.arange(n_points) * dt
    data = {
        "timestamp": timestamps,
        "mean_frame_diff": np.full(n_points, 1.0),
        "p95_frame_diff": np.full(n_points, 2.0),
        "diff_area_ratio": np.full(n_points, 0.01),
        "luminance_step": np.full(n_points, 0.5),
        "hist_distance": np.full(n_points, 0.02),
        "flow_mean": np.full(n_points, 0.2),
        "flow_p95": np.full(n_points, 0.5),
        "flow_area_ratio": np.full(n_points, 0.01),
        "flow_divergence": np.full(n_points, 0.0),
        "radial_expansion": np.full(n_points, 0.0),
        "flow_residual_mean": np.full(n_points, 0.1),
        "flow_residual_p95": np.full(n_points, 0.2),
        "phase_motion_score": np.full(n_points, 0.05),
    }
    return pd.DataFrame(data)


def test_audio_visual_synchrony_calculation():
    """Synchronous peaks produce high synchrony; asynchronous signals drop to zero."""
    timestamps = np.linspace(0.0, 5.0, 101)
    s_audio = np.full(101, 0.1)
    s_video = np.full(101, 0.1)

    # Coincident peaks at t = 2.5s (index 50)
    s_audio[50] = 5.0
    s_video[50] = 5.0

    sync = compute_audio_visual_synchrony(timestamps, s_audio, s_video, sigma=0.25, max_window=0.5)

    assert sync[50] > 0.9

    # Unsynchronized: audio peak at index 50, video peak at index 80 (separated by 1.5s)
    s_video_offset = np.full(101, 0.1)
    s_video_offset[80] = 5.0
    sync_offset = compute_audio_visual_synchrony(
        timestamps, s_audio, s_video_offset, sigma=0.25, max_window=0.5
    )

    assert sync_offset[50] == 0.0
    assert sync_offset[80] == 0.0


def test_multimodal_coincident_jumpscare_detected():
    """Coincident audio and visual shock triggers fused jump scare candidate."""
    df_audio = _generate_synthetic_audio_df(100, dt=0.05)
    df_video = _generate_synthetic_video_df(100, dt=0.05)

    shock_idx = 50  # t = 2.50s
    # Audio shock
    df_audio.loc[shock_idx, "rms_db"] = -5.0
    df_audio.loc[shock_idx, "rise_time_ms"] = 15.0
    df_audio.loc[shock_idx, "rise_score"] = 6.0
    df_audio.loc[shock_idx, "flux_high"] = 2.5
    df_audio.loc[shock_idx, "onset_strength"] = 3.0

    # Visual shock
    df_video.loc[shock_idx, "mean_frame_diff"] = 80.0
    df_video.loc[shock_idx, "p95_frame_diff"] = 150.0
    df_video.loc[shock_idx, "luminance_step"] = 70.0
    df_video.loc[shock_idx, "flow_mean"] = 12.0
    df_video.loc[shock_idx, "flow_divergence"] = 3.0
    df_video.loc[shock_idx, "radial_expansion"] = 5.0

    config = MultimodalConfig(detection_threshold=2.0)
    candidates, timeline = detect_multimodal_jumpscares(df_audio, df_video, config)

    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(2.50, abs=0.06)
    assert candidates[0].sync_score > 0.5
    assert candidates[0].score > 2.0


def test_multimodal_suppresses_unsynchronized_audio_dialogue():
    """High audio activity with calm video is attenuated relative to true multimodal shock."""
    df_audio = _generate_synthetic_audio_df(100, dt=0.05)
    df_video = _generate_synthetic_video_df(100, dt=0.05)

    shock_idx = 50
    # Moderate speech/dialogue transient
    df_audio.loc[shock_idx, "rms_db"] = -18.0
    df_audio.loc[shock_idx, "rise_time_ms"] = 120.0
    df_audio.loc[shock_idx, "rise_score"] = 1.0
    df_audio.loc[shock_idx, "flux_high"] = 0.3
    df_audio.loc[shock_idx, "onset_strength"] = 0.5

    # Video is calm
    config = MultimodalConfig(detection_threshold=2.2)
    candidates, timeline = detect_multimodal_jumpscares(df_audio, df_video, config)

    # Speech without visual shock must not trigger candidate event
    assert len(candidates) == 0


def test_minimum_separation_filters_aftershocks():
    """Secondary cluster spikes within separation window are suppressed."""
    df_audio = _generate_synthetic_audio_df(100, dt=0.05)
    df_video = _generate_synthetic_video_df(100, dt=0.05)

    # Primary shock at index 40 (t = 2.0s)
    df_audio.loc[40, "rms_db"] = -5.0
    df_audio.loc[40, "rise_score"] = 6.0
    df_video.loc[40, "mean_frame_diff"] = 80.0
    df_video.loc[40, "flow_mean"] = 10.0

    # Aftershock at index 44 (t = 2.2s, only 0.2s later)
    df_audio.loc[44, "rms_db"] = -8.0
    df_audio.loc[44, "rise_score"] = 4.0
    df_video.loc[44, "mean_frame_diff"] = 60.0
    df_video.loc[44, "flow_mean"] = 8.0

    config = MultimodalConfig(detection_threshold=1.5, min_separation=1.0)
    candidates, _ = detect_multimodal_jumpscares(df_audio, df_video, config)

    # Exactly 1 event should be picked, not 2
    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(2.0, abs=0.06)


def test_audio_only_fallback():
    """Pipeline works when video DataFrame is None."""
    df_audio = _generate_synthetic_audio_df(100, dt=0.05)
    df_audio.loc[50, "rms_db"] = -5.0
    df_audio.loc[50, "rise_score"] = 5.0
    df_audio.loc[50, "flux_high"] = 2.0
    df_audio.loc[50, "onset_strength"] = 2.0

    config = MultimodalConfig(detection_threshold=1.5)
    candidates, timeline = detect_multimodal_jumpscares(df_audio, None, config)

    assert len(candidates) == 1
    assert candidates[0].timestamp == pytest.approx(2.50, abs=0.06)
    assert candidates[0].video_score == 0.0
    assert "fused_score" in timeline.columns
