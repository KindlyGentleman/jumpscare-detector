"""Unit tests for Phase 2 computer vision transient detection."""

import numpy as np
import pytest
from jumpscare_detector.video import (
    VideoConfig,
    analyze_video_frames,
    compute_frame_difference_features,
    compute_histogram_distance,
    compute_luminance_step,
    compute_radial_expansion,
    compute_residual_flow,
    process_frame_pair,
)


def test_static_scene_produces_zero_transients():
    """Identical consecutive frames produce zero motion and zero difference."""
    h, w = 120, 160
    frame = np.full((h, w), 100, dtype=np.uint8)

    diff_mean, diff_p95, diff_area = compute_frame_difference_features(frame, frame)
    lum_step = compute_luminance_step(frame, frame)
    hist_dist = compute_histogram_distance(frame, frame)

    assert diff_mean == 0.0
    assert diff_p95 == 0.0
    assert diff_area == 0.0
    assert lum_step == 0.0
    assert hist_dist == 0.0


def test_sudden_flash_luminance_and_histogram():
    """Sudden transition from pure black to pure white triggers maximum contrast."""
    h, w = 120, 160
    black_frame = np.zeros((h, w), dtype=np.uint8)
    white_frame = np.full((h, w), 255, dtype=np.uint8)

    diff_mean, diff_p95, diff_area = compute_frame_difference_features(black_frame, white_frame)
    lum_step = compute_luminance_step(black_frame, white_frame)
    hist_dist = compute_histogram_distance(black_frame, white_frame)

    assert diff_mean == 255.0
    assert diff_p95 == 255.0
    assert diff_area == 1.0
    assert lum_step == 255.0
    assert hist_dist == pytest.approx(1.0, abs=1e-3)


def test_expanding_object_divergence_and_radial_expansion():
    """Expanding object produces positive optical flow divergence and radial outward velocity."""
    h, w = 160, 160
    prev_frame = np.zeros((h, w), dtype=np.uint8)
    curr_frame = np.zeros((h, w), dtype=np.uint8)

    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    dist_from_center = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    # Frame 1: Circle with radius 15
    prev_frame[dist_from_center <= 15] = 200
    # Frame 2: Circle expands to radius 35
    curr_frame[dist_from_center <= 35] = 200

    config = VideoConfig(scale_factor=1.0)
    metric = process_frame_pair(prev_frame, curr_frame, 1, 0.033, config)

    assert metric.mean_frame_diff > 5.0
    assert metric.flow_mean > 0.5
    # Outward rushing must produce positive radial expansion
    assert metric.radial_expansion > 0.0
    # Flow divergence across the frame must be positive
    assert metric.flow_divergence > 0.0


def test_camera_pan_suppression_via_residual_flow():
    """Uniform camera pan shifts the whole image, dropping residual flow near zero."""
    h, w = 160, 200
    np.random.seed(42)
    # Textured surface (high-frequency noise) so Farneback can track displacement cleanly
    base = (np.random.rand(h, w) * 255).astype(np.uint8)

    # Pure horizontal camera pan of 5 pixels
    shifted = np.roll(base, shift=5, axis=1)

    config = VideoConfig(scale_factor=1.0)
    metric = process_frame_pair(base, shifted, 1, 0.033, config)

    # Raw flow should detect the shift
    assert metric.flow_mean > 2.0
    # Residual flow after median subtraction should be substantially lower than raw flow
    assert metric.flow_residual_mean < metric.flow_mean * 0.4


def test_analyze_video_frames_pipeline():
    """Sequence processing produces aligned timestamps and catches transient spike."""
    h, w = 100, 100
    fps = 30.0
    frames = []

    # 10 frames of static dark grey
    for _ in range(10):
        frames.append(np.full((h, w), 50, dtype=np.uint8))

    # Frame 5 has sudden flash
    frames[5] = np.full((h, w), 240, dtype=np.uint8)

    df = analyze_video_frames(frames, fps=fps, config=VideoConfig(scale_factor=1.0))

    assert len(df) == 10
    assert "timestamp" in df.columns
    assert df.loc[0, "mean_frame_diff"] == 0.0
    # Frame 5 must show sharp difference spike
    assert df.loc[5, "mean_frame_diff"] > 180.0
    assert df.loc[5, "luminance_step"] > 180.0
    # Frame 6 drops back to dark grey, also showing difference spike
    assert df.loc[6, "mean_frame_diff"] > 180.0
    # Frame 7 returns to baseline
    assert df.loc[7, "mean_frame_diff"] == 0.0
