"""Multimodal fusion and audio-visual synchrony engine for jump scare detection."""

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from jumpscare_detector.audio_transient import robust_zscore


@dataclass(frozen=True)
class MultimodalConfig:
    """Parameters for multimodal fusion, gating, and event peak picking."""

    weight_audio: float = 0.40
    weight_video: float = 0.30
    weight_contrast: float = 0.10
    weight_sync: float = 0.20
    gate_audio_threshold: float = 1.2
    gate_video_threshold: float = 1.2
    detection_threshold: float = 2.2
    min_separation: float = 1.0
    sync_sigma: float = 0.25
    sync_max_window: float = 0.50


@dataclass(frozen=True)
class CandidateEvent:
    """Audit record for a detected jump scare candidate event."""

    timestamp: float
    score: float
    audio_score: float
    video_score: float
    sync_score: float
    contrast_score: float
    rms_db: float
    rise_time_ms: float
    spectral_flux: float
    mean_frame_diff: float
    flow_divergence: float
    radial_expansion: float


def compute_audio_modality_score(
    df_audio: pd.DataFrame,
    min_audible_db: float = -35.0,
) -> np.ndarray:
    """Calculate normalized audio transient score from acoustic shock features."""
    z_rms = robust_zscore(df_audio["rms_db"].values, min_scale=3.0)
    z_rise = robust_zscore(df_audio["rise_score"].values, min_scale=0.5)
    z_flux = robust_zscore(df_audio["flux_high"].values + df_audio["flux_mid"].values, min_scale=0.2)
    z_onset = robust_zscore(df_audio["onset_strength"].values, min_scale=0.2)
    z_attack = robust_zscore(df_audio["attack_slope"].values, min_scale=0.1)
    z_crest = robust_zscore(df_audio["crest_factor"].values, min_scale=0.5)
    z_kurt = robust_zscore(df_audio["kurtosis"].values, min_scale=1.0)
    
    # Physics Phase 1 & 3 Advanced Metrics
    z_tkeo = robust_zscore(df_audio["tkeo_peak"].values, min_scale=0.01)
    rough_vals = df_audio["roughness_score"].values if "roughness_score" in df_audio else np.zeros_like(z_rms)
    z_rough = robust_zscore(rough_vals, min_scale=0.05)

    raw_sa = (
        0.20 * z_rms
        + 0.15 * z_rise
        + 0.15 * z_flux
        + 0.10 * z_onset
        + 0.10 * z_attack
        + 0.05 * z_crest
        + 0.05 * z_kurt
        + 0.10 * z_tkeo
        + 0.10 * z_rough
    )
    # Attenuate sub-audible room tone and digital silence
    audibility = np.clip((df_audio["rms_db"].values - min_audible_db) / 10.0, 0.0, 1.0)
    return raw_sa * audibility


def compute_video_modality_score(df_video: pd.DataFrame) -> np.ndarray:
    """Calculate normalized visual transient score from motion and contrast features."""
    z_diff = robust_zscore(df_video["mean_frame_diff"].values, min_scale=2.0)
    z_diff95 = robust_zscore(df_video["p95_frame_diff"].values, min_scale=3.0)
    z_lum = robust_zscore(df_video["luminance_step"].values, min_scale=2.0)
    z_hist = robust_zscore(df_video["hist_distance"].values, min_scale=0.05)
    z_flow = robust_zscore(df_video["flow_mean"].values, min_scale=0.5)
    z_flow95 = robust_zscore(df_video["flow_p95"].values, min_scale=1.0)
    z_area = robust_zscore(df_video["flow_area_ratio"].values, min_scale=0.02)

    # Inverse Time-To-Contact (Tau) - higher means more imminent collision
    inv_tau = 1.0 / (df_video["time_to_contact_tau"].values + 1e-3)
    z_tau = robust_zscore(inv_tau, min_scale=0.01)

    # Phase 4 Phase-Based Motion
    phase_vals = df_video["phase_motion_score"].values if "phase_motion_score" in df_video else np.zeros_like(z_diff)
    z_phase = robust_zscore(phase_vals, min_scale=0.01)

    s_v = (
        0.20 * z_diff
        + 0.15 * z_diff95
        + 0.15 * z_lum
        + 0.10 * z_hist
        + 0.15 * z_flow
        + 0.05 * z_flow95
        + 0.05 * z_area
        + 0.05 * z_tau
        + 0.10 * z_phase
    )
    return s_v


def compute_audio_visual_synchrony(
    timestamps: np.ndarray,
    s_audio: np.ndarray,
    s_video: np.ndarray,
    sigma: float = 0.25,
    max_window: float = 0.50,
) -> np.ndarray:
    """Calculate temporal coincidence score between audio and visual shock peaks."""
    n_pts = len(timestamps)
    s_sync = np.zeros(n_pts, dtype=np.float64)

    # Find prominent peaks in each modality
    audio_peaks, _ = find_peaks(s_audio, height=1.0, distance=3)
    video_peaks, _ = find_peaks(s_video, height=1.0, distance=3)

    audio_peak_times = timestamps[audio_peaks] if len(audio_peaks) > 0 else np.array([])
    video_peak_times = timestamps[video_peaks] if len(video_peaks) > 0 else np.array([])

    for i in range(n_pts):
        t = timestamps[i]
        sa = s_audio[i]
        sv = s_video[i]

        if sa < 0.5 or sv < 0.5:
            # Unbalanced or negligible activity
            s_sync[i] = 0.0
            continue

        # Magnitude symmetry factor in [0.0, 1.0]
        balance = 2.0 * min(sa, sv) / (sa + sv + 1e-6)

        # Time difference to nearest opposing peak
        dt = 0.0
        if len(audio_peak_times) > 0 and len(video_peak_times) > 0:
            nearest_a = float(np.min(np.abs(audio_peak_times - t)))
            nearest_v = float(np.min(np.abs(video_peak_times - t)))
            dt = abs(nearest_a - nearest_v)
        else:
            dt = 0.0

        if dt > max_window:
            gaussian_coincidence = 0.0
        else:
            gaussian_coincidence = np.exp(-(dt**2) / (2.0 * sigma**2))

        s_sync[i] = balance * gaussian_coincidence

    return s_sync


def compute_temporal_contrast_score(
    s_combined: np.ndarray,
    timestamps: np.ndarray,
    pre_window_sec: float = 4.0,
    dead_zone_sec: float = 0.3,
    baseline_floor: float = 0.5,
) -> np.ndarray:
    """Measure transient spike ratio against pre-event background context."""
    n = len(timestamps)
    contrast = np.zeros(n, dtype=np.float64)

    for i in range(n):
        t = timestamps[i]
        # Baseline window: [t - pre_window_sec, t - dead_zone_sec]
        mask = (timestamps >= (t - pre_window_sec)) & (timestamps <= (t - dead_zone_sec))
        if np.any(mask):
            baseline = max(baseline_floor, float(np.mean(s_combined[mask])))
            ratio = s_combined[i] / baseline
            contrast[i] = min(5.0, max(0.0, ratio))
        else:
            contrast[i] = 1.0

    return contrast


def fuse_multimodal_timeline(
    df_audio: pd.DataFrame,
    df_video: Optional[pd.DataFrame] = None,
    config: MultimodalConfig = MultimodalConfig(),
) -> pd.DataFrame:
    """Align modalities on audio time base, compute synchrony, and produce fused audit timeline."""
    out = df_audio.copy()
    timestamps = out["timestamp"].values

    s_a = compute_audio_modality_score(df_audio)
    out["audio_score"] = s_a

    if df_video is not None and not df_video.empty:
        # Interpolate video features onto audio timeline
        v_times = df_video["timestamp"].values
        for col in [
            "mean_frame_diff",
            "p95_frame_diff",
            "luminance_step",
            "hist_distance",
            "flow_mean",
            "flow_p95",
            "flow_area_ratio",
            "flow_divergence",
            "radial_expansion",
            "time_to_contact_tau",
            "flow_residual_mean",
            "flow_residual_p95",
        ]:
            if col in df_video.columns:
                out[col] = np.interp(timestamps, v_times, df_video[col].values)
            else:
                out[col] = 0.0

        s_v = compute_video_modality_score(out)
        out["video_score"] = s_v

        s_sync = compute_audio_visual_synchrony(
            timestamps,
            s_a,
            s_v,
            sigma=config.sync_sigma,
            max_window=config.sync_max_window,
        )
        out["sync_score"] = s_sync

        s_combined = s_a + s_v
        s_contrast = compute_temporal_contrast_score(s_combined, timestamps)
        out["contrast_score"] = s_contrast

        # Activity gate
        gate = (s_a > config.gate_audio_threshold) | (s_v > config.gate_video_threshold)
        raw_fused = (
            config.weight_audio * s_a
            + config.weight_video * s_v
            + config.weight_contrast * s_contrast
            + config.weight_sync * s_sync
        )
        out["fused_score"] = np.where(gate, raw_fused, 0.0)

    else:
        # Graceful audio-only fallback mode
        out["video_score"] = 0.0
        out["sync_score"] = 0.0
        s_contrast = compute_temporal_contrast_score(s_a, timestamps)
        out["contrast_score"] = s_contrast

        gate = s_a > config.gate_audio_threshold
        raw_fused = 0.75 * s_a + 0.25 * s_contrast
        out["fused_score"] = np.where(gate, raw_fused, 0.0)

        # Fill default video columns for CSV consistency
        for col in [
            "mean_frame_diff",
            "p95_frame_diff",
            "luminance_step",
            "hist_distance",
            "flow_mean",
            "flow_p95",
            "flow_area_ratio",
            "flow_divergence",
            "radial_expansion",
        ]:
            out[col] = 0.0
        
        out["time_to_contact_tau"] = 99.0

    return out


def extract_candidate_events(
    df_timeline: pd.DataFrame,
    config: MultimodalConfig = MultimodalConfig(),
) -> List[CandidateEvent]:
    """Pick non-overlapping transient peaks satisfying detection threshold and separation."""
    scores = df_timeline["fused_score"].values
    timestamps = df_timeline["timestamp"].values

    if len(scores) == 0:
        return []

    # Find all local peaks above detection threshold
    raw_peak_indices, _ = find_peaks(scores, height=config.detection_threshold)
    if len(raw_peak_indices) == 0:
        return []

    # Greedy non-maximum suppression: sort by score descending, breaking ties by earlier timestamp
    sorted_candidates = sorted(
        raw_peak_indices,
        key=lambda idx: (-scores[idx], timestamps[idx]),
    )

    selected_indices: List[int] = []
    for idx in sorted_candidates:
        t = timestamps[idx]
        if any(abs(t - timestamps[s]) < config.min_separation for s in selected_indices):
            continue
        selected_indices.append(idx)

    # Re-order selected peaks chronologically
    selected_indices.sort(key=lambda idx: timestamps[idx])

    candidates: List[CandidateEvent] = []
    for idx in selected_indices:
        row = df_timeline.iloc[idx]
        event = CandidateEvent(
            timestamp=float(row["timestamp"]),
            score=float(row["fused_score"]),
            audio_score=float(row["audio_score"]),
            video_score=float(row["video_score"]),
            sync_score=float(row["sync_score"]),
            contrast_score=float(row["contrast_score"]),
            rms_db=float(row["rms_db"]),
            rise_time_ms=float(row["rise_time_ms"]),
            spectral_flux=float(row.get("flux_high", 0.0) + row.get("flux_mid", 0.0)),
            mean_frame_diff=float(row.get("mean_frame_diff", 0.0)),
            flow_divergence=float(row.get("flow_divergence", 0.0)),
            radial_expansion=float(row.get("radial_expansion", 0.0)),
        )
        candidates.append(event)

    return candidates


def detect_multimodal_jumpscares(
    df_audio: pd.DataFrame,
    df_video: Optional[pd.DataFrame] = None,
    config: MultimodalConfig = MultimodalConfig(),
) -> Tuple[List[CandidateEvent], pd.DataFrame]:
    """Run full multimodal fusion pipeline and return detected events with audit timeline."""
    timeline = fuse_multimodal_timeline(df_audio, df_video, config)
    candidates = extract_candidate_events(timeline, config)
    return candidates, timeline
