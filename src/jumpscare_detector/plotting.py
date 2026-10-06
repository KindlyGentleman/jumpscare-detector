"""Diagnostic plotting for explainable multimodal jump scare detection."""

from pathlib import Path
from typing import List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jumpscare_detector.multimodal import CandidateEvent


def generate_diagnostic_plot(
    df_timeline: pd.DataFrame,
    candidates: List[CandidateEvent],
    output_path: Union[str, Path],
    audio_waveform: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
    threshold: float = 2.2,
) -> Path:
    """Generate 6-panel aligned diagnostic plot showing acoustic, visual, and fused metrics."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timestamps = df_timeline["timestamp"].values
    t_min = timestamps[0] if len(timestamps) > 0 else 0.0
    t_max = timestamps[-1] if len(timestamps) > 0 else 1.0

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    candidate_times = [c.timestamp for c in candidates]

    # Panel 1: Audio Waveform and Onsets
    ax0 = axes[0]
    if audio_waveform is not None and len(audio_waveform) > 0:
        time_audio = np.linspace(0, len(audio_waveform) / sample_rate, len(audio_waveform))
        # Downsample waveform for plotting speed
        ds_factor = max(1, len(audio_waveform) // 20000)
        ax0.plot(time_audio[::ds_factor], audio_waveform[::ds_factor], color="lightgray", label="Waveform")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Panel 1: Audio Waveform and Candidate Onsets")
    for ct in candidate_times:
        ax0.axvline(ct, color="red", linestyle="--", alpha=0.7)
    ax0.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Acoustic Level and Rise Score
    ax1 = axes[1]
    ax1.plot(timestamps, df_timeline["rms_db"], color="#1f77b4", label="RMS Level (dB)")
    ax1.set_ylabel("RMS (dB)", color="#1f77b4")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(timestamps, df_timeline["rise_score"], color="#ff7f0e", linestyle="--", label="Rise Score")
    ax1_twin.set_ylabel("Rise Score", color="#ff7f0e")
    ax1.set_title("Panel 2: Audio RMS Level and Hilbert Rise Score")
    for ct in candidate_times:
        ax1.axvline(ct, color="red", linestyle="--", alpha=0.7)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 3: Multi-Band Spectral Flux
    ax2 = axes[2]
    ax2.plot(timestamps, df_timeline.get("flux_low", np.zeros_like(timestamps)), color="#2ca02c", label="Low (20-250 Hz)")
    ax2.plot(timestamps, df_timeline.get("flux_mid", np.zeros_like(timestamps)), color="#d62728", label="Mid (250-2k Hz)")
    ax2.plot(timestamps, df_timeline.get("flux_high", np.zeros_like(timestamps)), color="#9467bd", label="High (2k-10k Hz)")
    ax2.plot(timestamps, df_timeline.get("onset_strength", np.zeros_like(timestamps)), color="#8c564b", linestyle=":", label="Onset Strength")
    ax2.set_ylabel("Flux / Onset")
    ax2.set_title("Panel 3: Multi-Band Positive Spectral Flux and Onset Strength")
    ax2.legend(loc="upper right", ncol=4, fontsize=9)
    for ct in candidate_times:
        ax2.axvline(ct, color="red", linestyle="--", alpha=0.7)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Check whether visual metrics are present
    has_video = (
        "mean_frame_diff" in df_timeline.columns
        and np.max(df_timeline["mean_frame_diff"].values) > 0.0
    )

    # Panel 4: Visual Frame Difference and Luminance Step
    ax3 = axes[3]
    if has_video:
        ax3.plot(timestamps, df_timeline["mean_frame_diff"], color="#17becf", label="Mean Frame Diff")
        ax3.plot(timestamps, df_timeline["p95_frame_diff"], color="#bcbd22", linestyle=":", label="P95 Frame Diff")
        ax3.plot(timestamps, df_timeline["luminance_step"], color="#e377c2", linestyle="--", label="Luminance Step")
        ax3.set_ylabel("Pixel Delta")
        ax3.legend(loc="upper right", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Video channel absent (Audio transient mode active)", ha="center", va="center", transform=ax3.transAxes, color="gray")
        ax3.set_ylabel("N/A")
    ax3.set_title("Panel 4: Visual Contrast and Frame Differences")
    for ct in candidate_times:
        ax3.axvline(ct, color="red", linestyle="--", alpha=0.7)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Panel 5: Optical Flow Dynamics
    ax4 = axes[4]
    if has_video:
        ax4.plot(timestamps, df_timeline["flow_mean"], color="#1f77b4", label="Mean Flow Mag")
        ax4.plot(timestamps, df_timeline["flow_divergence"], color="#d62728", linestyle="--", label="Flow Divergence")
        ax4.plot(timestamps, df_timeline["radial_expansion"], color="#2ca02c", linestyle="-.", label="Radial Expansion")
        ax4.set_ylabel("Flow (px)")
        ax4.legend(loc="upper right", fontsize=9)
    else:
        ax4.text(0.5, 0.5, "Video channel absent (Audio transient mode active)", ha="center", va="center", transform=ax4.transAxes, color="gray")
        ax4.set_ylabel("N/A")
    ax4.set_title("Panel 5: Dense Optical Flow Magnitude, Divergence, and Radial Expansion")
    for ct in candidate_times:
        ax4.axvline(ct, color="red", linestyle="--", alpha=0.7)
    ax4.grid(True, linestyle=":", alpha=0.5)

    # Panel 6: Fused Candidate Score
    ax5 = axes[5]
    ax5.plot(timestamps, df_timeline["fused_score"], color="black", linewidth=1.5, label="Fused Score S(t)")
    ax5.axhline(threshold, color="crimson", linestyle="--", label=f"Detection Threshold (T={threshold:.1f})")
    for cand in candidates:
        ax5.plot(cand.timestamp, cand.score, "ro", markersize=7)
        ax5.annotate(
            f"{cand.timestamp:.2f}s\n(S={cand.score:.1f})",
            (cand.timestamp, cand.score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="red",
        )
    ax5.set_ylabel("Fused Score S(t)")
    ax5.set_xlabel("Time (seconds)")
    ax5.set_title("Panel 6: Multimodal Fused Candidate Score with Detection Threshold")
    ax5.legend(loc="upper right", fontsize=9)
    ax5.set_xlim(t_min, t_max)
    ax5.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

    return out_file
