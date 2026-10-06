"""Generate comprehensive multimodal showcase figure with video filmstrip, audio waveform, and transient subfigures."""

from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jumpscare_detector.media import load_audio


def generate_showcase_figure(
    video_path: str = "showcase_clip.mp4",
    audit_csv: str = "output/showcase_multimodal/jumpscare_signal_audit.csv",
    output_png: str = "output/showcase_multimodal/showcase_all_in_one.png",
    clip_start_offset_sec: float = 545.0,  # 09:05 in original video
) -> Path:
    """Render unified publication-quality showcase figure containing video strip and aligned telemetry."""
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_csv)
    t = df["timestamp"].values

    # Load audio waveform
    audio, sr = load_audio(video_path, target_sr=16000)
    audio_t = np.linspace(0, len(audio) / sr, len(audio))
    ds = max(1, len(audio) // 10000)
    audio_sub_t = audio_t[::ds]
    audio_sub_val = audio[::ds]

    # Extract 6 key video frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_timestamps = [5.0, 10.91, 18.0, 23.85, 24.50, 32.0]
    frame_titles = [
        "Buildup (09:10)",
        "Visual Shock (09:16)",
        "Quiet Stalking (09:23)",
        "Red Figure Seen (09:29)",
        "PINNACLE ROAR (09:30)",
        "Aftermath (09:37)",
    ]
    frame_images = []

    for f_t in frame_timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_t * fps))
        ret, frame = cap.read()
        if ret:
            # Crop black pillarbox sidebars (cols 310 to 1250)
            cropped = frame[:, 310:1250]
            frame_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            frame_images.append(frame_rgb)
        else:
            frame_images.append(np.zeros((720, 940, 3), dtype=np.uint8))
    cap.release()

    # Define anomaly highlight windows
    anom1_start, anom1_end = 10.5, 12.0  # Visual flash anomaly
    anom2_start, anom2_end = 23.5, 25.5  # Pinnacle jump scare anomaly

    # Construct overall figure layout
    fig = plt.figure(figsize=(16, 18), dpi=150)
    gs = gridspec.GridSpec(
        nrows=7,
        ncols=6,
        figure=fig,
        height_ratios=[1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2],
        hspace=0.28,
        wspace=0.10,
    )

    # Section 1: Video Filmstrip Keyframes
    for col_idx, (img, title, f_t) in enumerate(zip(frame_images, frame_titles, frame_timestamps)):
        ax_frame = fig.add_subplot(gs[0, col_idx])
        ax_frame.imshow(img)
        ax_frame.set_xticks([])
        ax_frame.set_yticks([])
        is_pinnacle = "PINNACLE" in title or "Visual" in title
        border_color = "#d62728" if is_pinnacle else "#4a4a4a"
        for spine in ax_frame.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2.5 if is_pinnacle else 1.0)
        ax_frame.set_title(f"{title}\nt = {f_t:.1f}s", fontsize=9, fontweight="bold", color=border_color, pad=4)

    # Section 2: Shared telemetry time-series plots
    axes_telemetry = []
    for row_idx in range(1, 7):
        ax = fig.add_subplot(gs[row_idx, :])
        axes_telemetry.append(ax)

    # Helper function to paint anomaly highlight bands
    def apply_anomaly_highlights(ax):
        ax.axvspan(anom1_start, anom1_end, color="#ff7f0e", alpha=0.16, label="Anomaly 1 (Visual Shock)")
        ax.axvspan(anom2_start, anom2_end, color="#d62728", alpha=0.20, label="Anomaly 2 (Pinnacle Shock)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xlim(0.0, 40.0)

    # Subfigure 1: Waveform and Envelope
    ax0 = axes_telemetry[0]
    ax0.plot(audio_sub_t, audio_sub_val, color="#8c8c8c", alpha=0.6, label="Raw Audio PCM")
    ax0.plot(t, df["envelope_peak"], color="#1f77b4", linewidth=1.5, label="Analytic Hilbert Envelope")
    apply_anomaly_highlights(ax0)
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Subfigure 1: Audio Waveform & Analytic Amplitude Envelope", fontsize=10.5, fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8)

    # Subfigure 2: Decibels and Contrast Spike
    ax1 = axes_telemetry[1]
    ax1.plot(t, df["rms_db"], color="#1f77b4", label="RMS Level (dBFS)")
    ax1.set_ylabel("RMS (dBFS)", color="#1f77b4")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, df["rms_spike_db"], color="#d62728", linestyle="--", label="Local Spike over Baseline (+dB)")
    ax1_twin.set_ylabel("Contrast (+dB)", color="#d62728")
    apply_anomaly_highlights(ax1)
    ax1.set_title("Subfigure 2: Acoustic Level & Trailing Dynamic Contrast Spike", fontsize=10.5, fontweight="bold")
    # Mark pinnacle peak on decibel plot
    ax1.scatter([24.38], [-3.95], color="#d62728", s=65, zorder=5)
    ax1.annotate("Loudest Scream (-3.9 dB)", (24.38, -3.95), xytext=(25.5, -14),
                 arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.5),
                 fontsize=8, fontweight="bold", color="#d62728")

    # Subfigure 3: Rise Time & Rise Score
    ax2 = axes_telemetry[2]
    ax2.plot(t, df["rise_time_ms"], color="#2ca02c", label="10-90% Rise Time (ms)")
    ax2.set_ylabel("Rise Time (ms)", color="#2ca02c")
    ax2.set_yscale("log")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, df["rise_score"], color="#ff7f0e", linestyle="-.", label="Hilbert Rise Score")
    ax2_twin.set_ylabel("Rise Score", color="#ff7f0e")
    apply_anomaly_highlights(ax2)
    ax2.set_title("Subfigure 3: Envelope Rise Time (ms) & Dimensionless Rise Score", fontsize=10.5, fontweight="bold")

    # Subfigure 4: Multi-Band Spectral Flux
    ax3 = axes_telemetry[3]
    ax3.plot(t, df["flux_low"], color="#17becf", label="Low Band (20-250 Hz)")
    ax3.plot(t, df["flux_mid"], color="#bcbd22", label="Mid Band (250-2k Hz)")
    ax3.plot(t, df["flux_high"], color="#9467bd", label="High Band (2k-10k Hz)")
    ax3.plot(t, df["onset_strength"], color="#8c564b", linestyle=":", label="Onset Strength")
    apply_anomaly_highlights(ax3)
    ax3.set_ylabel("Normalized Flux")
    ax3.set_title("Subfigure 4: Multi-Band Positive Spectral Flux & Onset Strength", fontsize=10.5, fontweight="bold")
    ax3.legend(loc="upper right", ncol=4, fontsize=8)

    # Subfigure 5: Computer Vision Metrics
    ax4 = axes_telemetry[4]
    ax4.plot(t, df["mean_frame_diff"], color="#1f77b4", label="Mean Frame Diff")
    ax4.plot(t, df["flow_mean"], color="#7f7f7f", linestyle=":", label="Flow Magnitude")
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t, df["radial_expansion"], color="#d62728", linewidth=1.5, label="Radial Expansion E_rad")
    ax4_twin.set_ylabel("Radial Expansion (px)", color="#d62728")
    apply_anomaly_highlights(ax4)
    ax4.set_ylabel("Pixel Delta", color="#1f77b4")
    ax4.set_title("Subfigure 5: Optical Flow Motion & Directional Radial Expansion", fontsize=10.5, fontweight="bold")
    # Mark visual explosion at 10.91s
    ax4_twin.scatter([10.91], [37.94], color="#d62728", s=70, zorder=5)
    ax4_twin.annotate("Visual Screen Rush (+37.9 px)", (10.91, 37.94), xytext=(12.0, 31),
                      arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.5),
                      fontsize=8, fontweight="bold", color="#d62728")

    # Subfigure 6: Fused Anomaly Score
    ax5 = axes_telemetry[5]
    ax5.plot(t, df["fused_score"], color="black", linewidth=2.0, label="Fused Score S(t)")
    ax5.axhline(1.8, color="#d62728", linestyle="--", linewidth=1.5, label="Detection Threshold (T=1.8)")
    apply_anomaly_highlights(ax5)
    ax5.set_ylim(-0.2, 3.5)

    # Anomaly indicator dots
    ax5.scatter([10.91], [2.81], color="#d62728", s=90, edgecolors="black", zorder=6)
    ax5.annotate("ANOMALY 1: Visual Shock (S=2.81)\n09:16 (Radial Rush)", (10.91, 2.81), xytext=(3.0, 2.7),
                 arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.5),
                 fontsize=8, fontweight="bold", color="#d62728")

    ax5.scatter([24.38], [1.81], color="#d62728", s=90, edgecolors="black", zorder=6)
    ax5.annotate("ANOMALY 2: PINNACLE JUMPSCARE (S=1.81)\n09:29 (Scream: -3.9 dB)", (24.38, 1.81), xytext=(25.5, 2.5),
                 arrowprops=dict(facecolor="#d62728", arrowstyle="->", lw=1.5),
                 fontsize=8, fontweight="bold", color="#d62728")

    ax5.set_ylabel("Fused Score S(t)")
    ax5.set_xlabel("Clip Elapsed Time (seconds)", fontsize=10, fontweight="bold")
    ax5.set_title("Subfigure 6: Explainable Multimodal Transient Score with Highlighted Shock Anomalies", fontsize=10.5, fontweight="bold")
    ax5.legend(loc="upper right", fontsize=8)

    # Bottom timestamp formatting with original video times
    sec_ticks = np.arange(0, 45, 5)
    orig_labels = [f"{s:.0f}s\n({int((clip_start_offset_sec + s)//60):02d}:{int((clip_start_offset_sec + s)%60):02d})" for s in sec_ticks]
    ax5.set_xticks(sec_ticks)
    ax5.set_xticklabels(orig_labels, fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Showcase figure successfully created at: {out_path}")
    return out_path


if __name__ == "__main__":
    generate_showcase_figure()
