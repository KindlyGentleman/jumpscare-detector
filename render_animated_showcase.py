"""Generate animated multimodal showcase video with IEEE scienceplots telemetry and synchronized A/V."""

from pathlib import Path
import subprocess
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
import scienceplots

from jumpscare_detector.media import load_audio


def render_telemetry_stages(
    df: pd.DataFrame,
    audio_pcm: np.ndarray,
    sr: int,
    fig_width: float = 12.0,
    fig_height: float = 9.2,
    dpi: int = 100,
):
    """Pre-render telemetry backgrounds for Stages 0, 1, and 2 using IEEE scienceplots."""
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    t = df["timestamp"].values

    # Normalize audio PCM and calculate smooth Hilbert envelope
    audio_max = np.max(np.abs(audio_pcm))
    norm_pcm = audio_pcm / audio_max if audio_max > 0 else audio_pcm
    analytic_env = np.abs(hilbert(norm_pcm))

    k_size = int(0.04 * sr)
    smooth_env = np.convolve(analytic_env, np.ones(k_size) / k_size, mode="same")

    ds_pcm = max(1, len(norm_pcm) // 4000)
    t_pcm = np.linspace(0, len(norm_pcm) / sr, len(norm_pcm))[::ds_pcm]
    val_pcm = np.abs(norm_pcm[::ds_pcm])

    ds_env = max(1, len(smooth_env) // 1500)
    t_env = np.linspace(0, len(smooth_env) / sr, len(smooth_env))[::ds_env]
    val_env = smooth_env[::ds_env]

    stage_images = {}
    geom = None

    for stage in [0, 1, 2]:
        fig, axes = plt.subplots(
            5,
            1,
            figsize=(fig_width, fig_height),
            dpi=dpi,
            sharex=True,
            gridspec_kw={
                "hspace": 0.18,
                "left": 0.14,
                "right": 0.96,
                "top": 0.98,
                "bottom": 0.11,
            },
        )
        ax1, ax2, ax3, ax4, ax5 = axes

        # Dynamic highlight bands according to stage
        for ax in axes:
            if stage >= 1:
                ax.axvspan(10.5, 12.0, color="#E69F00", alpha=0.18, zorder=0)
            if stage >= 2:
                ax.axvspan(23.5, 25.5, color="#D55E00", alpha=0.20, zorder=0)
            ax.grid(True, linestyle=":", alpha=0.6)

        # 1. Envelope & PCM - No in-plot legend
        ax1.plot(t_pcm, val_pcm, color="#cccccc", alpha=0.6, linewidth=0.8, linestyle="-")
        ax1.plot(t_env, val_env, color="#0072B2", linewidth=2.4, linestyle="-")
        ax1.set_ylabel("Envelope", color="#0072B2")
        ax1.set_ylim(-0.02, 1.15)

        # 2. Contrast Spike
        ax2.plot(t, df["rms_spike_db"], color="#D55E00", linewidth=2.2, linestyle="-")
        ax2.set_ylabel("Contrast (+dB)", color="#D55E00")
        ax2.set_ylim(-2, 32)
        if stage >= 2:
            ax2.scatter([24.38], [10.04], color="#D55E00", s=90, edgecolors="black", zorder=5)
            ax2.annotate(
                "Loudest Scream (-3.9 dBFS)",
                (24.38, 10.04),
                xytext=(12.0, 22),
                arrowprops=dict(facecolor="#D55E00", arrowstyle="->", lw=2.0),
                fontsize=13,
                fontweight="bold",
                color="#D55E00",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#D55E00", lw=1.5),
            )

        # 3. Spectral Flux
        ax3.plot(t, df["flux_low"], color="#009E73", linewidth=2.0, linestyle="-")
        ax3.plot(t, df["flux_mid"], color="#E69F00", linewidth=2.0, linestyle="-")
        ax3.plot(t, df["flux_high"], color="#CC79A7", linewidth=2.0, linestyle="-")
        ax3.set_ylabel("Spectral Flux", color="#111111")
        ax3.set_ylim(-0.02, 0.75)

        # 4. Computer Vision Radial Expansion
        ax4.plot(t, df["flow_mean"], color="#888888", linestyle=":", linewidth=1.4)
        ax4.plot(t, df["radial_expansion"], color="#0072B2", linewidth=2.4, linestyle="-")
        ax4.fill_between(t, 0, np.maximum(0, df["radial_expansion"]), color="#0072B2", alpha=0.18)
        ax4.set_ylabel("Radial (px)", color="#0072B2")
        ax4.set_ylim(-3, 44)
        if stage >= 1:
            ax4.scatter([10.91], [37.94], color="#0072B2", s=90, edgecolors="black", zorder=5)
            ax4.annotate(
                "Visual Rush (+37.9 px)",
                (10.91, 37.94),
                xytext=(12.5, 30),
                arrowprops=dict(facecolor="#0072B2", arrowstyle="->", lw=2.0),
                fontsize=13,
                fontweight="bold",
                color="#0072B2",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E1F5FE", edgecolor="#0072B2", lw=1.5),
            )

        # 5. Fused Transient Anomaly Score S(t)
        ax5.plot(t, df["fused_score"], color="#111111", linewidth=2.6, linestyle="-")
        ax5.axhline(1.8, color="#D55E00", linestyle="--", linewidth=2.2)
        ax5.text(
            33.5,
            1.95,
            "Threshold T = 1.8",
            color="#D55E00",
            fontsize=11.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D55E00", lw=1.2, alpha=0.9),
        )
        ax5.set_ylabel("Score S(t)", color="#111111")
        ax5.set_ylim(-0.15, 3.4)

        if stage >= 1:
            ax5.scatter([10.91], [2.81], color="#D55E00", s=100, edgecolors="black", zorder=6)
            ax5.annotate(
                "ANOMALY 1: Visual Shock (S=2.81)\n09:16 (Radial Rush)",
                (10.91, 2.81),
                xytext=(0.8, 2.5),
                arrowprops=dict(facecolor="#D55E00", arrowstyle="->", lw=2.2),
                fontsize=12,
                fontweight="bold",
                color="#D55E00",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF3E0", edgecolor="#D55E00", lw=1.8),
            )

        if stage >= 2:
            ax5.scatter([24.38], [1.81], color="#D55E00", s=100, edgecolors="black", zorder=6)
            ax5.annotate(
                "ANOMALY 2: PINNACLE JUMP SCARE\n09:29 | Scream -3.9 dBFS (S=1.81)",
                (24.38, 1.81),
                xytext=(25.5, 2.5),
                arrowprops=dict(facecolor="#D55E00", arrowstyle="->", lw=2.2),
                fontsize=11.5,
                fontweight="bold",
                color="#D55E00",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFEBEE", edgecolor="#D55E00", lw=1.8),
            )

        ax5.set_xlim(0, 40)
        sec_ticks = np.arange(0, 45, 5)
        clip_start_offset_sec = 545.0
        orig_labels = [
            f"{s:.0f}s\n({int((clip_start_offset_sec + s)//60):02d}:{int((clip_start_offset_sec + s)%60):02d})"
            for s in sec_ticks
        ]
        ax5.set_xticks(sec_ticks)
        ax5.set_xticklabels(orig_labels)
        ax5.set_xlabel("Clip Elapsed Time [seconds] (Original Video Time [mm:ss])", labelpad=10, fontsize=14, fontweight="bold")

        fig.canvas.draw()

        if geom is None:
            t0_x = ax1.transData.transform((0.0, 0))[0]
            t40_x = ax1.transData.transform((40.0, 0))[0]
            y_top = fig.bbox.height - ax1.transAxes.transform((0, 1))[1]
            y_bot = fig.bbox.height - ax5.transAxes.transform((0, 0))[1]
            geom = (t0_x, t40_x, y_top, y_bot)

        buf = fig.canvas.buffer_rgba()
        img_np = np.asarray(buf)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        stage_images[stage] = bgr

    return stage_images, geom


def build_animated_showcase(
    clip_path: str = "showcase_clip.mp4",
    audit_csv: str = "output/showcase_multimodal/jumpscare_signal_audit.csv",
    output_mp4: str = "output/showcase_multimodal_ieee.mp4",
    target_fps: float = 30.0,
) -> Path:
    """Generate master animated showcase video synchronized with original video and audio."""
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = out_path.parent / "temp_showcase_silent.mp4"

    # Load audio and audit telemetry
    df = pd.read_csv(audit_csv)
    audio_pcm, sr = load_audio(clip_path, target_sr=16000)

    # Pre-render telemetry backgrounds (Stages 0, 1, 2)
    print("Pre-rendering IEEE scienceplots telemetry stages...")
    stage_images, (t0_x, t40_x, y_top, y_bot) = render_telemetry_stages(
        df=df,
        audio_pcm=audio_pcm,
        sr=sr,
        fig_width=12.0,
        fig_height=9.2,
        dpi=100,
    )

    # Master canvas dimensions
    canvas_w = 1200
    canvas_h = 1620
    tel_h = stage_images[0].shape[0]
    tel_w = stage_images[0].shape[1]

    # Video placement: clean, no HUD overlays, no top header
    v_w = 1140
    v_h = 642
    v_x = 30
    v_y = 20
    tel_y = v_y + v_h + 20

    # Initialize video capture
    cap = cv2.VideoCapture(clip_path)
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    clip_duration = total_src_frames / src_fps
    total_out_frames = int(clip_duration * target_fps)

    print(f"Compositing {total_out_frames} frames ({clip_duration:.1f}s at {target_fps} fps)...")

    # Video writer for silent composite
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, target_fps, (canvas_w, canvas_h))

    for frame_idx in range(total_out_frames):
        t_curr = frame_idx / target_fps
        src_frame_idx = int(t_curr * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(src_frame_idx, total_src_frames - 1))
        ret, vframe = cap.read()
        if not ret:
            break

        # Select telemetry stage according to timestamp
        if t_curr < 10.91:
            stage_idx = 0
        elif t_curr < 24.38:
            stage_idx = 1
        else:
            stage_idx = 2

        # Clone telemetry background
        tel_frame = stage_images[stage_idx].copy()

        # Compute sweep needle X position
        cur_x = int(t0_x + (t_curr / 40.0) * (t40_x - t0_x))
        cur_x = max(int(t0_x), min(int(t40_x), cur_x))

        # Draw sweep line across all 5 subfigures
        cv2.line(tel_frame, (cur_x, int(y_top)), (cur_x, int(y_bot)), (200, 200, 255), 4, cv2.LINE_AA)
        cv2.line(tel_frame, (cur_x, int(y_top)), (cur_x, int(y_bot)), (0, 45, 215), 2, cv2.LINE_AA)

        # Draw cursor triangle at top of subfigures
        pt1 = (cur_x, int(y_top) - 1)
        pt2 = (cur_x - 6, int(y_top) - 10)
        pt3 = (cur_x + 6, int(y_top) - 10)
        cv2.fillPoly(tel_frame, [np.array([pt1, pt2, pt3], np.int32)], (0, 45, 215))

        # Create master frame
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

        # Place resized pure video with crisp border
        vframe_resized = cv2.resize(vframe, (v_w, v_h), interpolation=cv2.INTER_AREA)
        canvas[v_y : v_y + v_h, v_x : v_x + v_w] = vframe_resized
        cv2.rectangle(canvas, (v_x - 2, v_y - 2), (v_x + v_w + 1, v_y + v_h + 1), (45, 45, 45), 2)

        # Place telemetry below
        canvas[tel_y : tel_y + tel_h, 0 : tel_w] = tel_frame

        writer.write(canvas)

    cap.release()
    writer.release()
    print("Video frame composition complete!")

    # Remux audio using FFmpeg
    print("Remuxing original AAC audio track with FFmpeg...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", str(temp_video),
        "-i", str(clip_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_path),
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    if temp_video.exists():
        temp_video.unlink()

    print(f"Master animated showcase successfully generated at: {out_path}")
    return out_path


if __name__ == "__main__":
    build_animated_showcase()
