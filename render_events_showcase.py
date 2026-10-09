"""Generate multi-event jumpscare showcase video focusing on +/- 5s around detected timestamps."""

import argparse
from pathlib import Path
import shutil
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


def render_event_telemetry(
    df_slice: pd.DataFrame,
    slice_pcm: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
    t_peak: float,
    peak_score: float,
    fig_width: float = 12.0,
    fig_height: float = 8.5,
    dpi: int = 100,
):
    """Pre-render telemetry background for a specific event window [t_start, t_end]."""
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    t = df_slice["timestamp"].values

    # Normalize audio PCM and calculate smooth Hilbert envelope
    pcm_clean = np.nan_to_num(slice_pcm)
    pcm_max = np.max(np.abs(pcm_clean))
    norm_pcm = pcm_clean / pcm_max if pcm_max > 0 else pcm_clean
    analytic_env = np.abs(hilbert(norm_pcm))

    k_size = max(1, int(0.04 * sr))
    smooth_env = np.convolve(analytic_env, np.ones(k_size) / k_size, mode="same")

    ds_pcm = max(1, len(norm_pcm) // 2000)
    t_pcm = np.linspace(t_start, t_end, len(norm_pcm))[::ds_pcm]
    val_pcm = np.abs(norm_pcm[::ds_pcm])

    ds_env = max(1, len(smooth_env) // 1000)
    t_env = np.linspace(t_start, t_end, len(smooth_env))[::ds_env]
    val_env = smooth_env[::ds_env]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(fig_width, fig_height),
        dpi=dpi,
        sharex=True,
        gridspec_kw={
            "hspace": 0.20,
            "left": 0.16,
            "right": 0.96,
            "top": 0.97,
            "bottom": 0.12,
        },
    )
    ax1, ax2, ax3, ax4, ax5 = axes

    # Highlight band around detected peak (+-0.75s)
    for ax in axes:
        ax.axvspan(max(t_start, t_peak - 0.75), min(t_end, t_peak + 0.75), color="#D55E00", alpha=0.15, zorder=0)
        ax.grid(True, linestyle=":", alpha=0.6)

    # 1. Envelope & PCM
    ax1.plot(t_pcm, val_pcm, color="#cccccc", alpha=0.6, linewidth=0.8, linestyle="-")
    ax1.plot(t_env, val_env, color="#0072B2", linewidth=2.2, linestyle="-")
    ax1.set_ylabel("Envelope", color="#0072B2")
    ax1.set_ylim(-0.02, 1.15)

    # 2. Contrast Spike
    rms_vals = df_slice["rms_spike_db"].values
    ax2.plot(t, rms_vals, color="#D55E00", linewidth=2.2, linestyle="-")
    ax2.set_ylabel("Contrast (+dB)", color="#D55E00")
    ax2.set_ylim(-2, max(28, float(np.max(rms_vals)) + 6) if len(rms_vals) > 0 else 28)

    # 3. Spectral Flux
    f_low = df_slice["flux_low"].values
    f_mid = df_slice["flux_mid"].values
    f_high = df_slice["flux_high"].values
    ax3.plot(t, f_low, color="#009E73", linewidth=1.8, label="Low")
    ax3.plot(t, f_mid, color="#E69F00", linewidth=1.8, label="Mid")
    ax3.plot(t, f_high, color="#CC79A7", linewidth=1.8, label="High")
    ax3.set_ylabel("Spectral Flux", color="#111111")
    max_flux = max(0.5, float(max(np.max(f_low), np.max(f_mid), np.max(f_high))) * 1.2) if len(f_low) > 0 else 0.5
    ax3.set_ylim(-0.02, max_flux)

    # 4. Computer Vision Radial Expansion
    flow_mean = df_slice["flow_mean"].values
    rad_exp = df_slice["radial_expansion"].values
    ax4.plot(t, flow_mean, color="#888888", linestyle=":", linewidth=1.4)
    ax4.plot(t, rad_exp, color="#0072B2", linewidth=2.2, linestyle="-")
    ax4.fill_between(t, 0, np.maximum(0, rad_exp), color="#0072B2", alpha=0.18)
    ax4.set_ylabel("Radial (px)", color="#0072B2")
    min_rad = min(-2, float(np.min(rad_exp)) - 2) if len(rad_exp) > 0 else -2
    max_rad = max(20, float(np.max(rad_exp)) + 5) if len(rad_exp) > 0 else 20
    ax4.set_ylim(min_rad, max_rad)

    # 5. Fused Transient Anomaly Score S(t)
    fused_vals = df_slice["fused_score"].values
    ax5.plot(t, fused_vals, color="#111111", linewidth=2.4, linestyle="-")
    ax5.axhline(1.95, color="#D55E00", linestyle="--", linewidth=2.0)
    ax5.scatter([t_peak], [peak_score], color="#D55E00", s=90, edgecolors="black", zorder=6)
    
    # Peak Callout Annotation
    m_p, s_p = int(t_peak // 60), t_peak % 60
    ax5.annotate(
        f"EVENT PEAK ({m_p:02d}:{s_p:04.1f})\nS={peak_score:.2f}",
        (t_peak, peak_score),
        xytext=(max(t_start + 0.5, t_peak - 2.8), min(peak_score + 0.6, 3.8)),
        arrowprops=dict(facecolor="#D55E00", arrowstyle="->", lw=1.8),
        fontsize=11.5,
        fontweight="bold",
        color="#D55E00",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF3E0", edgecolor="#D55E00", lw=1.4),
    )

    ax5.set_ylabel("Score S(t)", color="#111111")
    ax5.set_ylim(-0.15, max(3.5, peak_score + 0.8))

    # X-Axis limits & ticks
    ax5.set_xlim(t_start, t_end)
    step = 2.0 if (t_end - t_start) >= 8.0 else 1.0
    sec_ticks = np.arange(np.ceil(t_start), np.floor(t_end) + 0.1, step)
    labels = []
    for s in sec_ticks:
        rel = s - t_peak
        rel_str = f"{rel:+.1f}s" if abs(rel) > 0.05 else "PEAK"
        labels.append(f"{rel_str}\n({int(s//60):02d}:{s%60:04.1f})")
    ax5.set_xticks(sec_ticks)
    ax5.set_xticklabels(labels)
    ax5.set_xlabel("Relative Timeline [seconds] (Video Time [mm:ss])", labelpad=8, fontsize=13, fontweight="bold")

    fig.canvas.draw()

    # Geometry for needle sweep line
    t0_x = ax1.transData.transform((t_start, 0))[0]
    t1_x = ax1.transData.transform((t_end, 0))[0]
    y_top = fig.bbox.height - ax1.transAxes.transform((0, 1))[1]
    y_bot = fig.bbox.height - ax5.transAxes.transform((0, 0))[1]
    geom = (t0_x, t1_x, y_top, y_bot)

    buf = fig.canvas.buffer_rgba()
    img_np = np.asarray(buf)
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    return bgr, geom


def select_prominent_events(
    candidates_csv: str,
    top_n: int = 12,
    min_separation: float = 10.0,
    min_score: float = 2.8,
) -> list[dict]:
    """Select distinct top jumpscare moments sorted chronologically."""
    df = pd.read_csv(candidates_csv)
    sub = df[df["score"] >= min_score].sort_values("score", ascending=False)
    
    distinct = []
    for _, row in sub.iterrows():
        t = row["timestamp"]
        if not any(abs(t - e["timestamp"]) < min_separation for e in distinct):
            distinct.append(row.to_dict())
        if len(distinct) >= top_n:
            break

    # Sort chronologically for natural video flow
    distinct.sort(key=lambda x: x["timestamp"])
    return distinct


def build_events_showcase(
    video_path: str = "youtube_test.mp4",
    audit_csv: str = "output/youtube_test/jumpscare_signal_audit.csv",
    candidates_csv: str = "output/youtube_test/jumpscare_candidates.csv",
    output_mp4: str = "output/youtube_test/showcase_detected_moments.mp4",
    window_half_sec: float = 5.0,
    top_n: int = 12,
    min_score: float = 2.8,
    target_fps: float = 30.0,
) -> Path:
    """Renders a unified showcase video jumping between all prominent detected events with 5-track telemetry."""
    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / "temp_segments"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading candidate events from: {candidates_csv}...")
    events = select_prominent_events(candidates_csv, top_n=top_n, min_score=min_score)
    print(f"Selected {len(events)} distinct candidate events for showcase:")
    for i, ev in enumerate(events):
        m, s = int(ev['timestamp'] // 60), ev['timestamp'] % 60
        print(f"  Event #{i+1:02d}: {m:02d}:{s:04.1f} | Score: {ev['score']:.2f} (A: {ev['audio_score']:.2f}, V: {ev['video_score']:.2f}, Sync: {ev['sync_score']:.2f})")

    # Load telemetry and audio
    print(f"Loading signal telemetry: {audit_csv}...")
    df_audit = pd.read_csv(audit_csv)
    print(f"Loading full audio waveform from: {video_path}...")
    audio_pcm, sr = load_audio(video_path, target_sr=16000)

    # Master canvas dimensions
    canvas_w = 1200
    canvas_h = 1520
    v_w = 1140
    v_h = 642
    v_x = 30
    v_y = 85
    tel_y = v_y + v_h + 15

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_src_frames / src_fps

    segment_files = []

    for idx, ev in enumerate(events):
        t_peak = ev["timestamp"]
        peak_score = ev["score"]
        t_start = max(0.0, t_peak - window_half_sec)
        t_end = min(video_duration, t_peak + window_half_sec)
        seg_duration = t_end - t_start

        m_p, s_p = int(t_peak // 60), t_peak % 60
        print(f"\n--- Processing Event #{idx+1}/{len(events)} at {m_p:02d}:{s_p:04.1f} (Window {t_start:.1f}s - {t_end:.1f}s) ---")

        # Telemetry slice
        df_slice = df_audit[(df_audit["timestamp"] >= t_start) & (df_audit["timestamp"] <= t_end)]
        idx_pcm_start = max(0, int(t_start * sr))
        idx_pcm_end = min(len(audio_pcm), int(t_end * sr))
        slice_pcm = audio_pcm[idx_pcm_start:idx_pcm_end]

        # Render background telemetry
        tel_bgr, (t0_x, t1_x, y_top, y_bot) = render_event_telemetry(
            df_slice=df_slice,
            slice_pcm=slice_pcm,
            sr=sr,
            t_start=t_start,
            t_end=t_end,
            t_peak=t_peak,
            peak_score=peak_score,
            fig_width=12.0,
            fig_height=7.6,
            dpi=100,
        )
        tel_h, tel_w = tel_bgr.shape[0], tel_bgr.shape[1]

        # Prepare segment video output
        temp_seg_raw = temp_dir / f"seg_raw_{idx:03d}.mp4"
        temp_seg_audio = temp_dir / f"seg_audio_{idx:03d}.m4a"
        temp_seg_muxed = temp_dir / f"seg_mux_{idx:03d}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_seg_raw), fourcc, target_fps, (canvas_w, canvas_h))

        seg_frames = int(seg_duration * target_fps)
        start_frame_idx = int(t_start * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(start_frame_idx, total_src_frames - 1))

        for f_idx in range(seg_frames):
            t_curr = t_start + (f_idx / target_fps)
            ret, vframe = cap.read()
            if not ret:
                break

            # 1. Base canvas with dark modern aesthetic
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

            # 2. Header banner (dark slate blue)
            cv2.rectangle(canvas, (0, 0), (canvas_w, 75), (32, 28, 24), -1)
            header_txt = f"JUMPSCARE DETECTION SHOWCASE  |  EVENT #{idx+1} OF {len(events)}"
            cv2.putText(canvas, header_txt, (30, 32), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
            sub_txt = f"Timestamp: {m_p:02d}:{s_p:04.1f}   |   Fused Score: S={peak_score:.2f}   |   Audio: {ev['audio_score']:.2f}   |   Video: {ev['video_score']:.2f}   |   Sync: {ev['sync_score']:.2f}"
            cv2.putText(canvas, sub_txt, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (72, 190, 245), 1, cv2.LINE_AA)

            # 3. Video frame placement
            vframe_resized = cv2.resize(vframe, (v_w, v_h), interpolation=cv2.INTER_AREA)
            canvas[v_y : v_y + v_h, v_x : v_x + v_w] = vframe_resized
            cv2.rectangle(canvas, (v_x - 2, v_y - 2), (v_x + v_w + 1, v_y + v_h + 1), (50, 50, 50), 2)

            # Live timer pill on video
            cur_m, cur_s = int(t_curr // 60), t_curr % 60
            pill_text = f"TIME: {cur_m:02d}:{cur_s:04.1f}"
            cv2.rectangle(canvas, (v_x + 15, v_y + 15), (v_x + 190, v_y + 50), (0, 0, 0), -1)
            cv2.rectangle(canvas, (v_x + 15, v_y + 15), (v_x + 190, v_y + 50), (0, 70, 220), 2)
            cv2.putText(canvas, pill_text, (v_x + 28, v_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # 4. Telemetry with live needle
            tel_frame = tel_bgr.copy()
            cur_needle_x = int(t0_x + ((t_curr - t_start) / (t_end - t_start)) * (t1_x - t0_x))
            cur_needle_x = max(int(t0_x), min(int(t1_x), cur_needle_x))

            # Vertical needle line
            cv2.line(tel_frame, (cur_needle_x, int(y_top)), (cur_needle_x, int(y_bot)), (210, 210, 255), 4, cv2.LINE_AA)
            cv2.line(tel_frame, (cur_needle_x, int(y_top)), (cur_needle_x, int(y_bot)), (0, 40, 220), 2, cv2.LINE_AA)

            # Needle pointer triangle
            pt1 = (cur_needle_x, int(y_top) - 1)
            pt2 = (cur_needle_x - 6, int(y_top) - 9)
            pt3 = (cur_needle_x + 6, int(y_top) - 9)
            cv2.fillPoly(tel_frame, [np.array([pt1, pt2, pt3], np.int32)], (0, 40, 220))

            canvas[tel_y : tel_y + tel_h, 0 : tel_w] = tel_frame
            writer.write(canvas)

        writer.release()

        # Extract matching audio segment using FFmpeg
        cmd_audio = [
            "ffmpeg", "-y",
            "-ss", f"{t_start:.3f}",
            "-t", f"{seg_duration:.3f}",
            "-i", str(video_path),
            "-vn",
            "-c:a", "aac",
            "-b:a", "192k",
            str(temp_seg_audio),
        ]
        subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Mux segment video + audio
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", str(temp_seg_raw),
            "-i", str(temp_seg_audio),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            str(temp_seg_muxed),
        ]
        subprocess.run(cmd_mux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        segment_files.append(temp_seg_muxed)
        print(f"Segment #{idx+1} rendered and muxed ({seg_duration:.1f}s).")

    cap.release()

    # Concatenate all segments using FFmpeg concat demuxer
    concat_list = temp_dir / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for sf in segment_files:
            # Use relative filename so ffmpeg concat demuxer resolves cleanly
            f.write(f"file '{sf.name}'\n")

    print("\nConcatenating all event segments into master showcase video...")
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Clean up temporary segments
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n=======================================================")
    print(f"SUCCESS: Master showcase video generated successfully!")
    print(f"Location: {out_path.resolve()}")
    print(f"=======================================================")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render animated showcase for detected jumpscare events.")
    parser.add_argument("--video", default="youtube_test.mp4", help="Source video file.")
    parser.add_argument("--audit-csv", default="output/youtube_test/jumpscare_signal_audit.csv", help="Audit telemetry CSV.")
    parser.add_argument("--candidates-csv", default="output/youtube_test/jumpscare_candidates.csv", help="Candidates CSV.")
    parser.add_argument("--output", default="output/youtube_test/showcase_detected_moments.mp4", help="Output MP4 showcase path.")
    parser.add_argument("--top-n", type=int, default=12, help="Number of distinct events to render.")
    parser.add_argument("--min-score", type=float, default=2.8, help="Minimum candidate score to consider.")
    parser.add_argument("--window", type=float, default=5.0, help="Half-window in seconds (+/- seconds).")
    args = parser.parse_args()

    build_events_showcase(
        video_path=args.video,
        audit_csv=args.audit_csv,
        candidates_csv=args.candidates_csv,
        output_mp4=args.output,
        window_half_sec=args.window,
        top_n=args.top_n,
        min_score=args.min_score,
    )
