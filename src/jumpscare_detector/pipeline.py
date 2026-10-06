"""End-to-end execution pipeline for explainable multimodal jump scare detection."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from jumpscare_detector.audio_transient import (
    AudioTransientConfig,
    extract_audio_transients,
)
from jumpscare_detector.media import load_audio
from jumpscare_detector.multimodal import (
    CandidateEvent,
    MultimodalConfig,
    detect_multimodal_jumpscares,
)
from jumpscare_detector.plotting import generate_diagnostic_plot
from jumpscare_detector.video import VideoConfig, extract_video_transients


def run_multimodal_pipeline(
    media_path: Union[str, Path],
    video_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "output",
    audio_config: AudioTransientConfig = AudioTransientConfig(),
    video_config: VideoConfig = VideoConfig(),
    multimodal_config: MultimodalConfig = MultimodalConfig(),
    generate_plot: bool = True,
) -> Dict[str, Any]:
    """Execute complete analysis pipeline across acoustic and visual channels."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(media_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input media path does not exist: {input_path}")

    # Determine visual source if available
    v_source: Optional[Path] = None
    if video_path is not None:
        v_cand = Path(video_path)
        if v_cand.exists():
            v_source = v_cand
    elif input_path.suffix.lower() in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
        v_source = input_path

    # Extract and analyze audio channel
    audio_data, sr = load_audio(input_path, target_sr=audio_config.sample_rate)
    df_audio, sr = extract_audio_transients(audio_data, sample_rate=sr, config=audio_config)

    # Extract and analyze video channel if present
    df_video: Optional[pd.DataFrame] = None
    if v_source is not None:
        df_video, _ = extract_video_transients(v_source, config=video_config)

    # Perform multimodal synchrony fusion
    candidates, timeline = detect_multimodal_jumpscares(
        df_audio=df_audio,
        df_video=df_video,
        config=multimodal_config,
    )

    # Export signal audit and candidate CSVs
    audit_csv_path = out_dir / "jumpscare_signal_audit.csv"
    timeline.to_csv(audit_csv_path, index=False)

    cand_records = [asdict(c) for c in candidates]
    df_candidates = pd.DataFrame.from_records(cand_records)
    candidates_csv_path = out_dir / "jumpscare_candidates.csv"
    df_candidates.to_csv(candidates_csv_path, index=False)

    plot_file: Optional[Path] = None
    if generate_plot:
        plot_file = out_dir / "jumpscare_diagnostic_audit.png"
        generate_diagnostic_plot(
            df_timeline=timeline,
            candidates=candidates,
            output_path=plot_file,
            audio_waveform=audio_data,
            sample_rate=sr,
            threshold=multimodal_config.detection_threshold,
        )

    return {
        "candidate_count": len(candidates),
        "candidates": cand_records,
        "timeline_rows": len(timeline),
        "audit_csv": str(audit_csv_path),
        "candidates_csv": str(candidates_csv_path),
        "diagnostic_plot": str(plot_file) if plot_file else None,
    }
