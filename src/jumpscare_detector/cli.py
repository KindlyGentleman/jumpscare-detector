"""Command line interface for jumpscare detection, benchmarks, and pre-scan analysis."""

import argparse
import json
from pathlib import Path
import sys
from typing import List

import numpy as np

from jumpscare_detector.benchmark import (
    evaluate_against_timestamps,
    format_benchmark_markdown,
    generate_ambient_drone,
    generate_scream_shock,
    run_comprehensive_benchmark,
    run_gain_stress_test,
)
from jumpscare_detector.detector import JumpscareDetector
from jumpscare_detector.media import save_audio_wav


def parse_timestamp_string(text: str) -> float:
    """Parse timestamp in seconds or MM:SS or HH:MM:SS format."""
    text = text.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
    return float(text)


def load_ground_truth_timestamps(filepath: Path) -> List[float]:
    """Load ground truth timestamps from plain text, CSV, or JSON file."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    timestamps: List[float] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Support JSON list
        if line.startswith("[") and line.endswith("]"):
            try:
                data = json.loads(line)
                return [float(x) for x in data]
            except Exception:
                pass
        # Support CSV or MM:SS lines
        parts = line.split(",")
        try:
            ts = parse_timestamp_string(parts[0])
            timestamps.append(ts)
        except ValueError:
            continue

    return sorted(timestamps)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run comprehensive anomaly detection benchmark suite."""
    print("Running physics-based acoustic anomaly detection benchmark...")
    print("Evaluating against horror jumpscares, musical crescendos, sustained action, and speech...")
    results = run_comprehensive_benchmark(sample_rate=args.sample_rate)
    md_output = format_benchmark_markdown(results)
    print("\n" + md_output + "\n")

    if args.stress_test:
        print("Running gain invariance stress test (-12 dB, -6 dB, 0 dB, +6 dB)...")
        stress_records = run_gain_stress_test(sample_rate=args.sample_rate)
        print("\n| Gain Offset | Scale | Precision | Recall | F1 Score | False Positives |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for rec in stress_records:
            print(
                f"| {rec['gain_db']:+5.1f} dB | {rec['scale_factor']:.3f} | "
                f"{rec['precision']*100:.1f}% | {rec['recall']*100:.1f}% | "
                f"{rec['f1_score']:.3f} | {int(rec['false_positives'])} |"
            )
        print("")

    if args.output:
        Path(args.output).write_text(md_output, encoding="utf-8")
        print(f"Benchmark results saved to: {args.output}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Pre-scan a media file for acoustic jumpscares and generate alerts."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Pre-scanning media file: {input_path}")
    detector = JumpscareDetector(
        sample_rate=args.sample_rate,
        shock_threshold=args.threshold,
        min_contrast_db=args.min_contrast,
    )

    events, timeline = detector.analyze_file(input_path, cooldown_sec=args.cooldown)
    duration_sec = float(timeline.timestamps[-1]) if len(timeline.timestamps) > 0 else 0.0
    summary = detector.compute_summary(events, duration_sec)

    print(f"\nPre-Scan Analysis Complete:")
    print(f"  Duration: {summary.duration_sec:.1f}s ({summary.duration_sec / 60.0:.1f} min)")
    print(f"  Total Detected Events: {summary.total_events}")
    print(f"  Event Frequency: {summary.events_per_hour:.1f} events/hour")
    print(f"  Severity Breakdown: {summary.extreme_severity_count} Extreme, {summary.high_severity_count} High, {summary.moderate_severity_count} Moderate")
    print(f"  Peak Contrast: +{summary.max_contrast_db:.1f} dB | Peak Jerk: {summary.max_jerk_value:.1f}\n")

    for idx, ev in enumerate(events, 1):
        warn_ts = max(0.0, ev.timestamp - args.lead_time)
        print(
            f"[{idx}] Impact at {ev.timestamp:.2f}s (Warning at {warn_ts:.2f}s) | "
            f"Duration: {ev.duration:.2f}s | Severity: {ev.severity} | "
            f"Score: {ev.peak_score:.2f} | Contrast: +{ev.dynamic_ratio_db:.1f} dB | "
            f"Jerk: {ev.jerk_value:.1f} | Centroid: {ev.spectral_centroid_hz:.0f} Hz"
        )

    if args.export_json:
        detector.export_json(
            events,
            args.export_json,
            duration_sec=duration_sec,
            warning_lead_time_sec=args.lead_time,
        )
        print(f"\nEvents exported to JSON: {args.export_json}")

    if args.export_srt:
        detector.export_srt(
            events,
            args.export_srt,
            warning_lead_time_sec=args.lead_time,
        )
        print(f"Warning subtitles exported to SRT: {args.export_srt}")


def cmd_evaluate_timestamps(args: argparse.Namespace) -> None:
    """Evaluate detector performance against an external ground truth timestamp file."""
    input_path = Path(args.input)
    truth_path = Path(args.timestamps)

    if not input_path.exists():
        print(f"Error: Media file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not truth_path.exists():
        print(f"Error: Timestamps file does not exist: {truth_path}", file=sys.stderr)
        sys.exit(1)

    ground_truth = load_ground_truth_timestamps(truth_path)
    print(f"Loaded {len(ground_truth)} ground truth events from {truth_path}")

    detector = JumpscareDetector(
        sample_rate=args.sample_rate,
        shock_threshold=args.threshold,
        min_contrast_db=args.min_contrast,
    )
    events, timeline = detector.analyze_file(input_path)
    detected_times = [e.timestamp for e in events]
    total_duration = float(timeline.timestamps[-1]) if len(timeline.timestamps) > 0 else 0.0

    eval_result = evaluate_against_timestamps(
        detected_timestamps=detected_times,
        ground_truth_timestamps=ground_truth,
        total_duration_sec=total_duration,
        tolerance_sec=args.tolerance,
    )

    print(f"\nEvaluation Results against {truth_path.name}:")
    print(f"  Ground Truth Events: {len(ground_truth)}")
    print(f"  Detected Events:     {len(detected_times)}")
    print(f"  True Positives:      {eval_result['true_positives']}")
    print(f"  False Positives:     {eval_result['false_positives']}")
    print(f"  False Negatives:     {eval_result['false_negatives']}")
    print(f"  Precision:           {eval_result['precision']*100:.1f}%")
    print(f"  Recall:              {eval_result['recall']*100:.1f}%")
    print(f"  F1 Score:            {eval_result['f1_score']:.3f}")
    print(f"  False Alarms / Hour: {eval_result['false_alarms_per_hour']:.1f}")
    print(f"  Mean Latency Error:  {eval_result['mean_latency_ms']:.1f} ms\n")


def cmd_generate_sample(args: argparse.Namespace) -> None:
    """Generate a synthetic horror sample WAV for testing."""
    sr = args.sample_rate
    amb1 = generate_ambient_drone(3.5, sr, amplitude=0.015)
    shock = generate_scream_shock(1.2, sr, amplitude=0.90)
    amb2 = generate_ambient_drone(3.0, sr, amplitude=0.02)
    sample_audio = np.concatenate([amb1, shock, amb2])

    save_audio_wav(args.output, sample_audio, sample_rate=sr)
    print(f"Generated test sample saved to: {args.output} (Duration: {len(sample_audio) / sr:.1f}s, Shock at 3.5s)")


def main() -> None:
    """Entry point for jumpscare-detector CLI."""
    parser = argparse.ArgumentParser(
        prog="jumpscare-detector",
        description="Physics-based acoustic jumpscare detector and benchmark tool.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Benchmark subcommand
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark suite.")
    p_bench.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate (default: 16000).")
    p_bench.add_argument("--stress-test", action="store_true", help="Run gain variation stress test.")
    p_bench.add_argument("--output", type=str, default=None, help="Optional output Markdown file.")
    p_bench.set_defaults(func=cmd_benchmark)

    # Analyze subcommand
    p_analyze = subparsers.add_parser("analyze", help="Pre-scan media file for jumpscares.")
    p_analyze.add_argument("input", type=str, help="Path to video or audio file.")
    p_analyze.add_argument("--threshold", type=float, default=0.55, help="Shock score threshold (default: 0.55).")
    p_analyze.add_argument("--min-contrast", type=float, default=15.0, help="Minimum dB jump over baseline (default: 15.0).")
    p_analyze.add_argument("--cooldown", type=float, default=0.8, help="Cooldown between events in seconds (default: 0.8).")
    p_analyze.add_argument("--lead-time", type=float, default=2.0, help="Warning lead time in seconds (default: 2.0).")
    p_analyze.add_argument("--sample-rate", type=int, default=16000, help="Sample rate for analysis (default: 16000).")
    p_analyze.add_argument("--export-json", type=str, default=None, help="Export events to JSON file.")
    p_analyze.add_argument("--export-srt", type=str, default=None, help="Export warning track to SRT file.")
    p_analyze.set_defaults(func=cmd_analyze)

    # Evaluate against timestamps subcommand
    p_eval = subparsers.add_parser("evaluate-timestamps", help="Evaluate detections against ground truth timestamps.")
    p_eval.add_argument("input", type=str, help="Path to media file.")
    p_eval.add_argument("timestamps", type=str, help="Path to ground truth timestamp file (txt, csv, json).")
    p_eval.add_argument("--threshold", type=float, default=0.55, help="Shock score threshold (default: 0.55).")
    p_eval.add_argument("--min-contrast", type=float, default=15.0, help="Minimum dB jump over baseline (default: 15.0).")
    p_eval.add_argument("--tolerance", type=float, default=0.75, help="Collar tolerance in seconds (default: 0.75).")
    p_eval.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (default: 16000).")
    p_eval.set_defaults(func=cmd_evaluate_timestamps)

    # Generate sample subcommand
    p_gen = subparsers.add_parser("generate-sample", help="Generate synthetic test WAV.")
    p_gen.add_argument("output", type=str, help="Path for output WAV file.")
    p_gen.add_argument("--sample-rate", type=int, default=16000, help="Sample rate (default: 16000).")
    p_gen.set_defaults(func=cmd_generate_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
