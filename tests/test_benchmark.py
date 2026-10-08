"""Tests for benchmark harness, stress tests, and timestamp evaluation."""

from pathlib import Path
from jumpscare_detector.benchmark import (
    evaluate_against_timestamps,
    format_benchmark_markdown,
    run_comprehensive_benchmark,
    run_gain_stress_test,
)
from jumpscare_detector.cli import load_ground_truth_timestamps, parse_timestamp_string


def test_benchmark_suite_execution() -> None:
    """Benchmark suite should execute and return valid results for all methods."""
    results = run_comprehensive_benchmark(sample_rate=16000)

    assert len(results) >= 4
    naive = results[0]
    physics = results[-1]

    # Physics engine should have superior precision and F1 score over naive threshold
    assert physics.precision > naive.precision
    assert physics.f1_score > naive.f1_score
    assert physics.false_positives == 0

    # Real-time factor should be well above 1.0x (e.g. >15x)
    assert physics.real_time_factor > 15.0

    # Markdown output should contain formatted table
    md = format_benchmark_markdown(results)
    assert "| Detection Method |" in md
    assert "Physics & DSP Engine" in md


def test_gain_stress_test_execution() -> None:
    """Gain stress test should execute across gain levels."""
    records = run_gain_stress_test(sample_rate=16000)
    assert len(records) == 4
    for r in records:
        assert "gain_db" in r
        assert "f1_score" in r
        assert r["f1_score"] >= 0.50


def test_evaluate_against_timestamps() -> None:
    """Timestamp evaluation helper should match detections to ground truth within tolerance."""
    detected = [10.05, 30.12, 55.40]
    ground_truth = [10.00, 30.00, 70.00]
    eval_res = evaluate_against_timestamps(
        detected_timestamps=detected,
        ground_truth_timestamps=ground_truth,
        total_duration_sec=100.0,
        tolerance_sec=0.50,
    )

    assert eval_res["true_positives"] == 2
    assert eval_res["false_positives"] == 1
    assert eval_res["false_negatives"] == 1
    assert eval_res["precision"] == 2.0 / 3.0
    assert eval_res["recall"] == 2.0 / 3.0


def test_timestamp_parser_and_loader(tmp_path: Path) -> None:
    """Verify parsing seconds, MM:SS, and loading from file."""
    assert parse_timestamp_string("45.2") == 45.2
    assert parse_timestamp_string("01:30") == 90.0
    assert parse_timestamp_string("01:02:03") == 3723.0

    test_file = tmp_path / "ground_truth.txt"
    test_file.write_text("# Ground truth events\n00:15\n45.5\n02:10,Jumpscare 3\n", encoding="utf-8")
    loaded = load_ground_truth_timestamps(test_file)
    assert loaded == [15.0, 45.5, 130.0]
