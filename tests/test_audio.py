"""Unit tests for audio physics anomaly detector and speech rejection."""

import json
from pathlib import Path
import numpy as np
import pytest

from jumpscare_detector.audio import (
    AudioPhysicsDetector,
    PhysicsConfig,
    StreamingAudioPhysicsDetector,
    compute_a_weighting,
)
from jumpscare_detector.detector import JumpscareDetector
from jumpscare_detector.benchmark import (
    generate_ambient_drone,
    generate_scream_shock,
)


def test_a_weighting_normalization() -> None:
    """Verify A-weighting yields unity gain near 1000 Hz."""
    freqs = np.array([100.0, 1000.0, 3000.0, 10000.0])
    weights = compute_a_weighting(freqs)

    # 1000 Hz should be approximately 1.0 (0 dB)
    assert np.isclose(weights[1], 1.0, atol=0.05)
    # Low frequency (100 Hz) should be strongly attenuated
    assert weights[0] < 0.3
    # Ear resonance range (3000 Hz) should have positive gain boost
    assert weights[2] > 1.0


def test_silence_handling() -> None:
    """Detector should handle absolute silence without error or false alarm."""
    sr = 16000
    silence = np.zeros(sr * 3, dtype=np.float64)
    detector = AudioPhysicsDetector(sample_rate=sr)

    feats = detector.extract_features(silence)
    assert len(feats.timestamps) > 0
    assert np.all(feats.rms_energy < 1e-4)
    assert np.all(feats.shock_score == 0.0)

    events = detector.detect_events(feats)
    assert len(events) == 0


def test_canonical_jumpscare_detection() -> None:
    """Explosive transient following quiet drone must be detected."""
    sr = 16000
    ambient = generate_ambient_drone(3.0, sr, amplitude=0.015)
    shock = generate_scream_shock(1.0, sr, amplitude=0.85)
    tail = generate_ambient_drone(2.0, sr, amplitude=0.02)
    audio = np.concatenate([ambient, shock, tail])

    detector = JumpscareDetector(sample_rate=sr)
    events, timeline = detector.analyze_audio(audio)

    assert len(events) == 1
    event = events[0]
    assert abs(event.timestamp - 3.0) < 0.25
    assert event.peak_score >= 0.50
    assert event.dynamic_ratio_db >= 20.0
    assert event.duration >= 0.35
    assert event.severity in ["High", "Extreme"]


def test_gain_invariance_detection() -> None:
    """Verify detection remains robust under -6 dB and +3 dB gain variations."""
    sr = 16000
    ambient = generate_ambient_drone(3.0, sr, amplitude=0.015)
    shock = generate_scream_shock(1.0, sr, amplitude=0.85)
    tail = generate_ambient_drone(2.0, sr, amplitude=0.02)
    base_audio = np.concatenate([ambient, shock, tail])

    detector = JumpscareDetector(sample_rate=sr)

    # -6 dB attenuation
    scaled_down = np.clip(base_audio * 0.50, -1.0, 1.0)
    evs_down, _ = detector.analyze_audio(scaled_down)
    assert len(evs_down) == 1
    assert abs(evs_down[0].timestamp - 3.0) < 0.25

    # +3 dB boost
    scaled_up = np.clip(base_audio * 1.41, -1.0, 1.0)
    evs_up, _ = detector.analyze_audio(scaled_up)
    assert len(evs_up) == 1
    assert abs(evs_up[0].timestamp - 3.0) < 0.25


def test_silence_to_speech_suppression() -> None:
    """Transition from near-silence to voiced speech must not trigger jumpscare."""
    sr = 16000
    silence = generate_ambient_drone(3.5, sr, amplitude=0.003)

    t_speech = np.linspace(0, 2.5, int(2.5 * sr), endpoint=False)
    f0 = 140.0
    speech = (
        0.50 * np.sin(2 * np.pi * f0 * t_speech)
        + 0.35 * np.sin(2 * np.pi * 2 * f0 * t_speech)
        + 0.25 * np.sin(2 * np.pi * 3 * f0 * t_speech)
        + 0.15 * np.sin(2 * np.pi * 5 * f0 * t_speech)
        + 0.10 * np.sin(2 * np.pi * 7 * f0 * t_speech)
    )
    syllables = np.abs(np.sin(2 * np.pi * 3.0 * t_speech)) ** 1.5
    speech_signal = speech * syllables * 0.22
    audio = np.concatenate([silence, speech_signal])

    detector = JumpscareDetector(sample_rate=sr)
    events, timeline = detector.analyze_audio(audio)

    assert len(events) == 0


def test_crescendo_suppression() -> None:
    """Gradual volume increase over several seconds must not trigger jumpscare shock."""
    sr = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    envelope = (t / duration) ** 3
    carrier = np.sin(2 * np.pi * 300 * t)
    audio = carrier * envelope * 0.90

    detector = JumpscareDetector(sample_rate=sr)
    events, _ = detector.analyze_audio(audio)

    assert len(events) == 0


def test_sustained_action_suppression() -> None:
    """Continuously loud noise with high baseline must not trigger jumpscare."""
    sr = 16000
    duration = 5.0
    audio = np.random.normal(0, 0.4, int(duration * sr))
    audio = np.clip(audio, -0.9, 0.9)

    detector = JumpscareDetector(sample_rate=sr)
    events, _ = detector.analyze_audio(audio)

    assert len(events) == 0


def test_prescan_summary_and_exports(tmp_path: Path) -> None:
    """Verify diagnostic pre-scan summary and subtitle/JSON exports."""
    sr = 16000
    amb = generate_ambient_drone(3.0, sr, amplitude=0.015)
    shock = generate_scream_shock(1.0, sr, amplitude=0.85)
    audio = np.concatenate([amb, shock])

    detector = JumpscareDetector(sample_rate=sr)
    events, timeline = detector.analyze_audio(audio)
    duration = float(timeline.timestamps[-1])

    summary = detector.compute_summary(events, duration)
    assert summary.total_events == 1
    assert summary.duration_sec > 3.5
    assert summary.events_per_hour > 0.0

    json_file = tmp_path / "test_prescan.json"
    detector.export_json(events, json_file, duration_sec=duration, warning_lead_time_sec=2.0)
    assert json_file.exists()

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert len(payload["events"]) == 1
    assert payload["events"][0]["warning_lead_time_sec"] == 2.0
    assert payload["events"][0]["warning_timestamp"] <= payload["events"][0]["timestamp"]

    srt_file = tmp_path / "test_warnings.srt"
    detector.export_srt(events, srt_file, warning_lead_time_sec=2.0)
    assert srt_file.exists()
    srt_text = srt_file.read_text(encoding="utf-8")
    assert "JUMPSCARE WARNING" in srt_text
    assert "Impact in 2.0s" in srt_text


def test_streaming_detector_matches_offline() -> None:
    """Streaming chunk processor should flag alert during jumpscare."""
    sr = 16000
    ambient = generate_ambient_drone(2.5, sr, amplitude=0.015)
    shock = generate_scream_shock(1.0, sr, amplitude=0.85)
    tail = generate_ambient_drone(1.5, sr, amplitude=0.02)
    audio = np.concatenate([ambient, shock, tail])

    streaming = StreamingAudioPhysicsDetector(sample_rate=sr)
    chunk_size = 800

    alerts: list[float] = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        results = streaming.process_chunk(chunk)
        for ts, score, is_alert in results:
            if is_alert:
                alerts.append(ts)

    assert len(alerts) > 0
    first_alert = alerts[0]
    assert abs(first_alert - 2.5) < 0.25
