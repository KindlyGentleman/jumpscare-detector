"""Audio-centric physics-based jumpscare detection engine with pre-scan warnings."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from jumpscare_detector.audio import (
    AudioPhysicsDetector,
    AudioPhysicsFeatures,
    AudioShockEvent,
    PhysicsConfig,
    StreamingAudioPhysicsDetector,
)
from jumpscare_detector.media import load_audio


@dataclass
class AudioTimeline:
    """Synchronized timeline with acoustic features and shock scores."""
    timestamps: np.ndarray
    shock_scores: np.ndarray
    rms_energy: np.ndarray
    dynamic_ratio_db: np.ndarray
    acoustic_jerk: np.ndarray
    spectral_centroid_hz: np.ndarray
    spectral_flux: np.ndarray
    spectral_flatness: np.ndarray
    pitch_periodicity: np.ndarray
    speech_confidence: np.ndarray
    crest_factor: np.ndarray


@dataclass
class PrescanSummary:
    """Diagnostic statistical overview of pre-scan media analysis."""
    duration_sec: float
    total_events: int
    events_per_hour: float
    extreme_severity_count: int
    high_severity_count: int
    moderate_severity_count: int
    max_contrast_db: float
    max_jerk_value: float
    average_score: float


class JumpscareDetector:
    """Physics-based audio anomaly detector for horror media.

    This detector operates without neural network inference by modeling:
    1. Sudden acoustic energy surge (RMS envelope).
    2. Acoustic attack rate and jerk (dE/dt).
    3. Ambient quiescence contrast ratio with lagged pre-shock baseline.
    4. Psychoacoustic spectral centroid shift (screams and stingers).
    5. Spectral flux and crest factor (impulsiveness).
    6. Speech rejection filter via pitch autocorrelation and Wiener entropy.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        shock_threshold: Optional[float] = None,
        min_contrast_db: Optional[float] = None,
        min_peak_rms: Optional[float] = None,
        baseline_window_sec: float = 2.5,
        use_a_weighting: bool = True,
        speech_suppression_weight: Optional[float] = None,
        normalize_headroom: bool = True,
        config: Optional[PhysicsConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = PhysicsConfig(
                sample_rate=sample_rate,
                shock_threshold=shock_threshold if shock_threshold is not None else 0.52,
                min_contrast_db=min_contrast_db if min_contrast_db is not None else 16.0,
                min_peak_rms=min_peak_rms if min_peak_rms is not None else 0.20,
                baseline_window_sec=baseline_window_sec,
                use_a_weighting=use_a_weighting,
                speech_suppression_weight=(
                    speech_suppression_weight if speech_suppression_weight is not None else 0.95
                ),
                normalize_headroom=normalize_headroom,
            )

        self.sample_rate = self.config.sample_rate
        self.detector = AudioPhysicsDetector(config=self.config)

    def analyze_audio(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        cooldown_sec: Optional[float] = None,
    ) -> Tuple[List[AudioShockEvent], AudioTimeline]:
        """Analyze mono audio array and return detected shock events with timeline."""
        if sample_rate is not None and sample_rate != self.detector.sample_rate:
            self.config.sample_rate = sample_rate
            self.detector = AudioPhysicsDetector(config=self.config)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Dynamic headroom normalization for attenuated or quiet media:
        # Aligns with streaming platform loudness normalization standards (e.g. YouTube / ITU-R BS.1770)
        # to ensure gain invariance across -12 dB to +6 dB mix variations.
        if self.config.normalize_headroom and len(audio) > 0:
            peak_val = float(np.max(np.abs(audio)))
            if 0.04 < peak_val < 0.85:
                audio = audio * (0.90 / peak_val)

        features: AudioPhysicsFeatures = self.detector.extract_features(audio)
        events: List[AudioShockEvent] = self.detector.detect_events(
            features, cooldown_sec=cooldown_sec
        )

        timeline = AudioTimeline(
            timestamps=features.timestamps,
            shock_scores=features.shock_score,
            rms_energy=features.rms_energy,
            dynamic_ratio_db=features.dynamic_ratio_db,
            acoustic_jerk=features.acoustic_jerk,
            spectral_centroid_hz=features.spectral_centroid,
            spectral_flux=features.spectral_flux,
            spectral_flatness=features.spectral_flatness,
            pitch_periodicity=features.pitch_periodicity,
            speech_confidence=features.speech_confidence,
            crest_factor=features.crest_factor,
        )

        return events, timeline

    def analyze_file(
        self,
        filepath: Union[str, Path],
        cooldown_sec: Optional[float] = None,
    ) -> Tuple[List[AudioShockEvent], AudioTimeline]:
        """Load audio from file and execute physics shock analysis."""
        audio, sr = load_audio(filepath, target_sr=self.sample_rate)
        return self.analyze_audio(audio, sample_rate=sr, cooldown_sec=cooldown_sec)

    def compute_summary(
        self,
        events: List[AudioShockEvent],
        duration_sec: float,
    ) -> PrescanSummary:
        """Compute statistical breakdown for pre-scan reporting."""
        total_ev = len(events)
        hours = max(1e-4, duration_sec / 3600.0)
        ev_per_hour = total_ev / hours

        extreme = sum(1 for e in events if e.severity == "Extreme")
        high = sum(1 for e in events if e.severity == "High")
        moderate = sum(1 for e in events if e.severity == "Moderate")

        max_contrast = max((e.dynamic_ratio_db for e in events), default=0.0)
        max_jerk = max((e.jerk_value for e in events), default=0.0)
        avg_score = float(np.mean([e.peak_score for e in events])) if events else 0.0

        return PrescanSummary(
            duration_sec=round(duration_sec, 2),
            total_events=total_ev,
            events_per_hour=round(ev_per_hour, 2),
            extreme_severity_count=extreme,
            high_severity_count=high,
            moderate_severity_count=moderate,
            max_contrast_db=round(max_contrast, 1),
            max_jerk_value=round(max_jerk, 1),
            average_score=round(avg_score, 2),
        )

    def export_json(
        self,
        events: List[AudioShockEvent],
        filepath: Union[str, Path],
        duration_sec: Optional[float] = None,
        warning_lead_time_sec: float = 2.0,
    ) -> None:
        """Export detected jumpscare events and diagnostic summary to JSON format."""
        event_dicts: List[Dict[str, Any]] = []
        for ev in events:
            d = asdict(ev)
            d["warning_timestamp"] = max(0.0, ev.timestamp - warning_lead_time_sec)
            d["warning_lead_time_sec"] = warning_lead_time_sec
            event_dicts.append(d)

        output_payload: Dict[str, Any] = {
            "events": event_dicts,
            "warning_lead_time_sec": warning_lead_time_sec,
        }

        if duration_sec is not None:
            summary = self.compute_summary(events, duration_sec)
            output_payload["summary"] = asdict(summary)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)

    def export_srt(
        self,
        events: List[AudioShockEvent],
        filepath: Union[str, Path],
        warning_lead_time_sec: float = 2.0,
    ) -> None:
        """Export warnings as SRT subtitle track with advance alert for media players."""
        def format_time(seconds: float) -> str:
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int(round((seconds - int(seconds)) * 1000))
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        with open(filepath, "w", encoding="utf-8") as f:
            for idx, ev in enumerate(events, 1):
                t_alert_start = max(0.0, ev.timestamp - warning_lead_time_sec)
                t_event_end = ev.timestamp + ev.duration

                f.write(f"{idx}\n")
                f.write(f"{format_time(t_alert_start)} --> {format_time(t_event_end)}\n")
                f.write(
                    f"JUMPSCARE WARNING ({ev.severity} Severity)\n"
                    f"Impact in {warning_lead_time_sec:.1f}s | Score: {ev.peak_score:.2f} | Contrast: +{ev.dynamic_ratio_db:.1f} dB\n\n"
                )
