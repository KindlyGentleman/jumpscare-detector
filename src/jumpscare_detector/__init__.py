"""Jumpscare Detector: Physics-based acoustic anomaly and shock detection."""

from jumpscare_detector.audio import (
    AudioPhysicsDetector,
    AudioPhysicsFeatures,
    AudioShockEvent,
    PhysicsConfig,
    StreamingAudioPhysicsDetector,
    compute_a_weighting,
)
from jumpscare_detector.detector import AudioTimeline, JumpscareDetector, PrescanSummary
from jumpscare_detector.media import extract_audio_ffmpeg, load_audio, save_audio_wav

__version__ = "0.2.0"

__all__ = [
    "AudioPhysicsDetector",
    "AudioPhysicsFeatures",
    "AudioShockEvent",
    "PhysicsConfig",
    "StreamingAudioPhysicsDetector",
    "compute_a_weighting",
    "JumpscareDetector",
    "AudioTimeline",
    "PrescanSummary",
    "load_audio",
    "save_audio_wav",
    "extract_audio_ffmpeg",
]
