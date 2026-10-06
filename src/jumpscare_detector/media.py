"""Media loading and audio extraction utilities using FFmpeg and Scipy."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Optional, Tuple, Union
import numpy as np
from scipy.io import wavfile


def load_audio_wav(filepath: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load standard WAV file and resample to target rate if needed."""
    sr, data = wavfile.read(str(filepath))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize integer PCM to [-1.0, 1.0]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    else:
        data = data.astype(np.float64)

    if sr != target_sr:
        num_samples = int(len(data) * float(target_sr) / sr)
        # Linear interpolation for fast resampling
        orig_indices = np.linspace(0, len(data) - 1, len(data))
        new_indices = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(new_indices, orig_indices, data)
        sr = target_sr

    return data, sr


def extract_audio_ffmpeg(
    media_path: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Extract mono PCM float32 audio from any video or audio file using FFmpeg."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg executable not found in system PATH.")

    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "-",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    raw_bytes, stderr = proc.communicate()

    if proc.returncode != 0:
        err_msg = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg audio extraction failed: {err_msg}")

    audio_array = np.frombuffer(raw_bytes, dtype=np.float32).astype(np.float64)
    return audio_array, target_sr


def load_audio(
    source: Union[str, Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load audio from WAV directly or fallback to FFmpeg for other media containers."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Media file does not exist: {path}")

    if path.suffix.lower() == ".wav":
        try:
            return load_audio_wav(path, target_sr=target_sr)
        except Exception:
            # Corrupted header or unusual format fallback to ffmpeg
            return extract_audio_ffmpeg(path, target_sr=target_sr)

    return extract_audio_ffmpeg(path, target_sr=target_sr)


def save_audio_wav(
    filepath: Union[str, Path],
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> None:
    """Save float audio array as standard 16-bit PCM WAV."""
    norm_audio = np.clip(audio, -1.0, 1.0)
    int16_audio = (norm_audio * 32767.0).astype(np.int16)
    wavfile.write(str(filepath), sample_rate, int16_audio)
