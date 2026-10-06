"""Unit tests for Phase 4 multimodal pipeline and reporting."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from jumpscare_detector.media import save_audio_wav
from jumpscare_detector.multimodal import MultimodalConfig
from jumpscare_detector.pipeline import run_multimodal_pipeline


@pytest.fixture
def sample_wav_file():
    """Create a temporary WAV file containing an ambient drone and an explosive shock."""
    sr = 16000
    # 2 seconds drone + 0.1s shock + 2 seconds drone
    drone1 = np.random.normal(0, 0.01, int(sr * 2.0))
    shock = np.sin(2 * np.pi * 800 * np.linspace(0, 0.2, int(sr * 0.2))) * 0.9
    drone2 = np.random.normal(0, 0.01, int(sr * 2.0))
    audio = np.concatenate([drone1, shock, drone2])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        wav_path = Path(fp.name)

    save_audio_wav(wav_path, audio, sample_rate=sr)
    yield wav_path

    if wav_path.exists():
        wav_path.unlink()


def test_multimodal_pipeline_execution_and_exports(sample_wav_file):
    """Pipeline executes end-to-end, exports signal audit CSV, candidates CSV, and plot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)

        config = MultimodalConfig(detection_threshold=1.5)
        res = run_multimodal_pipeline(
            media_path=sample_wav_file,
            output_dir=out_dir,
            multimodal_config=config,
            generate_plot=True,
        )

        assert res["candidate_count"] >= 1
        assert Path(res["audit_csv"]).exists()
        assert Path(res["candidates_csv"]).exists()
        assert Path(res["diagnostic_plot"]).exists()

        # Verify CSV contents
        df_audit = pd.read_csv(res["audit_csv"])
        assert "timestamp" in df_audit.columns
        assert "fused_score" in df_audit.columns
        assert len(df_audit) > 50

        df_cand = pd.read_csv(res["candidates_csv"])
        assert len(df_cand) >= 1
        assert "timestamp" in df_cand.columns
        assert "score" in df_cand.columns
        # Candidate timestamp should be near 2.0s
        assert df_cand.loc[0, "timestamp"] == pytest.approx(2.0, abs=0.25)
