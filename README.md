# Jumpscare Detector

An explainable, physics-based multimodal transient anomaly detector designed to identify jumpscares and sudden audiovisual shocks in media streams.

Instead of relying on heavy black-box neural networks, this system detects abrupt sensory transients relative to local dynamic baselines by modeling acoustic attack dynamics and optical flow divergence.

---

## Signal Model & Mathematical Formulation

The detector evaluates candidate events via an explainable multimodal transient score:

$$S(t) = w_a S_a(t) + w_v S_v(t) + w_c S_c(t) + w_s S_{\mathrm{sync}}(t)$$

Where:
- $S_a(t)$: Audio transient score.
- $S_v(t)$: Visual transient score.
- $S_c(t)$: Dynamic context/contrast score.
- $S_{\mathrm{sync}}(t)$: Audio–visual synchrony score.
- $w_a, w_v, w_c, w_s$: Fusion weights (default: $0.35, 0.35, 0.15, 0.15$).

A candidate jumpscare is declared when $S(t) > T$ and satisfies a minimum temporal separation constraint ($t - t_{\mathrm{prev}} \ge T_{\mathrm{min}}$, default $1.0\text{ s}$).

### 1. Audio Transient Mechanics

1. **Analytic Amplitude Envelope:** Computed via Hilbert Transform $\mathcal{H}\{x[n]\}$ to track instantaneous acoustic peaks:
   $$z[n] = x[n] + j \mathcal{H}\{x[n]\}, \quad a[n] = |z[n]|$$

2. **10%–90% Rise Time ($T_r$):** Quantifies transient attack sharpness over envelope segments:
   $$S_{\mathrm{rise}} = \frac{1}{1 + T_r / \tau_{\mathrm{rise}}}$$

3. **Dynamic Contrast Spike ($\Delta L_{\mathrm{RMS}}$):** Measures loudness level deltas ($+\text{dB}$) relative to an exponential moving average (EMA) ambient floor:
   $$\Delta L(t) = L_{\mathrm{RMS}}(t) - L_{\mathrm{baseline}}(t)$$

4. **Multi-Band Positive Spectral Flux:** Half-wave rectified frequency changes across Low (20–250 Hz), Mid (250–2,000 Hz), and High (2,000–10,000 Hz) bands:
   $$\text{Flux}(t) = \sum_{r} \max\left(0, M(t, r) - M(t-1, r)\right)$$

### 2. Visual Motion Kinematics

1. **Dense Optical Flow:** Gunnar Farnebäck algorithm extracts per-pixel velocity vectors $(u, v)$ across consecutive video frames.
2. **Camera-Pan Suppressed Radial Expansion ($E_{\mathrm{rad}}$):** Global translational camera movement is subtracted, and divergence from the frame center is calculated to detect sudden forward rushes toward the camera:
   $$\vec{v}_{\mathrm{corr}} = \vec{v} - \vec{v}_{\mathrm{mean}}, \quad E_{\mathrm{rad}} = \frac{1}{|\Omega|} \sum_{(x,y) \in \Omega} \vec{v}_{\mathrm{corr}}(x, y) \cdot \hat{r}(x, y)$$

### 3. Audio–Visual Synchrony & Peak Selection

- **Synchrony Kernel:** Evaluates Gaussian temporal coincidence between audio onset and visual expansion peaks ($\sigma = 200\text{ ms}$):
  $$S_{\mathrm{sync}}(t) = \exp\left(-\frac{\Delta t^2}{2\sigma^2}\right)$$
- **Greedy Non-Maximum Suppression (NMS):** Filters clustered transient detections within window $T_{\mathrm{min}}$ to select the true physical impact peak.

---

## Project Structure

```text
jumpscare-detector/
├── src/
│   └── jumpscare_detector/
│       ├── __init__.py           # Package exports
│       ├── audio.py              # Classical STFT and acoustic jerk engine
│       ├── audio_transient.py    # Hilbert envelope, rise-time, and spectral flux
│       ├── video.py              # Farneback optical flow and radial expansion
│       ├── multimodal.py         # AV synchrony, contrast scoring, and NMS
│       ├── pipeline.py           # Unified multimodal pipeline
│       ├── plotting.py           # Diagnostic figure generator
│       ├── media.py              # FFmpeg audio/video extraction utilities
│       ├── benchmark.py          # Synthetic DSP test harness
│       └── cli.py                # Command-line interface
├── tests/
│   ├── test_audio.py             # Audio DSP unit tests
│   ├── test_audio_transient.py   # Hilbert envelope & flux tests
│   ├── test_video.py             # Optical flow & radial expansion tests
│   ├── test_multimodal.py        # Fusion & NMS tests
│   ├── test_pipeline.py          # End-to-end pipeline tests
│   └── test_benchmark.py         # Synthetic benchmark tests
├── render_animated_showcase.py   # Synchronized video showcase generator
├── create_showcase_figure.py     # Publication figure generator
├── pyproject.toml                # Project packaging & dependencies
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- FFmpeg (installed and available on `PATH`)

### Setup

```bash
git clone https://github.com/KindlyGentleman/jumpscare-detector.git
cd jumpscare-detector

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## Usage

### 1. Multimodal Audio-Visual Analysis

Analyze a video file across both audio and visual channels to generate candidate timestamps and telemetry:

```bash
python -m jumpscare_detector.cli multimodal horror_video.mp4 --threshold 1.8 --output-dir output
```

Output exports include:
- `output/jumpscare_candidates.csv`: Timestamped shock events with individual sub-scores.
- `output/jumpscare_signal_audit.csv`: Frame-by-frame physical telemetry (RMS dB, envelope, flux, radial expansion, $S(t)$).
- `output/jumpscare_diagnostic_plot.png`: Aligned time-series diagnostic plots.

### 2. Audio-Only Pre-Scan & Subtitle Warning Export

Run high-speed audio pre-scanning to generate advance warning subtitles (SRT format) 2 seconds before each shock:

```bash
python -m jumpscare_detector.cli analyze horror_video.mp4 --lead-time 2.0 --export-srt warnings.srt --export-json results.json
```

### 3. DSP Benchmark Suite

Run the procedural benchmark harness evaluating transient detection across 12 synthetic acoustic scenarios (door slams, stingers, speech, action crescendos):

```bash
python -m jumpscare_detector.cli benchmark
```

### 4. Running Unit Tests

```bash
pytest -v
```

All 29 tests validate unit and integration properties across DSP, optical flow, and multimodal fusion.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
