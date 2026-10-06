# Jumpscare Detector

A tool designed to analyze video and audio streams for sudden jumpscare cues, such as rapid amplitude spikes and abrupt visual frame differences.

## Overview

Jumpscares in digital media typically combine two distinct sensory signals:
1. **Audio cues:** Sudden spikes in loudness (RMS energy) following quiet passages, sharp onset transients, or high-frequency bursts.
2. **Visual cues:** Abrupt shifts in frame luminance, high inter-frame motion deltas, or sudden contrast changes.

This project provides an automated detection system to identify these timestamped events and generate warnings or reports.

## Project Structure

```text
jumpscare-detector/
├── src/
│   └── jumpscare_detector/
│       ├── __init__.py
│       ├── audio.py        # Audio signal processing and transient detection
│       ├── video.py        # Visual frame delta and motion analysis
│       └── detector.py     # Multi-modal fusion and scoring pipeline
├── tests/                  # Unit and integration tests
├── .gitignore
├── LICENSE
└── README.md
```

## Detection Pipeline (Planned)

- **Audio Analysis:**
  - Short-Time Fourier Transform (STFT) and RMS loudness envelope extraction.
  - Dynamic range delta calculation (comparing transient peak volume against preceding rolling averages).
  - High-frequency transient detection.

- **Video Analysis:**
  - Frame-to-frame pixel intensity difference (luminance delta).
  - Structural similarity (SSIM) drops.
  - Optical flow or motion vector variance.

- **Scoring & Reporting:**
  - Combined threshold scoring for audio-visual alignment.
  - Exporting timestamps and severity ratings in JSON, CSV, or SRT subtitle format.

## Setup & Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg (for media demuxing and extraction)

### Installation

Clone the repository:

```bash
git clone https://github.com/KindlyGentleman/jumpscare-detector.git
cd jumpscare-detector
```

Set up a virtual environment:

```bash
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

*(Dependency list and implementation modules will be added as development begins.)*

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
