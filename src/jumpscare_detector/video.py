"""Computer vision transient detection using frame differences and Farneback optical flow."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoConfig:
    """Configuration parameters for visual transient analysis."""

    scale_factor: float = 0.5
    target_fps: Optional[float] = None
    diff_threshold: float = 20.0
    flow_threshold: float = 2.0
    hist_bins: int = 32
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2


@dataclass(frozen=True)
class VideoFrameMetrics:
    """Computed transient metrics for a single video frame."""

    timestamp: float
    frame_index: int
    mean_frame_diff: float
    p95_frame_diff: float
    diff_area_ratio: float
    luminance_step: float
    hist_distance: float
    flow_mean: float
    flow_p95: float
    flow_area_ratio: float
    flow_divergence: float
    radial_expansion: float
    time_to_contact_tau: float
    flow_residual_mean: float
    flow_residual_p95: float
    phase_motion_score: float


def compute_frame_difference_features(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    diff_threshold: float = 20.0,
) -> Tuple[float, float, float]:
    """Compute mean difference, spatial 95th percentile, and active difference area ratio."""
    diff = cv2.absdiff(curr_gray, prev_gray)
    mean_diff = float(np.mean(diff))
    p95_diff = float(np.percentile(diff, 95))
    diff_area = float(np.mean(diff > diff_threshold))
    return mean_diff, p95_diff, diff_area


def compute_luminance_step(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute absolute change in mean frame luminance."""
    curr_y = float(np.mean(curr_gray))
    prev_y = float(np.mean(prev_gray))
    return abs(curr_y - prev_y)


def compute_histogram_distance(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    bins: int = 32,
) -> float:
    """Compute histogram intersection distance across normalized grayscale histograms."""
    hist_prev = cv2.calcHist([prev_gray], [0], None, [bins], [0, 256]).flatten()
    hist_curr = cv2.calcHist([curr_gray], [0], None, [bins], [0, 256]).flatten()

    norm_prev = hist_prev / (np.sum(hist_prev) + 1e-9)
    norm_curr = hist_curr / (np.sum(hist_curr) + 1e-9)

    intersection = float(np.sum(np.minimum(norm_prev, norm_curr)))
    return max(0.0, min(1.0, 1.0 - intersection))


def compute_dense_optical_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    config: VideoConfig,
) -> np.ndarray:
    """Estimate dense Farneback optical flow between two consecutive grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=config.farneback_pyr_scale,
        levels=config.farneback_levels,
        winsize=config.farneback_winsize,
        iterations=config.farneback_iterations,
        poly_n=config.farneback_poly_n,
        poly_sigma=config.farneback_poly_sigma,
        flags=0,
    )
    return flow


def compute_flow_divergence(flow: np.ndarray) -> float:
    """Compute spatial divergence du/dx + dv/dy using Sobel spatial derivatives."""
    u = flow[..., 0]
    v = flow[..., 1]

    # 3x3 Sobel filter has smoothing factor 8 along the orthogonal axis
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3) / 8.0

    divergence = du_dx + dv_dy
    return float(np.mean(divergence))


def compute_time_to_contact_tau(
    flow: np.ndarray,
    flow_mag: np.ndarray,
    flow_threshold: float = 2.0,
) -> float:
    """Compute Time-To-Contact (Tau) using Ecological Optics theory.
    
    Tau is estimated inversely from the local spatial divergence of the optical flow.
    A smaller Tau indicates an imminent biological startle trigger (looming object).
    Returns Tau in frames. 99.0 represents infinite/safe distance.
    """
    u = flow[..., 0]
    v = flow[..., 1]
    
    du_dx = cv2.Sobel(u, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    dv_dy = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    divergence = du_dx + dv_dy
    
    # We only care about areas where divergence is positive (expanding)
    motion_mask = (flow_mag > flow_threshold) & (divergence > 0.05)
    
    if not np.any(motion_mask):
        return 99.0
        
    # Tau = 2 / divergence (for 2D isotropic expansion)
    tau_field = 2.0 / divergence[motion_mask]
    
    # The 5th percentile represents the most threatening (fastest approaching) large object
    return float(np.percentile(tau_field, 5))


def compute_radial_expansion(
    flow: np.ndarray,
    flow_mag: np.ndarray,
    flow_threshold: float = 2.0,
) -> float:
    """Project motion vectors onto unit radial vectors centered at the frame midpoint."""
    h, w = flow.shape[:2]
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    y_coords, x_coords = np.indices((h, w), dtype=np.float64)
    dx = x_coords - cx
    dy = y_coords - cy
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6

    rx = dx / dist
    ry = dy / dist

    radial_proj = flow[..., 0] * rx + flow[..., 1] * ry
    motion_mask = flow_mag > flow_threshold

    if not np.any(motion_mask):
        return 0.0

    return float(np.median(radial_proj[motion_mask]))


def compute_residual_flow(flow: np.ndarray) -> Tuple[float, float]:
    """Compute residual motion after subtracting global camera panning vector."""
    u = flow[..., 0]
    v = flow[..., 1]

    u_median = np.median(u)
    v_median = np.median(v)

    u_res = u - u_median
    v_res = v - v_median

    res_mag = np.sqrt(u_res * u_res + v_res * v_res)
    mean_res = float(np.mean(res_mag))
    p95_res = float(np.percentile(res_mag, 95))
    return mean_res, p95_res


_PHASE_FILTERS_CACHE = {}

def compute_phase_based_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute illumination-invariant structural motion using Riesz transform phase difference.
    
    This function isolates true structural displacement (motion) from 
    pure illumination changes (flickering lights, strobe) which corrupt standard optical flow.
    """
    target_size = (128, 128)
    
    if target_size not in _PHASE_FILTERS_CACHE:
        rows, cols = target_size
        u = np.fft.fftfreq(cols)
        v = np.fft.fftfreq(rows)
        U, V = np.meshgrid(u, v)
        
        radius = np.sqrt(U**2 + V**2)
        radius[0, 0] = 1e-9
        
        f0 = 0.1
        sigma = 0.5
        bandpass = np.exp(- (np.log(radius / f0)**2) / (2 * np.log(sigma)**2))
        bandpass[0, 0] = 0
        
        R_u = 1j * U / radius
        R_v = 1j * V / radius
        
        _PHASE_FILTERS_CACHE[target_size] = (R_u, R_v, bandpass)
        
    R_u, R_v, bandpass = _PHASE_FILTERS_CACHE[target_size]
    
    prev_small = cv2.resize(prev_gray, target_size, interpolation=cv2.INTER_AREA)
    curr_small = cv2.resize(curr_gray, target_size, interpolation=cv2.INTER_AREA)

    def compute_local_phase(img):
        F = np.fft.fft2(img.astype(np.float32))
        F_bp = F * bandpass
        
        r_x = np.fft.ifft2(F_bp * R_u).real
        r_y = np.fft.ifft2(F_bp * R_v).real
        r_mag = np.sqrt(r_x**2 + r_y**2)
        
        i_bp = np.fft.ifft2(F_bp).real
        
        return np.arctan2(r_mag, i_bp)

    phase_prev = compute_local_phase(prev_small)
    phase_curr = compute_local_phase(curr_small)
    
    phase_diff = np.angle(np.exp(1j * (phase_curr - phase_prev)))
    
    return float(np.percentile(np.abs(phase_diff), 95))


def process_frame_pair(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    frame_index: int,
    timestamp: float,
    config: VideoConfig,
) -> VideoFrameMetrics:
    """Calculate all visual transient features for a pair of consecutive frames."""
    mean_diff, p95_diff, diff_area = compute_frame_difference_features(
        prev_gray, curr_gray, diff_threshold=config.diff_threshold
    )
    lum_step = compute_luminance_step(prev_gray, curr_gray)
    hist_dist = compute_histogram_distance(prev_gray, curr_gray, bins=config.hist_bins)

    flow = compute_dense_optical_flow(prev_gray, curr_gray, config)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    flow_mean = float(np.mean(flow_mag))
    flow_p95 = float(np.percentile(flow_mag, 95))
    flow_area = float(np.mean(flow_mag > config.flow_threshold))

    flow_div = compute_flow_divergence(flow)
    radial_exp = compute_radial_expansion(flow, flow_mag, flow_threshold=config.flow_threshold)
    tau = compute_time_to_contact_tau(flow, flow_mag, flow_threshold=config.flow_threshold)
    flow_res_mean, flow_res_p95 = compute_residual_flow(flow)
    
    phase_motion = compute_phase_based_motion(prev_gray, curr_gray)

    return VideoFrameMetrics(
        timestamp=timestamp,
        frame_index=frame_index,
        mean_frame_diff=mean_diff,
        p95_frame_diff=p95_diff,
        diff_area_ratio=diff_area,
        luminance_step=lum_step,
        hist_distance=hist_dist,
        flow_mean=flow_mean,
        flow_p95=flow_p95,
        flow_area_ratio=flow_area,
        flow_divergence=flow_div,
        radial_expansion=radial_exp,
        time_to_contact_tau=tau,
        flow_residual_mean=flow_res_mean,
        flow_residual_p95=flow_res_p95,
        phase_motion_score=phase_motion,
    )


def analyze_video_frames(
    frames: List[np.ndarray],
    fps: float,
    config: VideoConfig = VideoConfig(),
) -> pd.DataFrame:
    """Extract visual transient features from a sequence of BGR or grayscale video frames."""
    if not frames:
        return pd.DataFrame()

    results: List[VideoFrameMetrics] = []

    # First frame has zero delta
    results.append(
        VideoFrameMetrics(
            timestamp=0.0,
            frame_index=0,
            mean_frame_diff=0.0,
            p95_frame_diff=0.0,
            diff_area_ratio=0.0,
            luminance_step=0.0,
            hist_distance=0.0,
            flow_mean=0.0,
            flow_p95=0.0,
            flow_area_ratio=0.0,
            flow_divergence=0.0,
            radial_expansion=0.0,
            time_to_contact_tau=99.0,
            flow_residual_mean=0.0,
            flow_residual_p95=0.0,
            phase_motion_score=0.0,
        )
    )

    def to_scaled_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        if config.scale_factor != 1.0:
            gray = cv2.resize(
                gray,
                (0, 0),
                fx=config.scale_factor,
                fy=config.scale_factor,
                interpolation=cv2.INTER_AREA,
            )
        return gray

    prev_gray = to_scaled_gray(frames[0])

    for idx in range(1, len(frames)):
        curr_gray = to_scaled_gray(frames[idx])
        timestamp = idx / fps
        metric = process_frame_pair(prev_gray, curr_gray, idx, timestamp, config)
        results.append(metric)
        prev_gray = curr_gray

    records = [asdict(m) for m in results]
    return pd.DataFrame.from_records(records)


def extract_video_transients(
    video_path: Union[str, Path],
    config: VideoConfig = VideoConfig(),
) -> Tuple[pd.DataFrame, float]:
    """Read video file and compute complete transient metrics across all frames."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 30.0

    frame_step = 1
    effective_fps = native_fps
    if config.target_fps is not None and config.target_fps < native_fps:
        frame_step = max(1, int(round(native_fps / config.target_fps)))
        effective_fps = native_fps / frame_step

    results: List[VideoFrameMetrics] = []
    prev_gray: Optional[np.ndarray] = None
    frame_counter = 0
    sampled_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_counter % frame_step != 0:
                frame_counter += 1
                continue

            if config.scale_factor != 1.0:
                scaled = cv2.resize(
                    frame,
                    (0, 0),
                    fx=config.scale_factor,
                    fy=config.scale_factor,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                scaled = frame

            curr_gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            timestamp = frame_counter / native_fps

            if prev_gray is None:
                results.append(
                    VideoFrameMetrics(
                        timestamp=timestamp,
                        frame_index=sampled_index,
                        mean_frame_diff=0.0,
                        p95_frame_diff=0.0,
                        diff_area_ratio=0.0,
                        luminance_step=0.0,
                        hist_distance=0.0,
                        flow_mean=0.0,
                        flow_p95=0.0,
                        flow_area_ratio=0.0,
                        flow_divergence=0.0,
                        radial_expansion=0.0,
                        time_to_contact_tau=99.0,
                        flow_residual_mean=0.0,
                        flow_residual_p95=0.0,
                        phase_motion_score=0.0,
                    )
                )
            else:
                metric = process_frame_pair(
                    prev_gray, curr_gray, sampled_index, timestamp, config
                )
                results.append(metric)

            prev_gray = curr_gray
            sampled_index += 1
            frame_counter += 1

    finally:
        cap.release()

    if not results:
        return pd.DataFrame(), effective_fps

    records = [asdict(m) for m in results]
    return pd.DataFrame.from_records(records), effective_fps
