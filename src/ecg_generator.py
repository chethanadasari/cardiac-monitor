"""
Synthetic ECG signal generator.

Produces realistic single-lead ECG waveforms by summing Gaussian-shaped
P, Q, R, S and T components at physiologically reasonable amplitudes and
durations. Supports four rhythm types and three noise sources commonly
seen in clinical signals.

References
----------
- McSharry et al., "A dynamical model for generating synthetic ECG
  signals", IEEE Trans. Biomedical Engineering, 50(3):289-294, 2003.
- AAMI EC57: Testing and reporting performance results of cardiac
  rhythm and ST-segment measurement algorithms.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Rhythm = Literal["normal", "bradycardia", "tachycardia", "afib"]


@dataclass(frozen=True)
class _Wave:
    name: str
    amp_mV: float
    mean_s: float
    sigma_s: float


_PQRST = (
    _Wave("P",  0.15, -0.20, 0.025),
    _Wave("Q", -0.10, -0.025, 0.012),
    _Wave("R",  1.20,  0.000, 0.012),
    _Wave("S", -0.25,  0.025, 0.012),
    _Wave("T",  0.30,  0.30, 0.040),
)


@dataclass
class ECGConfig:
    fs_hz: int = 250
    duration_s: float = 10.0
    rhythm: Rhythm = "normal"
    base_hr_bpm: float = 75.0
    awgn_db: float | None = None
    baseline_wander_amp: float = 0.0
    powerline_amp: float = 0.0
    powerline_hz: float = 50.0
    seed: int | None = None


def _rr_intervals(cfg: ECGConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.rhythm == "normal":
        mean = 60.0 / cfg.base_hr_bpm
        n_max = int(np.ceil(cfg.duration_s / mean * 1.5))
        rr = rng.normal(mean, 0.05 * mean, size=n_max)
    elif cfg.rhythm == "bradycardia":
        mean = 60.0 / 45.0
        n_max = int(np.ceil(cfg.duration_s / mean * 1.5))
        rr = rng.normal(mean, 0.04 * mean, size=n_max)
    elif cfg.rhythm == "tachycardia":
        mean = 60.0 / 130.0
        n_max = int(np.ceil(cfg.duration_s / mean * 1.5))
        rr = rng.normal(mean, 0.03 * mean, size=n_max)
    elif cfg.rhythm == "afib":
        mean_hr = cfg.base_hr_bpm
        mean = 60.0 / mean_hr
        n_max = int(np.ceil(cfg.duration_s / mean * 2.5))
        rr = rng.lognormal(mean=np.log(mean), sigma=0.32, size=n_max)
    else:
        raise ValueError(f"Unknown rhythm: {cfg.rhythm}")
    return np.clip(rr, 0.25, 2.0)


def _beat_template(fs_hz: int, hr_bpm: float):
    pre_s, post_s = 0.4, 0.6
    n_pre = int(round(pre_s * fs_hz))
    n_post = int(round(post_s * fs_hz))
    t = np.arange(-n_pre, n_post) / fs_hz
    qt_scale = np.sqrt(60.0 / max(hr_bpm, 30.0))
    out = np.zeros_like(t)
    for w in _PQRST:
        sigma = w.sigma_s * (qt_scale if w.name == "T" else 1.0)
        mean = w.mean_s * (qt_scale if w.name == "T" else 1.0)
        out += w.amp_mV * np.exp(-((t - mean) ** 2) / (2 * sigma ** 2))
    return out, n_pre


def generate_ecg(cfg: ECGConfig):
    rng = np.random.default_rng(cfg.seed)
    n = int(round(cfg.duration_s * cfg.fs_hz))
    t = np.arange(n) / cfg.fs_hz
    ecg = np.zeros(n)
    rr = _rr_intervals(cfg, rng)
    r_times = np.cumsum(rr)
    r_times = r_times[r_times < cfg.duration_s - 0.5]
    r_indices = []
    for r_t in r_times:
        idx = np.searchsorted(np.cumsum(rr), r_t)
        if idx == 0:
            inst_hr = 60.0 / rr[0]
        else:
            inst_hr = 60.0 / rr[idx - 1]
        beat, r_off = _beat_template(cfg.fs_hz, inst_hr)
        r_idx_global = int(round(r_t * cfg.fs_hz))
        start = r_idx_global - r_off
        end = start + len(beat)
        b_start = max(0, -start)
        b_end = len(beat) - max(0, end - n)
        s_start = max(0, start)
        s_end = min(n, end)
        ecg[s_start:s_end] += beat[b_start:b_end]
        if 0 <= r_idx_global < n:
            r_indices.append(r_idx_global)
    if cfg.baseline_wander_amp > 0:
        f_bw = rng.uniform(0.15, 0.4)
        ecg += cfg.baseline_wander_amp * np.sin(2 * np.pi * f_bw * t + rng.uniform(0, 2 * np.pi))
    if cfg.powerline_amp > 0:
        ecg += cfg.powerline_amp * np.sin(2 * np.pi * cfg.powerline_hz * t)
    if cfg.awgn_db is not None:
        sig_p = np.mean(ecg ** 2)
        snr_lin = 10 ** (cfg.awgn_db / 10)
        noise_p = sig_p / snr_lin
        ecg += rng.normal(0, np.sqrt(noise_p), size=n)
    return t, ecg, np.array(r_indices, dtype=int)


def heart_rate_bpm(r_peaks: np.ndarray, fs_hz: int) -> float:
    if len(r_peaks) < 2:
        return float("nan")
    rr_s = np.diff(r_peaks) / fs_hz
    return float(60.0 / np.mean(rr_s))
