"""
Heart-rate variability (HRV) and rhythm classification.
Time-domain HRV per ESC/NASPE 1996 guidelines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Rhythm = Literal["normal", "bradycardia", "tachycardia", "afib"]


@dataclass
class HRVMetrics:
    n_beats: int
    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    sdnn_ms: float
    rmssd_ms: float
    pnn50_pct: float
    cv: float


def rr_intervals_s(r_peaks: np.ndarray, fs_hz: float) -> np.ndarray:
    if len(r_peaks) < 2:
        return np.array([])
    return np.diff(r_peaks) / fs_hz


def hrv_metrics(r_peaks: np.ndarray, fs_hz: float) -> HRVMetrics:
    rr = rr_intervals_s(r_peaks, fs_hz)
    if len(rr) < 2:
        nan = float("nan")
        return HRVMetrics(len(r_peaks), nan, nan, nan, nan, nan, nan, nan)
    rr_ms = rr * 1000.0
    hr = 60.0 / rr
    diff = np.diff(rr_ms)
    return HRVMetrics(
        n_beats=int(len(r_peaks)),
        mean_hr_bpm=float(np.mean(hr)),
        min_hr_bpm=float(np.min(hr)),
        max_hr_bpm=float(np.max(hr)),
        sdnn_ms=float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
        rmssd_ms=float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0,
        pnn50_pct=float(100.0 * np.mean(np.abs(diff) > 50.0)) if len(diff) else 0.0,
        cv=float(np.std(rr_ms, ddof=1) / np.mean(rr_ms)) if np.mean(rr_ms) else 0.0,
    )


def classify_rhythm(metrics: HRVMetrics) -> Rhythm:
    if np.isnan(metrics.mean_hr_bpm):
        return "normal"
    if metrics.n_beats >= 6 and (metrics.cv > 0.20 or metrics.pnn50_pct > 60):
        return "afib"
    if metrics.mean_hr_bpm < 60:
        return "bradycardia"
    if metrics.mean_hr_bpm > 100:
        return "tachycardia"
    return "normal"


def rhythm_label(rhythm: Rhythm) -> str:
    return {
        "normal": "Normal Sinus Rhythm",
        "bradycardia": "Bradycardia (HR < 60)",
        "tachycardia": "Tachycardia (HR > 100)",
        "afib": "Atrial Fibrillation",
    }[rhythm]


def rhythm_color(rhythm: Rhythm) -> str:
    return {
        "normal": "#16a34a",
        "bradycardia": "#f59e0b",
        "tachycardia": "#f59e0b",
        "afib": "#dc2626",
    }[rhythm]
