"""
Pan-Tompkins QRS detection (Pan & Tompkins, IEEE TBME 1985).
"""
from __future__ import annotations

import numpy as np

from filters import apply_fir, design_bandpass


def _moving_window_integration(x: np.ndarray, win_samples: int) -> np.ndarray:
    if win_samples < 1:
        return x.copy()
    kernel = np.ones(win_samples) / win_samples
    return np.convolve(x, kernel, mode="same")


def _adaptive_thresholding(integrated: np.ndarray, fs_hz: float) -> np.ndarray:
    n = len(integrated)
    refractory = int(0.2 * fs_hz)
    search_back_window = int(1.66 * fs_hz)
    init_n = min(int(2 * fs_hz), n)
    spki = 0.5 * np.max(integrated[:init_n])
    npki = 0.5 * np.mean(integrated[:init_n])
    threshold_i1 = npki + 0.25 * (spki - npki)
    peaks = []
    last_peak = -refractory
    candidates = []
    for i in range(1, n - 1):
        if integrated[i] > integrated[i - 1] and integrated[i] >= integrated[i + 1]:
            candidates.append(i)
    for c in candidates:
        if c - last_peak < refractory:
            continue
        v = integrated[c]
        if v > threshold_i1:
            peaks.append(c)
            spki = 0.125 * v + 0.875 * spki
            last_peak = c
        else:
            npki = 0.125 * v + 0.875 * npki
        threshold_i1 = npki + 0.25 * (spki - npki)
        if peaks and c - peaks[-1] > search_back_window:
            threshold_i2 = 0.5 * threshold_i1
            window = integrated[peaks[-1] + refractory: c]
            if len(window):
                local_max = peaks[-1] + refractory + int(np.argmax(window))
                if integrated[local_max] > threshold_i2:
                    peaks.append(local_max)
                    last_peak = local_max
                    spki = 0.25 * integrated[local_max] + 0.75 * spki
    return np.array(sorted(set(peaks)), dtype=int)


def _refine_to_raw(raw: np.ndarray, peaks_int: np.ndarray, fs_hz: float) -> np.ndarray:
    half = int(0.10 * fs_hz)
    refined = []
    seen = set()
    for p in peaks_int:
        lo, hi = max(0, p - half), min(len(raw), p + half + 1)
        if hi > lo:
            r = lo + int(np.argmax(raw[lo:hi]))
            if r not in seen:
                refined.append(r)
                seen.add(r)
    return np.array(sorted(refined), dtype=int)


def detect_r_peaks(signal: np.ndarray, fs_hz: float) -> np.ndarray:
    if len(signal) < int(2 * fs_hz):
        return np.array([], dtype=int)
    bp_taps = design_bandpass(5.0, 15.0, fs_hz, num_taps=int(0.5 * fs_hz) | 1)
    bp = apply_fir(signal - np.mean(signal), bp_taps)
    deriv = np.gradient(bp) * fs_hz / 8.0
    sq = deriv ** 2
    win = max(1, int(0.150 * fs_hz))
    integ = _moving_window_integration(sq, win)
    peaks_int = _adaptive_thresholding(integ, fs_hz)
    return _refine_to_raw(signal, peaks_int, fs_hz)


def detection_accuracy(true_peaks: np.ndarray, det_peaks: np.ndarray,
                       fs_hz: float, tolerance_ms: float = 50.0) -> dict:
    tol = int(tolerance_ms / 1000 * fs_hz)
    matched_true = set()
    matched_det = set()
    for i, t in enumerate(true_peaks):
        for j, d in enumerate(det_peaks):
            if j in matched_det:
                continue
            if abs(int(d) - int(t)) <= tol:
                matched_true.add(i)
                matched_det.add(j)
                break
    tp = len(matched_true)
    fn = len(true_peaks) - tp
    fp = len(det_peaks) - len(matched_det)
    sens = tp / (tp + fn) if tp + fn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    return {"TP": tp, "FN": fn, "FP": fp, "sensitivity": sens, "ppv": ppv}
