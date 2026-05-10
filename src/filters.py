"""
ECG digital filters built from scratch (numpy-only).
Bandpass FIR via windowed-sinc design (Hamming window) and a notch
filter as a difference of two lowpass kernels.
"""
from __future__ import annotations

import numpy as np


def _sinc_lowpass(cutoff_hz: float, fs_hz: float, num_taps: int) -> np.ndarray:
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps) - (num_taps - 1) / 2
    fc = cutoff_hz / fs_hz
    h = np.sinc(2 * fc * n) * 2 * fc
    h *= np.hamming(num_taps)
    h /= h.sum()
    return h


def design_bandpass(low_hz: float, high_hz: float, fs_hz: float,
                    num_taps: int = 201) -> np.ndarray:
    if not 0 < low_hz < high_hz < fs_hz / 2:
        raise ValueError("Need 0 < low < high < fs/2")
    h_low = _sinc_lowpass(high_hz, fs_hz, num_taps)
    h_high = _sinc_lowpass(low_hz, fs_hz, num_taps)
    return h_low - h_high


def design_notch(notch_hz: float, fs_hz: float, bw_hz: float = 2.0,
                 num_taps: int = 201) -> np.ndarray:
    if not 0 < notch_hz < fs_hz / 2:
        raise ValueError("notch_hz must be in (0, fs/2)")
    low = max(0.5, notch_hz - bw_hz / 2)
    high = min(fs_hz / 2 - 1, notch_hz + bw_hz / 2)
    h_bp = design_bandpass(low, high, fs_hz, num_taps)
    impulse = np.zeros(num_taps)
    impulse[num_taps // 2] = 1.0
    return impulse - h_bp


def apply_fir(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    fwd = np.convolve(signal, taps, mode="same")
    rev = np.convolve(fwd[::-1], taps, mode="same")[::-1]
    return rev


def remove_baseline_wander(signal: np.ndarray, fs_hz: float,
                           cutoff_hz: float = 0.5) -> np.ndarray:
    n = len(signal)
    target = int(8 * fs_hz / cutoff_hz)
    num_taps = min(target | 1, max(3, (n - 1) | 1))
    if num_taps % 2 == 0:
        num_taps -= 1
    h_lp = _sinc_lowpass(cutoff_hz, fs_hz, num_taps)
    baseline = apply_fir(signal, h_lp)
    return signal - baseline


def clean_ecg(signal: np.ndarray, fs_hz: float, *,
              powerline_hz: float = 50.0,
              bandpass: tuple[float, float] = (0.5, 40.0)) -> np.ndarray:
    sig = remove_baseline_wander(signal, fs_hz)
    sig = apply_fir(sig, design_notch(powerline_hz, fs_hz))
    sig = apply_fir(sig, design_bandpass(bandpass[0], bandpass[1], fs_hz))
    return sig
