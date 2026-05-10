"""Unit tests for the cardiac-monitor pipeline."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ecg_generator import ECGConfig, generate_ecg, heart_rate_bpm
from filters import (apply_fir, clean_ecg, design_bandpass, design_notch,
                     remove_baseline_wander)
from qrs_detector import detect_r_peaks, detection_accuracy
from hrv import classify_rhythm, hrv_metrics

FS = 250


@pytest.mark.parametrize("rhythm,low_bpm,high_bpm", [
    ("normal", 65, 90),
    ("bradycardia", 35, 55),
    ("tachycardia", 115, 145),
])
def test_ecg_generator_hr_in_expected_band(rhythm, low_bpm, high_bpm):
    cfg = ECGConfig(rhythm=rhythm, duration_s=20.0, seed=0)
    _, _, peaks = generate_ecg(cfg)
    hr = heart_rate_bpm(peaks, FS)
    assert low_bpm <= hr <= high_bpm


def test_ecg_generator_signal_amplitude_reasonable():
    cfg = ECGConfig(rhythm="normal", duration_s=10.0, seed=0)
    _, ecg, _ = generate_ecg(cfg)
    assert 0.6 < np.max(ecg) < 2.0
    assert -1.0 < np.min(ecg) < 0.0


def test_ecg_generator_afib_more_irregular_than_normal():
    n = generate_ecg(ECGConfig(rhythm="normal", duration_s=30.0, seed=1))[2]
    a = generate_ecg(ECGConfig(rhythm="afib", duration_s=30.0, seed=1))[2]
    assert np.std(np.diff(a) / FS) > 2 * np.std(np.diff(n) / FS)


def test_bandpass_passes_centre_frequency():
    fs = 250
    taps = design_bandpass(5.0, 15.0, fs, num_taps=201)
    t = np.arange(2 * fs) / fs
    out = apply_fir(np.sin(2 * np.pi * 10 * t), taps)[fs // 2:-fs // 2]
    assert 0.7 < np.max(np.abs(out)) < 1.3


def test_bandpass_attenuates_dc():
    fs = 250
    taps = design_bandpass(5.0, 15.0, fs, num_taps=201)
    out = apply_fir(np.ones(2 * fs), taps)[fs // 2:-fs // 2]
    assert np.max(np.abs(out)) < 0.05


def test_notch_attenuates_powerline():
    fs = 250
    taps = design_notch(50.0, fs, bw_hz=2.0, num_taps=201)
    t = np.arange(4 * fs) / fs
    out = apply_fir(np.sin(2 * np.pi * 50 * t), taps)[fs:-fs]
    assert np.max(np.abs(out)) < 0.3


def test_notch_preserves_passband():
    fs = 250
    taps = design_notch(50.0, fs, bw_hz=2.0, num_taps=201)
    t = np.arange(2 * fs) / fs
    out = apply_fir(np.sin(2 * np.pi * 10 * t), taps)[fs // 2:-fs // 2]
    assert 0.7 < np.max(np.abs(out)) < 1.3


def test_remove_baseline_wander_kills_drift():
    fs = 250
    t = np.arange(30 * fs) / fs
    drift = 0.5 * np.sin(2 * np.pi * 0.2 * t)
    clean = remove_baseline_wander(drift, fs)
    mid = clean[5 * fs:-5 * fs]
    drift_mid = drift[5 * fs:-5 * fs]
    assert np.std(mid) < 0.15 * np.std(drift_mid)


def test_clean_ecg_keeps_qrs_intact():
    cfg = ECGConfig(rhythm="normal", duration_s=10.0, awgn_db=15,
                    baseline_wander_amp=0.1, powerline_amp=0.05, seed=0)
    _, raw, _ = generate_ecg(cfg)
    cleaned = clean_ecg(raw, FS)
    assert np.max(cleaned) > 0.4


@pytest.mark.parametrize("rhythm", ["normal", "bradycardia", "tachycardia"])
def test_qrs_detection_clean_signal(rhythm):
    cfg = ECGConfig(rhythm=rhythm, duration_s=20.0, seed=0)
    _, ecg, true_p = generate_ecg(cfg)
    det_p = detect_r_peaks(ecg, FS)
    acc = detection_accuracy(true_p, det_p, FS, tolerance_ms=80)
    assert acc["sensitivity"] >= 0.95
    assert acc["ppv"] >= 0.95


def test_qrs_detection_with_noise():
    cfg = ECGConfig(rhythm="normal", duration_s=20.0, awgn_db=10,
                    baseline_wander_amp=0.1, powerline_amp=0.05, seed=0)
    _, raw, true_p = generate_ecg(cfg)
    cleaned = clean_ecg(raw, FS)
    det_p = detect_r_peaks(cleaned, FS)
    acc = detection_accuracy(true_p, det_p, FS, tolerance_ms=80)
    assert acc["sensitivity"] >= 0.85
    assert acc["ppv"] >= 0.85


def test_qrs_detection_short_signal_returns_empty():
    assert len(detect_r_peaks(np.zeros(50), FS)) == 0


def test_hrv_metrics_normal_signal():
    _, _, peaks = generate_ecg(ECGConfig(rhythm="normal", duration_s=30.0, seed=0))
    m = hrv_metrics(peaks, FS)
    assert 60 < m.mean_hr_bpm < 100
    assert 10 < m.sdnn_ms < 200
    assert m.cv < 0.20


def test_hrv_metrics_afib_high_variability():
    _, _, peaks = generate_ecg(ECGConfig(rhythm="afib", duration_s=30.0, seed=0))
    m = hrv_metrics(peaks, FS)
    assert m.cv > 0.20
    assert m.sdnn_ms > 100


def test_hrv_metrics_few_beats_safe():
    m = hrv_metrics(np.array([100], dtype=int), FS)
    assert m.n_beats == 1
    assert np.isnan(m.mean_hr_bpm)


@pytest.mark.parametrize("rhythm", ["normal", "bradycardia", "tachycardia", "afib"])
def test_rhythm_classifier_round_trip(rhythm):
    cfg = ECGConfig(rhythm=rhythm, duration_s=30.0, seed=0)
    _, ecg, _ = generate_ecg(cfg)
    cleaned = clean_ecg(ecg, FS)
    det_p = detect_r_peaks(cleaned, FS)
    metrics = hrv_metrics(det_p, FS)
    predicted = classify_rhythm(metrics)
    assert predicted == rhythm
