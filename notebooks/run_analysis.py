"""Generate the analysis plots embedded in the README."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from ecg_generator import ECGConfig, generate_ecg
from filters import clean_ecg
from qrs_detector import detect_r_peaks, detection_accuracy
from hrv import hrv_metrics, rhythm_label

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
FS = 250


def _ecg_paper_grid(ax, t):
    ax.set_xticks(np.arange(np.floor(t.min()), np.ceil(t.max()) + 0.001, 0.2))
    ax.set_xticks(np.arange(np.floor(t.min()), np.ceil(t.max()) + 0.001, 0.04), minor=True)
    ax.grid(which="major", color="#fca5a5", linewidth=0.8, alpha=0.4)
    ax.grid(which="minor", color="#fecaca", linewidth=0.5, alpha=0.3)


def plot_four_rhythms():
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, rhythm in zip(axes, ["normal", "bradycardia", "tachycardia", "afib"]):
        cfg = ECGConfig(rhythm=rhythm, duration_s=8.0, seed=2)
        t, ecg, _ = generate_ecg(cfg)
        det_p = detect_r_peaks(ecg, FS)
        ax.plot(t, ecg, color="#1e293b", linewidth=1.0)
        if len(det_p):
            ax.plot(t[det_p], ecg[det_p], "o", color="#dc2626", markersize=6)
        _ecg_paper_grid(ax, t)
        m = hrv_metrics(det_p, FS)
        ax.set_title(f"{rhythm_label(rhythm)}    HR={m.mean_hr_bpm:.0f} BPM    SDNN={m.sdnn_ms:.0f} ms",
                     fontsize=10)
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rhythms.png"))
    plt.close()
    print("  saved results/rhythms.png")


def plot_clean_pipeline():
    cfg = ECGConfig(rhythm="normal", duration_s=6.0, awgn_db=10,
                    baseline_wander_amp=0.2, powerline_amp=0.08, seed=4)
    t, raw, _ = generate_ecg(cfg)
    cleaned = clean_ecg(raw, FS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(t, raw, color="#9ca3af", linewidth=0.9)
    axes[0].set_title("Raw ECG  (AWGN + baseline wander + 50 Hz powerline)")
    axes[0].set_ylabel("mV")
    _ecg_paper_grid(axes[0], t)
    axes[1].plot(t, cleaned, color="#1e293b", linewidth=1.0)
    det_p = detect_r_peaks(cleaned, FS)
    axes[1].plot(t[det_p], cleaned[det_p], "o", color="#dc2626", markersize=7,
                 label=f"R peaks ({len(det_p)} found)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("After 0.5 Hz HP + 50 Hz notch + 0.5–40 Hz BP")
    axes[1].set_ylabel("mV")
    axes[1].set_xlabel("Time (s)")
    _ecg_paper_grid(axes[1], t)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "clean_pipeline.png"))
    plt.close()
    print("  saved results/clean_pipeline.png")


def plot_sensitivity_vs_snr():
    snrs = np.arange(-5, 31, 2)
    sens = []
    ppv = []
    for snr in snrs:
        cfg = ECGConfig(rhythm="normal", duration_s=20.0, awgn_db=snr,
                        baseline_wander_amp=0.1, powerline_amp=0.05, seed=11)
        _, raw, true_p = generate_ecg(cfg)
        cleaned = clean_ecg(raw, FS)
        det = detect_r_peaks(cleaned, FS)
        acc = detection_accuracy(true_p, det, FS, tolerance_ms=80)
        sens.append(acc["sensitivity"])
        ppv.append(acc["ppv"])
    plt.figure(figsize=(7, 4.5))
    plt.plot(snrs, np.array(sens) * 100, "o-", color="#dc2626", label="Sensitivity")
    plt.plot(snrs, np.array(ppv) * 100, "s-", color="#0ea5e9", label="PPV")
    plt.axhline(95, color="#94a3b8", linewidth=0.8, linestyle="--",
                label="95% clinical target")
    plt.xlabel("SNR (dB)")
    plt.ylabel("%")
    plt.title("Pan-Tompkins QRS detection vs SNR (cleaned signal)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "sensitivity_vs_snr.png"))
    plt.close()
    print("  saved results/sensitivity_vs_snr.png")


def plot_poincare():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, rhythm, color in zip(axes, ["normal", "afib"], ["#16a34a", "#dc2626"]):
        cfg = ECGConfig(rhythm=rhythm, duration_s=120.0, seed=3)
        _, _, peaks = generate_ecg(cfg)
        rr = np.diff(peaks) / FS * 1000
        ax.scatter(rr[:-1], rr[1:], c=color, alpha=0.6, s=22)
        m = hrv_metrics(peaks, FS)
        lo = max(200, np.min(rr) - 100)
        hi = min(2200, np.max(rr) + 100)
        ax.plot([lo, hi], [lo, hi], "--", color="#94a3b8", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("RR(n)  (ms)")
        ax.set_ylabel("RR(n+1)  (ms)")
        ax.set_title(f"{rhythm_label(rhythm)}\nSDNN={m.sdnn_ms:.0f} ms · "
                     f"RMSSD={m.rmssd_ms:.0f} ms · CV={m.cv:.2f}")
        ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "poincare.png"))
    plt.close()
    print("  saved results/poincare.png")


if __name__ == "__main__":
    print("Generating cardiac analysis plots...")
    plot_four_rhythms()
    plot_clean_pipeline()
    plot_sensitivity_vs_snr()
    plot_poincare()
    print("Done. See the results/ folder.")
