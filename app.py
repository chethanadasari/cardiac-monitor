"""
Cardiac Monitor — live ECG dashboard.

Streamlit app that simulates a single-lead ECG, runs Pan-Tompkins
QRS detection in real time, displays live heart rate, HRV metrics,
and rhythm classification.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from ecg_generator import ECGConfig, generate_ecg
from filters import clean_ecg
from qrs_detector import detect_r_peaks
from hrv import classify_rhythm, hrv_metrics, rhythm_color, rhythm_label


st.set_page_config(page_title="Cardiac Monitor — Live ECG",
                   page_icon="❤️", layout="wide")

st.markdown(
    """
    <style>
      .metric-big {font-size: 3rem; font-weight: 700; line-height: 1;}
      .metric-unit {font-size: 1rem; color: #64748b; margin-left: 4px;}
      .rhythm-badge {
          display: inline-block; padding: 8px 18px; border-radius: 999px;
          color: white; font-weight: 600; font-size: 1.05rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("❤️ Cardiac Monitor — Live ECG Dashboard")
st.caption(
    "Single-lead ECG simulator with Pan-Tompkins QRS detection, HRV analysis, "
    "and real-time rhythm classification."
)


with st.sidebar:
    st.header("Patient simulation")
    rhythm = st.selectbox(
        "Rhythm",
        options=["normal", "bradycardia", "tachycardia", "afib"],
        index=0,
        format_func=lambda r: rhythm_label(r),
    )
    base_hr = st.slider("Base heart rate (BPM)", 40, 160, 75, 5)
    st.divider()
    st.header("Noise (per-channel)")
    snr_db = st.slider("AWGN SNR (dB)", -5, 40, 20, 1)
    bw_amp = st.slider("Baseline wander amplitude (mV)", 0.0, 0.4, 0.05, 0.01)
    pl_amp = st.slider("Powerline interference (mV)", 0.0, 0.3, 0.03, 0.01)
    pl_hz = st.radio("Powerline frequency", [50, 60], index=0, horizontal=True)
    st.divider()
    st.header("Signal processing")
    apply_filter = st.toggle("Apply ECG cleaning pipeline", value=True)
    st.divider()
    st.header("Live mode")
    duration_s = st.slider("Window length (s)", 5, 30, 10, 1)
    live = st.toggle("Run live (animated)", value=False)
    if live:
        speed = st.slider("Playback speed", 0.25, 2.0, 1.0, 0.25)


@st.cache_data(show_spinner=False)
def _make_signal(rhythm, base_hr, snr_db, bw_amp, pl_amp, pl_hz,
                 duration_s, seed):
    cfg = ECGConfig(fs_hz=250, duration_s=duration_s, rhythm=rhythm,
                    base_hr_bpm=base_hr, awgn_db=snr_db,
                    baseline_wander_amp=bw_amp, powerline_amp=pl_amp,
                    powerline_hz=pl_hz, seed=seed)
    return generate_ecg(cfg)


if "seed" not in st.session_state:
    st.session_state.seed = 7

col_seed, _ = st.columns([1, 5])
with col_seed:
    if st.button("🔄 New patient"):
        st.session_state.seed = int(time.time()) % 10_000

t, ecg_raw, true_peaks = _make_signal(
    rhythm, base_hr, snr_db, bw_amp, pl_amp, int(pl_hz),
    duration_s, st.session_state.seed,
)

ecg = clean_ecg(ecg_raw, fs_hz=250, powerline_hz=float(pl_hz)) if apply_filter else ecg_raw
det_peaks = detect_r_peaks(ecg, fs_hz=250)
metrics = hrv_metrics(det_peaks, fs_hz=250)
predicted = classify_rhythm(metrics)


m1, m2, m3, m4, m5 = st.columns([1.2, 1, 1, 1, 1.6])

with m1:
    hr_display = "—" if np.isnan(metrics.mean_hr_bpm) else f"{metrics.mean_hr_bpm:.0f}"
    st.markdown(
        f'<div class="metric-big" style="color:#dc2626;">{hr_display}'
        f'<span class="metric-unit">BPM</span></div>'
        f'<div style="color:#64748b;">Mean heart rate</div>',
        unsafe_allow_html=True,
    )
with m2:
    st.metric("SDNN", f"{metrics.sdnn_ms:.0f} ms" if metrics.n_beats > 1 else "—")
with m3:
    st.metric("RMSSD", f"{metrics.rmssd_ms:.0f} ms" if metrics.n_beats > 2 else "—")
with m4:
    st.metric("pNN50", f"{metrics.pnn50_pct:.0f} %" if metrics.n_beats > 2 else "—")
with m5:
    color = rhythm_color(predicted)
    st.markdown(
        f'<div style="margin-top:6px;">Rhythm classification</div>'
        f'<div class="rhythm-badge" style="background:{color}; margin-top:6px;">'
        f'{rhythm_label(predicted)}</div>',
        unsafe_allow_html=True,
    )

st.caption(f"True beats: {len(true_peaks)}  ·  Detected: {len(det_peaks)}")
st.divider()


def _draw_ecg(ax, t_window, ecg_window, peaks_window, t_min, t_max):
    ax.clear()
    ax.plot(t_window, ecg_window, color="#1e293b", linewidth=1.0)
    if len(peaks_window):
        ax.plot(t_window[peaks_window], ecg_window[peaks_window],
                "o", color="#dc2626", markersize=8, label="R peak")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(np.min(ecg_window) - 0.3, np.max(ecg_window) + 0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_xticks(np.arange(np.floor(t_min), np.ceil(t_max) + 0.001, 0.2))
    ax.set_xticks(np.arange(np.floor(t_min), np.ceil(t_max) + 0.001, 0.04), minor=True)
    ax.grid(which="major", color="#fca5a5", linewidth=0.8, alpha=0.4)
    ax.grid(which="minor", color="#fecaca", linewidth=0.5, alpha=0.3)


st.subheader("ECG strip")
plot_slot = st.empty()

if not live:
    fig, ax = plt.subplots(figsize=(13, 3.5), dpi=110)
    show_n = min(len(t), int(10 * 250))
    _draw_ecg(ax, t[:show_n], ecg[:show_n],
              det_peaks[det_peaks < show_n], 0, t[show_n - 1])
    fig.tight_layout()
    plot_slot.pyplot(fig)
    plt.close(fig)
else:
    fps = 20
    window_s = 6.0
    win_n = int(window_s * 250)
    end = win_n
    while end <= len(t):
        fig, ax = plt.subplots(figsize=(13, 3.5), dpi=110)
        start = end - win_n
        seg_t = t[start:end] - t[start]
        seg_ecg = ecg[start:end]
        local_peaks = det_peaks[(det_peaks >= start) & (det_peaks < end)] - start
        _draw_ecg(ax, seg_t, seg_ecg, local_peaks, 0, window_s)
        fig.tight_layout()
        plot_slot.pyplot(fig)
        plt.close(fig)
        time.sleep(max(0.0, (1.0 / fps) / float(speed)))
        end += int(250 / fps)


st.divider()
st.subheader("Rhythm analysis")

g1, g2 = st.columns(2)

with g1:
    st.markdown("**Tachogram (RR intervals over time)**")
    if metrics.n_beats > 2:
        rr_ms = np.diff(det_peaks) / 250 * 1000.0
        beat_idx = np.arange(1, len(rr_ms) + 1)
        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=110)
        ax.plot(beat_idx, rr_ms, "o-", color="#0ea5e9", markersize=4, linewidth=1)
        ax.set_xlabel("Beat #")
        ax.set_ylabel("RR (ms)")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough beats yet.")

with g2:
    st.markdown("**Poincaré plot — RR(n) vs RR(n+1)**")
    if metrics.n_beats > 3:
        rr_ms = np.diff(det_peaks) / 250 * 1000.0
        x = rr_ms[:-1]
        y = rr_ms[1:]
        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=110)
        ax.scatter(x, y, color=rhythm_color(predicted), alpha=0.7)
        lim_low = max(200, min(np.min(x), np.min(y)) - 100)
        lim_high = min(2200, max(np.max(x), np.max(y)) + 100)
        ax.plot([lim_low, lim_high], [lim_low, lim_high],
                "--", color="#94a3b8", linewidth=0.8)
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_xlabel("RR(n)  (ms)")
        ax.set_ylabel("RR(n+1)  (ms)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Need at least 4 beats.")


st.divider()
st.caption(
    "Built with NumPy + Matplotlib + Streamlit. DSP pipeline implemented from "
    "scratch (windowed-sinc FIR + Pan-Tompkins QRS). Educational simulator — "
    "not for clinical use."
)
