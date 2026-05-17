"""Metrics computation — duration curves, best averages, heat stress."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fit_analyser.constants import CURVE_DURATIONS, HEAT_ZONES

# np.trapezoid was introduced in NumPy 2.0; older versions use np.trapz.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def best_average(series: pd.Series, window_minutes: int, label: str = "") -> float:
    """
    Return the highest rolling average of series over window_minutes minutes.

    Resamples to 1-second intervals (forward-filling gaps <= 5 s) so the
    window is strictly time-based regardless of recording cadence.
    Returns NaN if the activity is shorter than the window.
    """
    resampled = series.resample("1s").mean().ffill(limit=5)
    window_s = window_minutes * 60
    if len(resampled) < window_s:
        return float("nan")
    # Use time-based offset string rather than integer window — integer windows
    # on a DatetimeIndex with freq set behave inconsistently across pandas versions.
    result = resampled.rolling(f"{window_s}s").mean().max()
    return float(result) if pd.notna(result) else float("nan")


def compute_curve(series: pd.Series, value_key: str) -> list[dict]:
    """
    Compute the best average for each duration in CURVE_DURATIONS.

    Returns a list of dicts: [{"duration_s": int, value_key: float}, ...]
    """
    resampled = series.resample("1s").mean().ffill(limit=5).fillna(0)
    out = []
    for d in CURVE_DURATIONS:
        if len(resampled) >= d:
            best = resampled.rolling(f"{d}s").mean().max()
            out.append({"duration_s": d, value_key: round(float(best), 1)})
    return out


def compute_pdc(
    df: pd.DataFrame,
    accumulated_power_series: "pd.Series | None" = None,
) -> list[dict]:
    """
    Power duration curve — best average watts per duration.

    If accumulated_power_series is provided (a time-indexed 1s Series derived
    from the accumulated_power field), it is used instead of the instantaneous
    power column. This eliminates zero-dropout artefacts from ERG mode or
    brief power meter signal losses.
    """
    if accumulated_power_series is not None and not accumulated_power_series.empty:
        series = accumulated_power_series.dropna()
    else:
        series = df.dropna(subset=["power"]).set_index("timestamp")["power"]
    return compute_curve(series, "power_w")


def compute_hdc(df: pd.DataFrame) -> list[dict]:
    """Heart rate duration curve — best average bpm per duration."""
    series = df.dropna(subset=["heart_rate"]).set_index("timestamp")["heart_rate"]
    return compute_curve(series, "hr_bpm")


def compute_heat_stress(df: pd.DataFrame) -> dict | None:
    """
    Compute cumulative heat stress metrics from the heat_strain_index column.

    Leading zeros (HRM-Pro sensor initialisation lag, typically 14-18 min)
    are excluded before integration.

    Returns a dict with:
        hsi_min    -- time-integral (area under HSI curve) in HSI.min
        active_min -- active sensor window in minutes
        init_min   -- initialisation period excluded in minutes
        peak_60s   -- peak 60-second rolling average HSI
        zones      -- per-zone breakdown list (time, %, HSI.min contribution)

    Returns None if fewer than 60 seconds of non-zero HSI data exist.

    Zone definitions (Garmin HRM-Pro guidance):
        Zone 1: 0.0-0.9  No Heat Strain          Optimal Performance
        Zone 2: 1.0-2.9  Moderate Heat Strain    Potential Performance Decline
        Zone 3: 3.0-6.9  High Heat Strain        Performance Decline
        Zone 4: 7.0-10   Extremely High Strain   Dangerous
    """
    if "heat_strain_index" not in df.columns:
        return None
    s = df.dropna(subset=["heat_strain_index"])[["timestamp", "heat_strain_index"]].copy()
    if s.empty:
        return None
    s = s.sort_values("timestamp").reset_index(drop=True)
    s["hsi"] = pd.to_numeric(s["heat_strain_index"], errors="coerce")
    t0 = s["timestamp"].iloc[0]
    s["elapsed_s"] = s["timestamp"].apply(lambda t: (t - t0).total_seconds())

    nonzero = s[s["hsi"] > 0]
    if len(nonzero) < 60:
        return None

    first_idx = nonzero.index[0]
    active = s.iloc[first_idx:].copy()
    init_min = active["elapsed_s"].iloc[0] / 60
    active_min = (active["elapsed_s"].iloc[-1] - active["elapsed_s"].iloc[0]) / 60

    hsi_min = float(_trapz(active["hsi"], active["elapsed_s"]) / 60.0)

    active_ts = active.set_index("timestamp")["hsi"].resample("1s").mean().ffill(limit=5)
    peak_60s = float(active_ts.rolling(60, min_periods=30).mean().max())

    zones = []
    for z in HEAT_ZONES:
        in_zone = active[(active["hsi"] >= z["lo"]) & (active["hsi"] < z["hi"])]
        mins = round(float(len(in_zone) / 60), 1)
        pct = round(mins / active_min * 100, 1) if active_min > 0 else 0.0
        z_integral = round(
            float(_trapz(in_zone["hsi"].values, in_zone["elapsed_s"].values) / 60.0)
            if len(in_zone) > 1
            else 0.0,
            1,
        )
        zones.append({**z, "mins": mins, "pct": pct, "hsi_min": z_integral})

    return {
        "hsi_min": round(hsi_min, 1),
        "active_min": round(active_min, 1),
        "init_min": round(init_min, 1),
        "peak_60s": round(peak_60s, 2),
        "zones": zones,
    }
