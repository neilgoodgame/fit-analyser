"""Display formatting helpers."""

from __future__ import annotations

import pandas as pd


def fmt_duration(seconds) -> str:
    """Format seconds as mm:ss or h:mm:ss."""
    if seconds is None or (isinstance(seconds, float) and pd.isna(seconds)):
        return "N/A"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_distance(metres) -> str:
    """Format metres as km to 2 dp, or 'indoor' for zero-distance activities."""
    if metres is None or (isinstance(metres, float) and pd.isna(metres)):
        return "N/A"
    if metres == 0:
        return "indoor"
    return f"{metres / 1000:.2f} km"


def fmt_pace(duration_s, distance_m) -> str:
    """
    Format pace as mm:ss /km from lap duration (seconds) and distance (metres).
    Returns 'N/A' for treadmill laps or missing data.
    """
    if (
        duration_s is None
        or distance_m is None
        or distance_m == 0
        or (isinstance(distance_m, float) and pd.isna(distance_m))
    ):
        return "N/A"
    pace_s_per_km = duration_s / (distance_m / 1000)
    m, s = divmod(int(round(pace_s_per_km)), 60)
    return f"{m}:{s:02d} /km"


def dur_label(duration_s: int) -> str:
    """Return a human-readable label for a curve duration in seconds."""
    if duration_s < 60:
        return f"{duration_s}s"
    if duration_s < 3600:
        return f"{duration_s // 60}min"
    return f"{duration_s // 3600}hr"
