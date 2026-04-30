"""FIT file parsing — records, laps, session metadata."""

from __future__ import annotations

import pandas as pd
from fitparse import FitFile


def get_session_meta(fit_path: str) -> dict:
    """
    Return session-level metadata from the FIT file.

    Keys: sport, sub_sport, start_time, total_distance, total_calories,
          total_elapsed_time, total_ascent, total_descent.
    Returns an empty dict if no session message is found or file is missing.
    """
    try:
        fit = FitFile(fit_path)
    except (FileNotFoundError, OSError):
        return {}
    for msg in fit.get_messages("session"):
        data = {f.name: f.value for f in msg}
        return {
            "sport": str(data.get("sport", "unknown")).lower(),
            "sub_sport": str(data.get("sub_sport", "")).lower(),
            "start_time": str(data.get("start_time", "")),
            "total_distance": data.get("total_distance"),
            "total_calories": data.get("total_calories"),
            "total_elapsed_time": data.get("total_elapsed_time"),
            "total_ascent": data.get("total_ascent"),
            "total_descent": data.get("total_descent"),
        }
    return {}


def decode_lr_balance(raw) -> float | None:
    """
    Decode Garmin's left_right_balance uint8 field.

    Bit 7 is the right-dominant flag; bits 0-6 are left percentage (0-100).
    Returns left% as a float, or None if the value cannot be decoded.
    """
    if raw is None:
        return None
    try:
        return float(int(raw) & 0x7F)
    except (TypeError, ValueError):
        return None


def parse_fit_to_dataframe(fit_path: str) -> pd.DataFrame:
    """
    Parse all record messages from a FIT file into a cleaned DataFrame.

    Columns: timestamp, heart_rate, power, accumulated_power, left_pct,
             core_temperature, skin_temperature, heat_strain_index,
             stryd_temp, stryd_humidity, altitude, distance.

    Power outlier filtering:
        Corrupt Stryd readings (sustained blocks of implausibly high watts)
        are removed using an IQR upper fence (Q75 + 3 x IQR).

    Raises ValueError if no record messages are found.
    """
    records = []
    for record in FitFile(fit_path).get_messages("record"):
        data = {f.name: f.value for f in record}
        ts = data.get("timestamp")
        if ts is None:
            continue
        power = data.get("power") or data.get("Power")
        lr_raw = data.get("left_right_balance")
        records.append(
            {
                "timestamp": ts,
                "heart_rate": data.get("heart_rate"),
                "power": power,
                "accumulated_power": data.get("accumulated_power"),
                "left_pct": decode_lr_balance(lr_raw),
                "core_temperature": data.get("core_temperature"),
                "skin_temperature": data.get("skin_temperature"),
                "heat_strain_index": data.get("heat_strain_index"),
                "stryd_temp": data.get("Stryd Temperature"),
                "stryd_humidity": data.get("Stryd Humidity"),
                "altitude": data.get("enhanced_altitude") or data.get("altitude"),
                "distance": data.get("distance"),
            }
        )

    if not records:
        raise ValueError(f"No record messages found in {fit_path}")

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [
        "heart_rate", "power", "accumulated_power", "left_pct",
        "core_temperature", "skin_temperature", "heat_strain_index",
        "stryd_temp", "stryd_humidity", "altitude", "distance",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove corrupt power readings via IQR upper fence
    if df["power"].notna().sum() > 60:
        q25 = df["power"].quantile(0.25)
        q75 = df["power"].quantile(0.75)
        ceiling = q75 + 3 * (q75 - q25)
        df.loc[df["power"] > ceiling, "power"] = float("nan")

    return df


def derive_power_from_accumulated(df: pd.DataFrame) -> pd.Series:
    """
    Derive a clean instantaneous power series from the accumulated_power field.

    The accumulated_power field is a monotonically increasing cumulative watt
    counter maintained by the power meter. It is immune to the zero-dropout
    problem (ERG mode blips, coasting, brief dropouts) that affects the
    instantaneous power field.

    Three artefact types are handled:

    1. Irregular update cadence -- power meters commonly write
       accumulated_power every 2 s rather than every 1 s, causing alternating
       zero/spike artefacts in a naive 1s diff. Resolved by differentiating
       over a 4-second rolling window.

    2. Counter resets -- accumulated_power occasionally resets to zero
       mid-session (lap boundary, power meter reconnection). Detected as any
       decrease in the accumulated value; each continuous segment is
       differentiated independently.

    3. Catch-up jumps after recording gaps -- if a gap longer than the
       forward-fill limit exists, the resuming value creates a brief burst of
       implausibly high derived power. Clipped using the session IQR upper
       fence.

    Returns an empty Series if accumulated_power data is not present.
    """
    ap = df.dropna(subset=["accumulated_power"]).set_index("timestamp")["accumulated_power"]
    if ap.empty:
        return pd.Series(dtype=float)

    # Detect counter resets: any point where accumulated_power decreases
    diffs = ap.diff()
    reset_points = diffs[diffs < 0].index
    segment_starts = [ap.index[0]] + list(reset_points)
    segment_ends   = list(reset_points) + [ap.index[-1] + pd.Timedelta(seconds=1)]

    WINDOW = 4
    segments = []
    for start, end in zip(segment_starts, segment_ends):
        seg = ap.loc[start:end]
        if len(seg) < WINDOW + 1:
            continue

        # Resample to 1s, forward-fill brief gaps (<=5 s)
        seg_1s = seg.resample("1s").mean().ffill(limit=5)

        # Rolling-window differentiation smooths the 2s update cadence
        power_seg = (seg_1s - seg_1s.shift(WINDOW)) / WINDOW
        power_seg = power_seg.clip(lower=0)
        segments.append(power_seg)

    if not segments:
        return pd.Series(dtype=float)

    power_1s = pd.concat(segments).sort_index()
    # Remove duplicate timestamps at segment boundaries
    power_1s = power_1s[~power_1s.index.duplicated(keep="last")]

    # Final clip: remove any residual catch-up spikes using IQR upper fence
    valid = power_1s[power_1s > 0]
    if len(valid) > 60:
        q25 = valid.quantile(0.25)
        q75 = valid.quantile(0.75)
        ceiling = q75 + 3 * (q75 - q25)
        power_1s = power_1s.clip(upper=ceiling)

    return power_1s


def parse_laps(fit_path: str, df_records: pd.DataFrame | None = None) -> list[dict]:
    """
    Parse lap messages from a FIT file.

    If df_records (the cleaned record DataFrame) is provided, lap average power
    is recomputed from the cleaned records rather than trusting the Lap Power
    field, which can be inflated by Stryd firmware glitches.

    Each lap dict contains: duration_s, distance_m, avg_hr, avg_power,
    start_time, end_time.
    """
    laps = []
    for lap in FitFile(fit_path).get_messages("lap"):
        data = {f.name: f.value for f in lap}
        fit_avg_power = data.get("avg_power") or data.get("Lap Power")
        laps.append(
            {
                "duration_s": data.get("total_timer_time"),
                "distance_m": data.get("total_distance"),
                "avg_hr": data.get("avg_heart_rate"),
                "avg_power": fit_avg_power,
                "start_time": data.get("start_time"),
                "end_time": data.get("timestamp"),
            }
        )

    if df_records is not None and "power" in df_records.columns:
        for lap in laps:
            s = lap.get("start_time")
            dur = lap.get("duration_s")
            if s and dur and df_records["power"].notna().any():
                s_ts = pd.Timestamp(s)
                e_ts = s_ts + pd.Timedelta(seconds=dur)
                mask = (df_records["timestamp"] >= s_ts) & (
                    df_records["timestamp"] <= e_ts
                )
                lap_pwr = df_records.loc[mask, "power"].dropna()
                lap["avg_power"] = (
                    round(float(lap_pwr.mean()), 0) if len(lap_pwr) > 0 else None
                )

    return laps
