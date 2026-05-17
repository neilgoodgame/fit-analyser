"""FIT file parsing — records, laps, session metadata."""

from __future__ import annotations

import pandas as pd
from fitparse import FitFile


def training_effect_label(te: float | None) -> str | None:
    """
    Map a Garmin training effect value (0–5) to its benefit label.

    Scale per Garmin documentation:
        0.0–0.9  No Benefit
        1.0–1.9  Minor Benefit
        2.0–2.9  Maintaining
        3.0–3.9  Improving
        4.0–4.9  Highly Improving
        5.0      Overreaching
    """
    if te is None:
        return None
    if te < 1.0:
        return "No Benefit"
    if te < 2.0:
        return "Minor Benefit"
    if te < 3.0:
        return "Maintaining"
    if te < 4.0:
        return "Improving"
    if te < 5.0:
        return "Highly Improving"
    return "Overreaching"


def get_session_meta(fit_path: str) -> dict:
    """
    Return session-level metadata from the FIT file.

    Keys: sport, sub_sport, start_time, total_distance, total_calories,
          total_elapsed_time, total_ascent, total_descent,
          total_training_effect, total_anaerobic_training_effect,
          primary_benefit.
    Returns an empty dict if no session message is found or file is missing.
    """
    try:
        fit = FitFile(fit_path)
    except (FileNotFoundError, OSError):
        return {}
    for msg in fit.get_messages("session"):
        data = {f.name: f.value for f in msg}
        aerobic_te  = data.get("total_training_effect")
        anaerobic_te = data.get("total_anaerobic_training_effect")
        return {
            "sport": str(data.get("sport", "unknown")).lower(),
            "sub_sport": str(data.get("sub_sport", "")).lower(),
            "start_time": str(data.get("start_time", "")),
            "total_distance": data.get("total_distance"),
            "total_calories": data.get("total_calories"),
            "total_elapsed_time": data.get("total_elapsed_time"),
            "total_ascent": data.get("total_ascent"),
            "total_descent": data.get("total_descent"),
            "total_training_effect": aerobic_te,
            "total_anaerobic_training_effect": anaerobic_te,
            "primary_benefit": training_effect_label(aerobic_te),
            "training_stress_score": data.get("training_stress_score"),
            "intensity_factor": data.get("intensity_factor"),
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
        seg_1s = seg.resample("1s").mean().ffill(limit=5)
        power_seg = (seg_1s - seg_1s.shift(WINDOW)) / WINDOW
        power_seg = power_seg.clip(lower=0)
        segments.append(power_seg)

    if not segments:
        return pd.Series(dtype=float)

    power_1s = pd.concat(segments).sort_index()
    power_1s = power_1s[~power_1s.index.duplicated(keep="last")]

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

    If df_records is provided, lap average power is recomputed from the
    cleaned records (guarding against corrupt Stryd Lap Power values).

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


def synthetic_laps(df: pd.DataFrame, lap_distance_km: float) -> list[dict]:
    """
    Generate synthetic laps at fixed distance intervals from the record DataFrame.

    Used when --lap-distance is passed on the CLI, overriding whatever lap
    structure is encoded in the FIT file.

    Args:
        df:               Cleaned record DataFrame (output of parse_fit_to_dataframe).
        lap_distance_km:  Interval in kilometres (e.g. 5.0 creates one lap per 5 km).

    Returns:
        List of lap dicts with the same keys as parse_laps:
            duration_s, distance_m, avg_hr, avg_power, start_time, end_time.
        The final partial lap (< lap_distance_km) is always included.

    Returns an empty list if the DataFrame has no distance data.
    """
    if "distance" not in df.columns or df["distance"].isna().all():
        return []

    lap_distance_m = lap_distance_km * 1000
    total_dist = df["distance"].dropna().max()

    laps = []
    boundary = 0.0

    while boundary < total_dist:
        next_boundary = boundary + lap_distance_m
        mask = (df["distance"] >= boundary) & (df["distance"] < next_boundary)
        seg = df[mask]

        if len(seg) < 2:
            boundary = next_boundary
            continue

        duration_s = (
            seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]
        ).total_seconds()
        distance_m = float(
            seg["distance"].iloc[-1] - seg["distance"].iloc[0]
        )
        avg_hr_vals = seg["heart_rate"].dropna()
        avg_hr = int(round(float(avg_hr_vals.mean()))) if len(avg_hr_vals) > 0 else None

        avg_pwr_vals = seg["power"].dropna() if "power" in seg.columns else pd.Series(dtype=float)
        avg_power = (
            round(float(avg_pwr_vals.mean()), 0) if len(avg_pwr_vals) > 0 else None
        )

        laps.append(
            {
                "duration_s": round(duration_s, 3),
                "distance_m": round(distance_m, 2),
                "avg_hr": avg_hr,
                "avg_power": avg_power,
                "start_time": seg["timestamp"].iloc[0],
                "end_time": seg["timestamp"].iloc[-1],
            }
        )

        boundary = next_boundary

    return laps
