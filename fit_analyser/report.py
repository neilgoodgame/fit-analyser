"""HTML report builder — renders activity data into a self-contained dark-themed HTML file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from fit_analyser.metrics import best_average, compute_heat_stress
from fit_analyser.formatting import fmt_duration, fmt_distance, fmt_pace, dur_label

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _ts(series, decimals: int = 1) -> list:
    return [round(v, decimals) if not pd.isna(v) else None for v in series]


def build_html_report(fit_path, df, laps, meta, hdc, pdc, power_source: str = "instantaneous power", hr_zones: list[dict] | None = None):
    sport     = meta.get("sport", "unknown")
    sub_sport = meta.get("sub_sport", "")
    is_run    = sport == "running"
    has_power = df["power"].notna().any()
    has_hr    = df["heart_rate"].notna().any()
    has_lr    = df["left_pct"].notna().any()
    has_core  = df["core_temperature"].notna().any()
    has_skin  = df["skin_temperature"].notna().any()
    has_hsi   = df["heat_strain_index"].notna().any()
    has_temp  = has_core or has_skin or has_hsi
    has_stryd_temp    = bool(df["stryd_temp"].notna().any())
    has_stryd_hum     = bool(df["stryd_humidity"].notna().any())
    has_stryd_ambient = has_stryd_temp or has_stryd_hum
    has_elevation = df["altitude"].notna().any() and df["distance"].notna().any()
    has_hr_zones  = hr_zones is not None and has_hr

    # Pre-compute zone distribution if zones provided
    zone_data = []
    if has_hr_zones:
        hr = df["heart_rate"].dropna()
        total_s = len(hr)
        for z in hr_zones:
            lo, hi = z["min"], z["max"]
            in_z = hr[(hr >= lo) & (hr <= hi)]
            secs = len(in_z)
            pct  = round(secs / total_s * 100, 1) if total_s else 0
            m, s = divmod(secs, 60)
            zone_data.append({
                "name":        z["name"],
                "description": z["description"].split("—")[0].strip() if "—" in z["description"] else z["description"],
                "min":         lo,
                "max":         hi,
                "secs":        secs,
                "time_label":  f"{m}:{s:02d}",
                "pct":         pct,
            })

    duration_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    total_dist   = meta.get("total_distance") or 0
    calories     = meta.get("total_calories") or 0
    start_time   = meta.get("start_time", "")
    total_ascent  = meta.get("total_ascent")
    total_descent = meta.get("total_descent")
    aerobic_te    = meta.get("total_training_effect")
    anaerobic_te  = meta.get("total_anaerobic_training_effect")
    primary_benefit = meta.get("primary_benefit")
    has_training_effect = aerobic_te is not None or anaerobic_te is not None

    hr_series  = df.dropna(subset=["heart_rate"]).set_index("timestamp")["heart_rate"]
    pwr_series = df.dropna(subset=["power"]).set_index("timestamp")["power"]

    avg_hr  = round(hr_series.mean(), 1)  if has_hr    else None
    max_hr  = int(hr_series.max())        if has_hr    else None
    b20_hr  = round(best_average(hr_series,  20, "hr"),  1) if has_hr    else None
    b60_hr  = round(best_average(hr_series,  60, "hr"),  1) if has_hr    else None

    avg_pwr = round(pwr_series.mean(), 1) if has_power else None
    max_pwr = int(pwr_series.max())       if has_power else None
    b20_pwr = round(best_average(pwr_series, 20, "pwr"), 1) if has_power else None
    b60_pwr = round(best_average(pwr_series, 60, "pwr"), 1) if has_power else None

    avg_left = avg_right = None
    if has_lr:
        lr_series = df.dropna(subset=["left_pct"]).set_index("timestamp")["left_pct"]
        avg_left  = round(lr_series.mean(), 1)
        avg_right = round(100 - avg_left, 1)

    balance_note = None
    if has_lr and avg_left is not None:
        balance_note = "balanced" if abs(avg_left - 50) < 1 else ("left-dominant" if avg_left > 50 else "right-dominant")

    def temp_stats(col):
        s = df.dropna(subset=[col]).set_index("timestamp")[col]
        if s.empty: return None, None, None
        return round(s.min(), 1), round(s.mean(), 1), round(s.max(), 1)

    core_min, core_avg, core_max = temp_stats("core_temperature") if has_core else (None, None, None)
    skin_min, skin_avg, skin_max = temp_stats("skin_temperature") if has_skin else (None, None, None)
    hsi_min,  hsi_avg,  hsi_max  = temp_stats("heat_strain_index") if has_hsi  else (None, None, None)

    heat_stress = compute_heat_stress(df) if has_hsi else None
    if heat_stress:
        heat_stress = dict(heat_stress)
        heat_stress["dominant"] = max(heat_stress["zones"], key=lambda z: z["mins"])

    stryd_temp_start = stryd_temp_end = stryd_temp_avg = None
    stryd_hum_start  = stryd_hum_end  = stryd_hum_avg  = None
    if has_stryd_ambient:
        st = df.dropna(subset=["stryd_temp"])["stryd_temp"]
        sh = df.dropna(subset=["stryd_humidity"])["stryd_humidity"]
        if len(st):
            stryd_temp_start = round(float(st.iloc[0]),  1)
            stryd_temp_end   = round(float(st.iloc[-1]), 1)
            stryd_temp_avg   = round(float(st.mean()),   1)
        if len(sh):
            stryd_hum_start  = round(float(sh.iloc[0]),  1)
            stryd_hum_end    = round(float(sh.iloc[-1]), 1)
            stryd_hum_avg    = round(float(sh.mean()),   1)

    hr_dropout_count = int(df["heart_rate"].isna().sum())
    if has_hr and hr_dropout_count > 0:
        hr_interp = (
            df.set_index("timestamp")["heart_rate"]
            .resample("1s").mean()
            .interpolate(method="linear", limit=60, limit_direction="both")
        )
        df = df.copy()
        df["heart_rate_display"] = df["timestamp"].map(
            hr_interp.reset_index().set_index("timestamp")["heart_rate"]
        )
    else:
        df["heart_rate_display"] = df["heart_rate"]

    ts_step   = max(1, len(df) // 600)
    ts_df     = df.iloc[::ts_step].copy()
    t0        = df["timestamp"].iloc[0]
    ts_labels = [str(int((r.timestamp - t0).total_seconds())) for r in ts_df.itertuples()]
    ts_hr          = _ts(ts_df["heart_rate_display"])
    ts_pwr         = _ts(ts_df["power"])
    ts_left        = _ts(ts_df["left_pct"])
    ts_core        = _ts(ts_df["core_temperature"], decimals=2)
    ts_skin        = _ts(ts_df["skin_temperature"], decimals=2)
    ts_hsi         = _ts(ts_df["heat_strain_index"], decimals=2)
    ts_stryd_temp  = _ts(ts_df["stryd_temp"])
    ts_stryd_hum   = _ts(ts_df["stryd_humidity"])

    if has_elevation:
        elev_df = df.dropna(subset=["altitude", "distance"]).copy()
        elev_df["altitude_smooth"] = elev_df["altitude"].rolling(window=30, center=True, min_periods=5).median()
        elev_step   = max(1, len(elev_df) // 500)
        elev_sample = elev_df.iloc[::elev_step]
        ts_elev_dist = [round(v / 1000, 3) for v in elev_sample["distance"]]
        ts_elev_alt  = _ts(elev_sample["altitude_smooth"])
        elev_min = round(float(elev_df["altitude_smooth"].min()), 0)
        elev_max = round(float(elev_df["altitude_smooth"].max()), 0)
    else:
        ts_elev_dist = ts_elev_alt = []
        elev_min = elev_max = 0.0

    hdc_labels = [dur_label(p["duration_s"]) for p in hdc]
    hdc_data   = [p["hr_bpm"]  for p in hdc]
    pdc_labels = [dur_label(p["duration_s"]) for p in pdc]
    pdc_data   = [p["power_w"] for p in pdc]

    zone_palette = ["#34c98a", "#4f8ef7", "#f5c842", "#f0823a", "#e8394a"]
    zone_colors  = [zone_palette[min(i, 4)] for i in range(len(zone_data))]

    laps_ctx = [
        {
            "num":       i,
            "duration":  fmt_duration(lap["duration_s"]),
            "distance":  fmt_distance(lap["distance_m"]),
            "avg_hr":    f"{int(lap['avg_hr'])} bpm" if lap["avg_hr"]    else "—",
            "avg_power": f"{int(lap['avg_power'])} W" if lap["avg_power"] else "—",
            "pace":      fmt_pace(lap["duration_s"], lap["distance_m"]),
        }
        for i, lap in enumerate(laps, 1)
    ]

    sport_icon  = "🚴" if sport == "cycling" else "🏃" if sport == "running" else "🏋️"
    sport_label = sub_sport.replace("_", " ").title() if sub_sport else sport.title()

    ctx = {
        # Header
        "sport_icon":   sport_icon,
        "sport_label":  sport_label,
        "filename":     Path(fit_path).stem,
        "fit_filename": Path(fit_path).name,
        "start_time":   start_time,
        "power_source": power_source,
        # Feature flags
        "is_run":            is_run,
        "has_hr":            has_hr,
        "has_power":         has_power,
        "has_lr":            has_lr,
        "has_temp":          has_temp,
        "has_core":          has_core,
        "has_skin":          has_skin,
        "has_hsi":           has_hsi,
        "has_elevation":     has_elevation,
        "has_hr_zones":      has_hr_zones,
        "has_stryd_ambient": has_stryd_ambient,
        "has_stryd_temp":    has_stryd_temp,
        "has_stryd_hum":     has_stryd_hum,
        # Overview
        "duration_fmt":  fmt_duration(duration_min * 60),
        "distance_fmt":  fmt_distance(total_dist),
        "calories":      calories if calories else None,
        "total_ascent":  int(total_ascent)  if total_ascent  else None,
        "total_descent": int(total_descent) if total_descent else None,
        "aerobic_te":    aerobic_te,
        "anaerobic_te":  anaerobic_te,
        "primary_benefit": primary_benefit,
        "has_training_effect": has_training_effect,
        # HR stats
        "avg_hr":  avg_hr,
        "max_hr":  max_hr,
        "b20_hr":  b20_hr,
        "b60_hr":  b60_hr,
        "hr_dropout_count": hr_dropout_count,
        # Power stats
        "avg_pwr": avg_pwr,
        "max_pwr": max_pwr,
        "b20_pwr": b20_pwr,
        "b60_pwr": b60_pwr,
        "ftp_est": round(b20_pwr * 0.95) if (has_power and b20_pwr) else None,
        # L/R balance
        "avg_left":     avg_left,
        "avg_right":    avg_right,
        "balance_note": balance_note,
        # Temperature
        "core_min": core_min, "core_avg": core_avg, "core_max": core_max,
        "skin_avg": skin_avg, "skin_max": skin_max,
        "hsi_avg":  hsi_avg,  "hsi_max":  hsi_max,
        "heat_stress": heat_stress,
        # Stryd ambient
        "stryd_temp_avg":   stryd_temp_avg,
        "stryd_temp_start": stryd_temp_start,
        "stryd_temp_end":   stryd_temp_end,
        "stryd_hum_avg":    stryd_hum_avg,
        "stryd_hum_start":  stryd_hum_start,
        "stryd_hum_end":    stryd_hum_end,
        # Time series
        "ts_labels":    ts_labels,
        "ts_hr":        ts_hr,
        "ts_pwr":       ts_pwr,
        "ts_left":      ts_left,
        "ts_core":      ts_core,
        "ts_skin":      ts_skin,
        "ts_hsi":       ts_hsi,
        "ts_stryd_temp": ts_stryd_temp,
        "ts_stryd_hum":  ts_stryd_hum,
        # Elevation
        "ts_elev_dist": ts_elev_dist,
        "ts_elev_alt":  ts_elev_alt,
        "elev_min":     elev_min,
        "elev_max":     elev_max,
        "elev_y_min":   max(0, int(elev_min) - 10),
        # Duration curves
        "hdc_labels": hdc_labels,
        "hdc_data":   hdc_data,
        "pdc_labels": pdc_labels,
        "pdc_data":   pdc_data,
        "hdc_y_min": max(0, int(min(p["hr_bpm"]  for p in hdc) if hdc else 0)  - 15),
        "hdc_y_max": int((max(p["hr_bpm"]  for p in hdc) if hdc else 200) + 10),
        "pdc_y_min": max(0, int(min(p["power_w"] for p in pdc) if pdc else 0)  - 20),
        "pdc_y_max": int((max(p["power_w"] for p in pdc) if pdc else 400) + 20),
        # Laps
        "laps": laps_ctx,
        # HR zones
        "zone_data":   zone_data,
        "zone_colors": zone_colors,
        "zone_names":  [z["name"]        for z in zone_data],
        "zone_secs":   [z["secs"]        for z in zone_data],
        "zone_pcts":   [z["pct"]         for z in zone_data],
        "zone_descs":  [z["description"] for z in zone_data],
    }

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    return env.get_template("report.html.j2").render(**ctx)
