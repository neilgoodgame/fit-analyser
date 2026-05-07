"""HTML report builder — renders activity data into a self-contained dark-themed HTML file."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fit_analyser.metrics import best_average, compute_heat_stress
from fit_analyser.formatting import fmt_duration, fmt_distance, fmt_pace, dur_label

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
    has_stryd_ambient = df["stryd_temp"].notna().any() or df["stryd_humidity"].notna().any()
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

    # L/R balance stats
    if has_lr:
        lr_series = df.dropna(subset=["left_pct"]).set_index("timestamp")["left_pct"]
        avg_left  = round(lr_series.mean(), 1)
        avg_right = round(100 - avg_left, 1)
    else:
        avg_left = avg_right = None

    # Temperature stats
    def temp_stats(col):
        s = df.dropna(subset=[col]).set_index("timestamp")[col]
        if s.empty: return None, None, None
        return round(s.min(), 1), round(s.mean(), 1), round(s.max(), 1)

    core_min, core_avg, core_max = temp_stats("core_temperature") if has_core else (None, None, None)
    skin_min, skin_avg, skin_max = temp_stats("skin_temperature") if has_skin else (None, None, None)
    hsi_min,  hsi_avg,  hsi_max  = temp_stats("heat_strain_index") if has_hsi  else (None, None, None)

    heat_stress = compute_heat_stress(df) if has_hsi else None

    # Stryd ambient conditions (start / mean / end)
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

    # Interpolate HR gaps for the time-series trace so dropouts show as smooth
    # transitions rather than broken lines. We do this on a resampled 1s grid,
    # interpolating gaps ≤ 60 s (sensor settling / brief contact loss) and leaving
    # longer genuine dropouts as null so they appear as breaks in the chart.
    # This only affects the visual trace — all stats still use the raw non-interpolated values.
    hr_dropout_count = int(df["heart_rate"].isna().sum())
    if has_hr and hr_dropout_count > 0:
        hr_interp = (
            df.set_index("timestamp")["heart_rate"]
            .resample("1s").mean()
            .interpolate(method="linear", limit=60, limit_direction="both")
        )
        # Re-align back onto our df index for downsampling
        df = df.copy()
        df["heart_rate_display"] = df["timestamp"].map(
            hr_interp.reset_index().set_index("timestamp")["heart_rate"]
        )
    else:
        df["heart_rate_display"] = df["heart_rate"]

    # time-series for sparklines — downsample to ~600 points max
    ts_step   = max(1, len(df) // 600)
    ts_df     = df.iloc[::ts_step].copy()
    t0        = df["timestamp"].iloc[0]
    ts_labels = [str(int((r.timestamp - t0).total_seconds())) for r in ts_df.itertuples()]
    ts_hr     = [round(v, 1) if not pd.isna(v) else "null" for v in ts_df["heart_rate_display"]]
    ts_pwr    = [round(v, 1) if not pd.isna(v) else "null" for v in ts_df["power"]]
    ts_left   = [round(v, 1) if not pd.isna(v) else "null" for v in ts_df["left_pct"]]
    ts_core   = [round(v, 2) if not pd.isna(v) else "null" for v in ts_df["core_temperature"]]
    ts_skin   = [round(v, 2) if not pd.isna(v) else "null" for v in ts_df["skin_temperature"]]
    ts_hsi    = [round(v, 2) if not pd.isna(v) else "null" for v in ts_df["heat_strain_index"]]
    ts_stryd_temp = [round(v, 1) if not pd.isna(v) else "null" for v in ts_df["stryd_temp"]]
    ts_stryd_hum  = [round(v, 1) if not pd.isna(v) else "null" for v in ts_df["stryd_humidity"]]

    # Elevation — downsample against distance (km) for x-axis rather than time
    if has_elevation:
        elev_df = df.dropna(subset=["altitude", "distance"]).copy()
        # Smooth altitude with a 30s rolling median to remove GPS noise
        elev_df["altitude_smooth"] = elev_df["altitude"].rolling(window=30, center=True, min_periods=5).median()
        elev_step = max(1, len(elev_df) // 500)
        elev_sample = elev_df.iloc[::elev_step]
        ts_elev_dist = [round(v / 1000, 3) for v in elev_sample["distance"]]
        ts_elev_alt  = [round(v, 1) if not pd.isna(v) else "null" for v in elev_sample["altitude_smooth"]]
        elev_min = round(float(elev_df["altitude_smooth"].min()), 0)
        elev_max = round(float(elev_df["altitude_smooth"].max()), 0)
    else:
        ts_elev_dist = ts_elev_alt = []
        elev_min = elev_max = 0

    hdc_labels  = json.dumps([dur_label(p["duration_s"]) for p in hdc])
    hdc_data    = json.dumps([p["hr_bpm"]  for p in hdc])
    pdc_labels  = json.dumps([dur_label(p["duration_s"]) for p in pdc])
    pdc_data    = json.dumps([p["power_w"] for p in pdc])

    sport_icon  = "🚴" if sport == "cycling" else "🏃" if sport == "running" else "🏋️"
    sport_label = sub_sport.replace("_", " ").title() if sub_sport else sport.title()
    filename    = Path(fit_path).stem

    def stat_card(label, value, unit=""):
        if value is None: return ""
        return f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}<span class="stat-unit">{unit}</span></div></div>'

    def opt_stat(label, value, unit=""):
        return stat_card(label, value, unit) if value is not None else ""

    # Lap table rows
    lap_rows = ""
    for i, lap in enumerate(laps, 1):
        hr_cell   = f"{int(lap['avg_hr'])} bpm" if lap["avg_hr"]    else "—"
        pwr_cell  = f"{int(lap['avg_power'])} W" if lap["avg_power"] else "—"
        pac_cell  = fmt_pace(lap["duration_s"], lap["distance_m"])
        dist_cell = fmt_distance(lap["distance_m"])
        row = f"""<tr>
          <td>{i}</td>
          <td>{fmt_duration(lap["duration_s"])}</td>
          <td>{dist_cell}</td>
          <td>{hr_cell}</td>
          {"<td>"+pac_cell+"</td>" if is_run else ""}
          {"<td>"+pwr_cell+"</td>" if has_power else ""}
        </tr>"""
        lap_rows += row

    lap_power_header = "<th>Avg Power</th>" if has_power else ""
    lap_pace_header  = "<th>Avg Pace</th>"  if is_run    else ""

    # FTP hint
    ftp_hint = ""
    if has_power and b20_pwr:
        ftp_est = round(b20_pwr * 0.95)
        ftp_hint = f'<p class="insight">💡 Implied FTP estimate (20-min best × 0.95): <strong>{ftp_est} W</strong></p>'

    # L/R balance bar HTML
    lr_bar_html = ""
    if has_lr and avg_left is not None:
        balance_note = "balanced" if abs(avg_left - 50) < 1 else ("left-dominant" if avg_left > 50 else "right-dominant")
        lr_bar_html = f"""
<div class="section">
  <div class="section-title">Left / right power balance</div>
  <div class="chart-wrap">
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.75rem;">
      <span style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:600;color:#4f8ef7;">L {avg_left}%</span>
      <div style="flex:1;height:14px;background:#2a3048;border-radius:7px;overflow:hidden;">
        <div style="height:100%;width:{avg_left}%;background:linear-gradient(90deg,#4f8ef7 0%,#f7694f 100%);border-radius:7px;"></div>
      </div>
      <span style="font-family:'DM Mono',monospace;font-size:1.1rem;font-weight:600;color:#f7694f;">R {avg_right}%</span>
    </div>
    <p style="font-size:0.8rem;color:var(--muted);">Session average — {balance_note}</p>
    <div class="chart-title" style="margin-top:1rem;">Balance over time</div>
    <div style="position:relative;height:140px;"><canvas id="lrTrace"></canvas></div>
  </div>
</div>"""

    # Temperature section HTML
    temp_html = ""
    if has_temp:
        temp_stats_cards = ""
        if has_core:
            temp_stats_cards += stat_card("Core start", core_min, "°C")
            temp_stats_cards += stat_card("Core avg",   core_avg, "°C")
            temp_stats_cards += stat_card("Core peak",  core_max, "°C")
        if has_skin:
            temp_stats_cards += stat_card("Skin avg",   skin_avg, "°C")
            temp_stats_cards += stat_card("Skin peak",  skin_max, "°C")
        if has_hsi:
            temp_stats_cards += stat_card("Heat strain avg", hsi_avg, "")
            temp_stats_cards += stat_card("Heat strain max", hsi_max, "")

        # Cumulative heat stress block
        hs_block = ""
        if heat_stress:
            hs = heat_stress

            # Zone rows
            zone_rows = ""
            for z in hs["zones"]:
                bar_pct = min(100, z["pct"])
                zone_rows += f"""<tr>
              <td style="padding:6px 8px;border-bottom:1px solid var(--border);">
                <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{z["color"]};margin-right:6px;vertical-align:middle;"></span>
                Zone {z["zone"]}
              </td>
              <td style="padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted);font-size:0.8rem;">{z["label"]}</td>
              <td style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);">{z["mins"]} min</td>
              <td style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted);">{z["pct"]}%</td>
              <td style="padding:6px 12px 6px 8px;border-bottom:1px solid var(--border);min-width:100px;">
                <div style="background:#2a3048;border-radius:3px;height:6px;overflow:hidden;">
                  <div style="height:100%;width:{bar_pct}%;background:{z["color"]};border-radius:3px;"></div>
                </div>
              </td>
              <td style="text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);color:var(--muted);">{z["hsi_min"]} HSI·min</td>
            </tr>"""

            # Dominant zone
            dominant = max(hs["zones"], key=lambda z: z["mins"])

            hs_block = f"""
  <div style="margin-top:1rem;padding:1rem;background:var(--surface2);border-radius:var(--radius);border:1px solid var(--border);">
    <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:0.75rem;">Cumulative heat stress</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-bottom:1rem;">
      <div class="stat" style="border:none;background:var(--surface);">
        <div class="stat-label">Total HSI·min</div>
        <div class="stat-value">{hs["hsi_min"]}</div>
      </div>
      <div class="stat" style="border:none;background:var(--surface);">
        <div class="stat-label">Peak 60s HSI</div>
        <div class="stat-value">{hs["peak_60s"]}</div>
      </div>
      <div class="stat" style="border:none;background:var(--surface);">
        <div class="stat-label">Active window</div>
        <div class="stat-value">{hs["active_min"]}<span class="stat-unit">min</span></div>
      </div>
      <div class="stat" style="border:none;background:var(--surface);border-left:3px solid {dominant["color"]} !important;">
        <div class="stat-label">Dominant zone</div>
        <div class="stat-value" style="font-size:1.1rem;color:{dominant["color"]};">Zone {dominant["zone"]}</div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">{dominant["label"]}</div>
      </div>
    </div>
    <table style="width:100%;font-size:0.82rem;font-family:'DM Mono',monospace;border-collapse:collapse;">
      <thead><tr>
        <th style="text-align:left;padding:4px 8px;color:var(--muted);font-weight:500;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);">Zone</th>
        <th style="text-align:left;padding:4px 8px;color:var(--muted);font-weight:500;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);">Description</th>
        <th style="text-align:right;padding:4px 8px;color:var(--muted);font-weight:500;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);">Time</th>
        <th style="text-align:right;padding:4px 8px;color:var(--muted);font-weight:500;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);">%</th>
        <th style="padding:4px 8px;border-bottom:1px solid var(--border);"></th>
        <th style="text-align:right;padding:4px 8px;color:var(--muted);font-weight:500;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;border-bottom:1px solid var(--border);">Contribution</th>
      </tr></thead>
      <tbody>{zone_rows}</tbody>
    </table>
    <p style="font-size:0.75rem;color:var(--muted);margin-top:0.75rem;">HSI·min = area under the heat strain index curve (excl. {hs["init_min"]} min sensor initialisation). Zones per Garmin HRM-Pro guidance: Zone 1 (0–0.9), Zone 2 (1–2.9), Zone 3 (3–6.9), Zone 4 (7–10).</p>
  </div>"""

        core_trace = f"""
    <div class="chart-title">Core &amp; skin temperature over time</div>
    <div style="position:relative;height:160px;"><canvas id="tempTrace"></canvas></div>""" if (has_core or has_skin) else ""

        hsi_trace = f"""
    <div class="chart-title" style="margin-top:1rem;">Heat strain index over time</div>
    <div style="position:relative;height:130px;"><canvas id="hsiTrace"></canvas></div>""" if has_hsi else ""

        # Stryd ambient conditions block
        stryd_ambient_html = ""
        if has_stryd_ambient:
            stryd_ambient_html = f"""
  <div class="chart-wrap" style="margin-top:1rem;">
    <div class="chart-title">Stryd ambient conditions</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-bottom:1rem;">
      {"" if stryd_temp_avg is None else f'''
      <div class="stat" style="border:none;background:var(--surface2);">
        <div class="stat-label">Ambient temp</div>
        <div class="stat-value">{stryd_temp_avg}<span class="stat-unit">°C</span></div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">{stryd_temp_start}°C → {stryd_temp_end}°C</div>
      </div>'''}
      {"" if stryd_hum_avg is None else f'''
      <div class="stat" style="border:none;background:var(--surface2);">
        <div class="stat-label">Ambient humidity</div>
        <div class="stat-value">{stryd_hum_avg}<span class="stat-unit">%</span></div>
        <div style="font-size:0.7rem;color:var(--muted);margin-top:2px;">{stryd_hum_start}% → {stryd_hum_end}%</div>
      </div>'''}
    </div>
    {"" if stryd_temp_avg is None and stryd_hum_avg is None else
    f'<div style="position:relative;height:130px;"><canvas id="strydAmbientTrace"></canvas></div>'}
  </div>"""

        temp_html = f"""
<div class="section">
  <div class="section-title">Temperature &amp; heat</div>
  <div class="stat-grid" style="margin-bottom:1rem;">{temp_stats_cards}</div>
  <div class="chart-wrap">
    {core_trace}
    {hsi_trace}
  </div>
  {hs_block}
  {stryd_ambient_html}
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{sport_label} — Activity Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0f1117;
    --surface:   #181c26;
    --surface2:  #1e2332;
    --border:    #2a3048;
    --accent:    #4f8ef7;
    --accent2:   #34c98a;
    --text:      #e8eaf0;
    --muted:     #7a84a0;
    --hr-color:  #f7694f;
    --pwr-color: #4f8ef7;
    --radius:    10px;
  }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem 1.5rem 4rem;
    max-width: 1000px;
    margin: 0 auto;
  }}

  header {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}

  .header-left {{ display: flex; align-items: center; gap: 1rem; }}

  .sport-badge {{
    font-size: 2.2rem;
    width: 60px; height: 60px;
    display: flex; align-items: center; justify-content: center;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
  }}

  h1 {{ font-size: 1.7rem; font-weight: 600; letter-spacing: -0.02em; }}
  .subtitle {{ font-size: 0.85rem; color: var(--muted); margin-top: 2px; font-family: 'DM Mono', monospace; }}

  .section {{
    margin-bottom: 2.5rem;
  }}

  .section-title {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  .stat-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
  }}

  .stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.9rem 1rem;
  }}

  .stat-label {{ font-size: 0.72rem; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 600; font-family: 'DM Mono', monospace; }}
  .stat-unit  {{ font-size: 0.75rem; font-weight: 400; color: var(--muted); margin-left: 2px; }}

  .chart-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.25rem 1rem;
    margin-bottom: 1rem;
  }}

  .chart-title {{
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}

  canvas {{ display: block; width: 100% !important; }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    font-family: 'DM Mono', monospace;
  }}

  th {{
    text-align: left;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
  }}

  td {{
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}

  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface2); }}

  .table-wrap {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}

  .insight {{
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.75rem;
    padding: 0.6rem 0.85rem;
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
  }}
  .insight strong {{ color: var(--accent); }}

  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }}

  @media (max-width: 600px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
  }}

  footer {{
    text-align: center;
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="sport-badge">{sport_icon}</div>
    <div>
      <h1>{sport_label}</h1>
      <div class="subtitle">{start_time} &nbsp;·&nbsp; {filename}</div>
    </div>
  </div>
</header>

<!-- Overview stats -->
<div class="section">
  <div class="section-title">Overview</div>
  <div class="stat-grid">
    {stat_card("Duration", fmt_duration(duration_min * 60))}
    {stat_card("Distance", fmt_distance(total_dist))}
    {stat_card("Calories", calories, "kcal") if calories else ""}
    {opt_stat("Avg Heart Rate", avg_hr, "bpm")}
    {opt_stat("Max Heart Rate", max_hr, "bpm")}
    {opt_stat("Avg Power", avg_pwr, "W")}
    {opt_stat("Max Power", max_pwr, "W")}
    {"" if not has_stryd_ambient else
      (stat_card("Ambient Temp", stryd_temp_avg, "°C") if stryd_temp_avg is not None else "") +
      (stat_card("Ambient Humidity", stryd_hum_avg, "%") if stryd_hum_avg is not None else "")}
    {stat_card("Ascent", int(total_ascent), "m") if total_ascent else ""}
    {stat_card("Descent", int(total_descent), "m") if total_descent else ""}
  </div>
</div>

<!-- Time-series charts -->
<div class="section">
  <div class="section-title">Activity trace</div>
  {"" if not has_hr else f'''
  <div class="chart-wrap">
    <div class="chart-title">Heart rate over time</div>
    <div style="position:relative;height:160px;"><canvas id="hrTrace"></canvas></div>
    {f'<p style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem;">⚠ {hr_dropout_count} missing HR samples detected (gaps ≤60s interpolated for display). Likely HRM strap contact issue at activity start.</p>' if hr_dropout_count > 0 else ""}
  </div>'''}
  {"" if not has_power else f'''
  <div class="chart-wrap">
    <div class="chart-title">Power over time</div>
    <div style="position:relative;height:160px;"><canvas id="pwrTrace"></canvas></div>
  </div>'''}
</div>

{"" if not has_elevation else f'''
<!-- Elevation profile -->
<div class="section">
  <div class="section-title">Elevation profile</div>
  <div class="chart-wrap">
    <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem;">
      <div class="chart-title" style="margin-bottom:0;">Altitude vs distance</div>
      <span style="font-size:0.75rem;color:var(--muted);">{elev_min:.0f} m – {elev_max:.0f} m</span>
    </div>
    <div style="position:relative;height:180px;"><canvas id="elevChart"></canvas></div>
  </div>
</div>'''}

<!-- HR stats -->
{"" if not has_hr else f'''
<div class="section">
  <div class="section-title">Heart rate</div>
  <div class="stat-grid">
    {stat_card("Average", avg_hr, "bpm")}
    {stat_card("Peak", max_hr, "bpm")}
    {opt_stat("20-min best", b20_hr, "bpm")}
    {opt_stat("60-min best", b60_hr, "bpm")}
  </div>
</div>'''}

<!-- Power stats -->
{"" if not has_power else f'''
<div class="section">
  <div class="section-title">Power <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:0.7rem;color:var(--muted);">({power_source})</span></div>
  <div class="stat-grid">
    {stat_card("Average", avg_pwr, "W")}
    {stat_card("Peak", max_pwr, "W")}
    {opt_stat("20-min best", b20_pwr, "W")}
    {opt_stat("60-min best", b60_pwr, "W")}
  </div>
  {ftp_hint}
</div>'''}

{lr_bar_html}

{temp_html}

<!-- Laps -->
{"" if not laps else f'''
<div class="section">
  <div class="section-title">Laps ({len(laps)})</div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>#</th><th>Duration</th><th>Distance</th><th>Avg HR</th>
        {lap_pace_header}{lap_power_header}
      </tr></thead>
      <tbody>{lap_rows}</tbody>
    </table>
  </div>
</div>'''}

<!-- Duration curves -->
<div class="section">
  <div class="section-title">Duration curves</div>
  <div class="two-col">
    {"" if not has_hr else f'''
    <div class="chart-wrap">
      <div class="chart-title">Heart rate duration curve</div>
      <div style="position:relative;height:220px;"><canvas id="hdcChart"></canvas></div>
    </div>'''}
    {"" if not has_power else f'''
    <div class="chart-wrap">
      <div class="chart-title">Power duration curve <span style="font-size:0.7rem;color:var(--muted);font-weight:400;">({power_source})</span></div>
      <div style="position:relative;height:220px;"><canvas id="pdcChart"></canvas></div>
    </div>'''}
  </div>
</div>

{"" if not has_hr_zones else f'''
<!-- HR zone distribution -->
<div class="section">
  <div class="section-title">Heart rate zones</div>
  <div class="two-col">
    <div class="chart-wrap">
      <div class="chart-title">Time in zone</div>
      <div style="position:relative;height:220px;"><canvas id="zoneBarChart"></canvas></div>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Zone distribution</div>
      <div style="position:relative;height:220px;"><canvas id="zoneDoughnut"></canvas></div>
    </div>
  </div>
  <div class="table-wrap" style="margin-top:1rem;">
    <table>
      <thead><tr>
        <th>Zone</th><th>Description</th><th>Range</th>
        <th style="text-align:right;">Time</th><th style="text-align:right;">%</th>
      </tr></thead>
      <tbody>
        {"".join(f"""<tr>
          <td><span style="display:inline-block;width:10px;height:10px;border-radius:2px;
            background:{["#34c98a","#4f8ef7","#f5c842","#f0823a","#e8394a"][min(i,4)]};
            margin-right:6px;vertical-align:middle;"></span>{z["name"]}</td>
          <td style="color:var(--muted);font-size:0.8rem;">{z["description"]}</td>
          <td style="font-family:\'DM Mono\',monospace;">{z["min"]}–{"max" if z["max"]>=999 else z["max"]} bpm</td>
          <td style="text-align:right;font-family:\'DM Mono\',monospace;">{z["time_label"]}</td>
          <td style="text-align:right;font-family:\'DM Mono\',monospace;">{z["pct"]}%</td>
        </tr>""" for i, z in enumerate(zone_data))}
      </tbody>
    </table>
  </div>
</div>'''}

<footer>Generated from {Path(fit_path).name}</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const C = Chart;
Chart.defaults.color = '#7a84a0';
Chart.defaults.borderColor = '#2a3048';
Chart.defaults.font.family = "'DM Mono', monospace";
Chart.defaults.font.size = 11;

const tsLabels = {json.dumps(ts_labels)};
const tsHR     = {json.dumps(ts_hr)};
const tsPwr    = {json.dumps(ts_pwr)};

function traceOpts(yLabel) {{
  return {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: c => ' '+Math.round(c.parsed.y)+' '+yLabel }} }} }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ grid: {{ color: '#2a3048' }}, ticks: {{ callback: v => v+' '+yLabel }} }}
    }}
  }};
}}

function curveOpts(yMin, yMax, unit) {{
  return {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: c => ' '+Math.round(c.parsed.y)+' '+unit }} }} }},
    scales: {{
      x: {{ grid: {{ color: '#2a3048' }}, ticks: {{ autoSkip: false, maxRotation: 0 }} }},
      y: {{ min: yMin, max: yMax, grid: {{ color: '#2a3048' }}, ticks: {{ callback: v => v+' '+unit }} }}
    }}
  }};
}}

{"" if not has_hr else f"""
new C(document.getElementById('hrTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: [{{ data: tsHR, borderColor: '#f7694f', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(247,105,79,0.08)', tension: 0.2 }}] }},
  options: traceOpts('bpm')
}});"""}

{"" if not has_elevation else f"""
const elevDist = {json.dumps(ts_elev_dist)};
const elevAlt  = {json.dumps(ts_elev_alt)};
new C(document.getElementById('elevChart'), {{
  type: 'line',
  data: {{ labels: elevDist, datasets: [{{
    data: elevAlt,
    borderColor: '#34c98a',
    backgroundColor: 'rgba(52,201,138,0.12)',
    borderWidth: 1.5,
    fill: true,
    tension: 0.3,
    pointRadius: 0
  }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{
        title: items => items[0].label + ' km',
        label: c => ' ' + Math.round(c.parsed.y) + ' m'
      }} }}
    }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{
        grid: {{ color: '#2a3048' }},
        ticks: {{ maxTicksLimit: 10, callback: v => elevDist[v] !== undefined ? Math.round(elevDist[v]) + ' km' : '' }}
      }},
      y: {{
        min: {max(0, int(elev_min) - 10)},
        grid: {{ color: '#2a3048' }},
        ticks: {{ callback: v => v + ' m' }}
      }}
    }}
  }}
}});"""}

{"" if not has_power else f"""
new C(document.getElementById('pwrTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: [{{ data: tsPwr, borderColor: '#4f8ef7', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(79,142,247,0.08)', tension: 0.2 }}] }},
  options: traceOpts('W')
}});"""}

{"" if not has_hr else f"""
const hdcLabels = {hdc_labels};
const hdcData   = {hdc_data};
new C(document.getElementById('hdcChart'), {{
  type: 'line',
  data: {{ labels: hdcLabels, datasets: [{{ data: hdcData, borderColor: '#f7694f', backgroundColor: 'rgba(247,105,79,0.1)', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#f7694f', fill: true, tension: 0.3 }}] }},
  options: curveOpts({max(0, int((min(p["hr_bpm"] for p in hdc) if hdc else 0) - 15))}, {int((max(p["hr_bpm"] for p in hdc) if hdc else 200) + 10)}, 'bpm')
}});"""}

{"" if not has_power else f"""
const pdcLabels = {pdc_labels};
const pdcData   = {pdc_data};
new C(document.getElementById('pdcChart'), {{
  type: 'line',
  data: {{ labels: pdcLabels, datasets: [{{ data: pdcData, borderColor: '#4f8ef7', backgroundColor: 'rgba(79,142,247,0.1)', borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#4f8ef7', fill: true, tension: 0.3 }}] }},
  options: curveOpts({max(0, int((min(p["power_w"] for p in pdc) if pdc else 0) - 20))}, {int((max(p["power_w"] for p in pdc) if pdc else 400) + 20)}, 'W')
}});"""}

{"" if not has_lr else f"""
const tsLeft = {json.dumps(ts_left)};
new C(document.getElementById('lrTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: [
    {{ label: 'Left %', data: tsLeft, borderColor: '#4f8ef7', borderWidth: 1.5, fill: false, tension: 0.2, pointRadius: 0 }},
    {{ label: '50%', data: tsLabels.map(()=>50), borderColor: '#2a3048', borderWidth: 1, borderDash: [4,4], pointRadius: 0, fill: false }}
  ]}},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: c => ' Left '+c.parsed.y.toFixed(1)+'%' }} }} }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ min: 40, max: 60, grid: {{ color: '#2a3048' }}, ticks: {{ callback: v => v+'%' }} }}
    }}
  }}
}});"""}

{"" if not (has_core or has_skin) else f"""
const tsCore = {json.dumps(ts_core)};
const tsSkin = {json.dumps(ts_skin)};
const tempDatasets = [];
if ({str(has_core).lower()}) tempDatasets.push({{ label: 'Core', data: tsCore, borderColor: '#f7694f', borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0 }});
if ({str(has_skin).lower()}) tempDatasets.push({{ label: 'Skin', data: tsSkin, borderColor: '#34c98a', borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0 }});
new C(document.getElementById('tempTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: tempDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: true, labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
               tooltip: {{ callbacks: {{ label: c => ' '+c.dataset.label+' '+c.parsed.y.toFixed(1)+'°C' }} }} }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ grid: {{ color: '#2a3048' }}, ticks: {{ callback: v => v+'°C' }} }}
    }}
  }}
}});"""}

{"" if not has_hsi else f"""
const tsHsi = {json.dumps(ts_hsi)};
new C(document.getElementById('hsiTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: [{{ data: tsHsi, borderColor: '#efb827', borderWidth: 1.5, fill: true, backgroundColor: 'rgba(239,184,39,0.08)', tension: 0.3, pointRadius: 0 }}] }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: c => ' HSI '+c.parsed.y.toFixed(1) }} }} }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ min: 0, grid: {{ color: '#2a3048' }}, ticks: {{ callback: v => v }} }}
    }}
  }}
}});"""}

{"" if not has_stryd_ambient else f"""
const tsStrydTemp = {json.dumps(ts_stryd_temp)};
const tsStrydHum  = {json.dumps(ts_stryd_hum)};
const ambientDatasets = [];
if ({str(df["stryd_temp"].notna().any()).lower()}) ambientDatasets.push({{
  label: 'Temp (°C)', data: tsStrydTemp, borderColor: '#f0823a',
  borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0,
  yAxisID: 'yTemp'
}});
if ({str(df["stryd_humidity"].notna().any()).lower()}) ambientDatasets.push({{
  label: 'Humidity (%)', data: tsStrydHum, borderColor: '#4f8ef7',
  borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0,
  yAxisID: 'yHum'
}});
new C(document.getElementById('strydAmbientTrace'), {{
  type: 'line',
  data: {{ labels: tsLabels, datasets: ambientDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: {{
      legend: {{ display: true, labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
      tooltip: {{ mode: 'index', intersect: false }}
    }},
    elements: {{ point: {{ radius: 0 }} }},
    scales: {{
      x: {{ display: false }},
      yTemp: {{
        type: 'linear', position: 'left',
        grid: {{ color: '#2a3048' }},
        ticks: {{ color: '#f0823a', callback: v => v+'°C' }}
      }},
      yHum: {{
        type: 'linear', position: 'right',
        grid: {{ drawOnChartArea: false }},
        ticks: {{ color: '#4f8ef7', callback: v => v+'%' }}
      }}
    }}
  }}
}});"""}

{"" if not has_hr_zones else f"""
const zoneColors  = {json.dumps(["#34c98a","#4f8ef7","#f5c842","#f0823a","#e8394a"][:len(zone_data)])};
const zoneLabels  = {json.dumps([z["name"] for z in zone_data])};
const zoneSecs    = {json.dumps([z["secs"] for z in zone_data])};
const zonePcts    = {json.dumps([z["pct"] for z in zone_data])};
const zoneDescs   = {json.dumps([z["description"] for z in zone_data])};

// Bar chart — time in zone
new C(document.getElementById('zoneBarChart'), {{
  type: 'bar',
  data: {{
    labels: zoneLabels,
    datasets: [{{
      data: zoneSecs.map(s => +(s/60).toFixed(1)),
      backgroundColor: zoneColors,
      borderRadius: 4,
      borderSkipped: false,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => {{
            const s = zoneSecs[ctx.dataIndex];
            const m = Math.floor(s/60), sec = s%60;
            return ` ${{m}}:${{sec.toString().padStart(2,'0')}} (${{zonePcts[ctx.dataIndex]}}%)`;
          }},
          title: ctx => zoneDescs[ctx[0].dataIndex]
        }}
      }}
    }},
    scales: {{
      x: {{ grid: {{ color: '#2a3048' }}, ticks: {{ color: '#7a84a0' }} }},
      y: {{
        grid: {{ color: '#2a3048' }},
        ticks: {{ color: '#7a84a0', callback: v => v + ' min' }}
      }}
    }}
  }}
}});

// Doughnut chart — zone proportion
new C(document.getElementById('zoneDoughnut'), {{
  type: 'doughnut',
  data: {{
    labels: zoneLabels,
    datasets: [{{
      data: zonePcts,
      backgroundColor: zoneColors,
      borderColor: '#181c26',
      borderWidth: 2,
      hoverOffset: 6,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    cutout: '62%',
    plugins: {{
      legend: {{
        display: true,
        position: 'right',
        labels: {{ boxWidth: 10, padding: 12, color: '#7a84a0', font: {{ size: 11 }} }}
      }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}}%`
        }}
      }}
    }}
  }}
}});"""}
</script>
</body>
</html>"""
    return html
