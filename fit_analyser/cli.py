"""Command-line interface for fit-analyser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from fit_analyser.formatting import dur_label, fmt_distance, fmt_duration, fmt_pace, sport_label
from fit_analyser.metrics import best_average, compute_hdc, compute_pdc
from fit_analyser.parser import (
    compute_combined_meta,
    derive_power_from_accumulated,
    get_all_sessions_meta,
    get_session_meta,
    is_multisport,
    parse_fit_to_dataframe,
    parse_laps,
    split_laps_by_session,
    split_records_by_session,
    split_series_by_session,
    synthetic_laps,
)
from fit_analyser.report import build_html_report, build_multisport_html_report


def load_hr_zones(zones_path: str) -> list[dict]:
    """Load HR zones from a JSON file."""
    try:
        raw = json.loads(Path(zones_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: HR zones file not found: {zones_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in {zones_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    zones = []
    for name, vals in raw.items():
        if "min" not in vals or "max" not in vals:
            print(f"ERROR: Zone '{name}' missing 'min' or 'max'", file=sys.stderr)
            sys.exit(1)
        zones.append(
            {
                "name": name,
                "min": int(vals["min"]),
                "max": int(vals["max"]),
                "description": vals.get("description", ""),
            }
        )
    return sorted(zones, key=lambda z: z["min"])


def print_zone_distribution(df: pd.DataFrame, zones: list[dict]) -> None:
    hr = df["heart_rate"].dropna()
    if hr.empty:
        print("  No heart rate data available for zone analysis.")
        return
    total_s = len(hr)
    print(f"  {'Zone':<6} {'Description':<22} {'Range':<13} {'Time':>8}  {'%':>6}  {'Bar'}")
    print("  " + "-" * 74)
    for z in zones:
        lo, hi = z["min"], z["max"]
        in_z = hr[(hr >= lo) & (hr <= hi)]
        secs = len(in_z)
        pct = secs / total_s * 100 if total_s else 0
        m, s = divmod(secs, 60)
        bar = "█" * int(pct / 2)
        hi_label = str(hi) if hi < 999 else "max"
        rng = f"{lo}-{hi_label} bpm"
        desc = (
            z["description"].split("—")[0].strip() if "—" in z["description"] else z["description"]
        )
        desc = desc[:21]
        print(f"  {z['name']:<6} {desc:<22} {rng:<13} {m:>4}:{s:02d}  {pct:>5.1f}%  {bar}")
    print()
    dominant = max(zones, key=lambda z: len(hr[(hr >= z["min"]) & (hr <= z["max"])]))
    d_desc = (
        dominant["description"].split("—")[0].strip()
        if "—" in dominant["description"]
        else dominant["description"][:40]
    )
    print(f"  Avg HR         : {hr.mean():.1f} bpm")
    print(f"  Max HR         : {hr.max():.0f} bpm")
    print(f"  Dominant zone  : {dominant['name']} ({d_desc})")


def _print_lap_table(laps: list[dict], is_run: bool, has_power: bool) -> None:
    cols = ["Lap", "Duration", "Distance", "Avg HR"]
    widths = [4, 9, 9, 7]
    if is_run:
        cols.append("Avg Pace")
        widths.append(10)
    if has_power:
        cols.append("Avg Pwr")
        widths.append(8)

    print("  " + "  ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("  " + "  ".join("-" * w for w in widths))

    for i, lap in enumerate(laps, 1):
        hr = f"{int(lap['avg_hr'])} bpm" if lap["avg_hr"] else "N/A"
        row = [
            f"{i:>{widths[0]}}",
            f"{fmt_duration(lap['duration_s']):>{widths[1]}}",
            f"{fmt_distance(lap['distance_m']):>{widths[2]}}",
            f"{hr:>{widths[3]}}",
        ]
        if is_run:
            row.append(f"{fmt_pace(lap['duration_s'], lap['distance_m']):>{widths[4]}}")
        if has_power:
            pwr_str = f"{int(lap['avg_power'])} W" if lap["avg_power"] else "N/A"
            row.append(f"{pwr_str:>{widths[-1]}}")
        print("  " + "  ".join(row))


def _resolve_power_series(df: pd.DataFrame, use_accumulated: bool) -> "pd.Series | None":
    if not use_accumulated:
        return None
    ap_series = derive_power_from_accumulated(df)
    if ap_series.empty:
        print(
            "WARNING: --accumulated-power requested but no accumulated_power "
            "data found. Falling back to instantaneous power.",
            file=sys.stderr,
        )
        return None
    return ap_series


def _resolve_laps(
    fit_path: str,
    df: pd.DataFrame,
    lap_distance_km: float | None,
) -> list[dict]:
    """
    Return laps either from the FIT file or synthetically at a fixed distance.

    If lap_distance_km is provided, synthetic_laps() is used and the
    FIT file's lap messages are ignored entirely.
    """
    if lap_distance_km is not None:
        laps = synthetic_laps(df, lap_distance_km)
        if not laps:
            print(
                f"WARNING: Could not generate synthetic laps at {lap_distance_km} km intervals "
                "(no distance data in file).",
                file=sys.stderr,
            )
        return laps
    return parse_laps(fit_path, df)


def print_activity_report(
    df: pd.DataFrame,
    laps: list[dict],
    meta: dict,
    hdc: list[dict],
    pdc: list[dict],
    ap_series: "pd.Series | None",
    power_source: str,
    args,
    lap_source: str,
    fit_path: str | None = None,
) -> None:
    """
    Print the full text report (overview, HR, power, laps, duration curves,
    HR zones) for a single activity or a single sport segment of a
    multisport activity. Pass fit_path to also print the leading
    "Parsing: <path>" line (single-sport files only — multisport segments
    share one such line printed by the caller).
    """
    sport = meta.get("sport", "unknown")
    duration_min = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 60
    hr_points = int(df["heart_rate"].notna().sum())
    power_points = (
        int(df["power"].notna().sum()) if ap_series is None else int(ap_series.notna().sum())
    )
    is_run = sport == "running"

    aerobic_te = meta.get("total_training_effect")
    anaerobic_te = meta.get("total_anaerobic_training_effect")
    primary_benefit = meta.get("primary_benefit")

    if fit_path is not None:
        print(f"Parsing: {fit_path}\n")
    print(f"Sport                   : {sport}")
    print(f"Activity duration       : {duration_min:.1f} min")
    print(f"HR data points          : {hr_points}")
    print(f"Power data points       : {power_points}  [{power_source}]")
    if aerobic_te is not None:
        print(f"Aerobic TE              : {aerobic_te:.1f}  ({primary_benefit})")
    if anaerobic_te is not None:
        print(f"Anaerobic TE            : {anaerobic_te:.1f}")
    has_stryd_temp = bool(df["stryd_temp"].notna().any()) if "stryd_temp" in df.columns else False
    has_stryd_hum = (
        bool(df["stryd_humidity"].notna().any()) if "stryd_humidity" in df.columns else False
    )
    if has_stryd_temp:
        avg_temp = df["stryd_temp"].dropna().mean()
        print(f"Avg Ambient Temp        : {avg_temp:.1f} °C")
    if has_stryd_hum:
        avg_hum = df["stryd_humidity"].dropna().mean()
        print(f"Avg Ambient Humidity    : {avg_hum:.1f} %")
    tss = meta.get("training_stress_score")
    if_val = meta.get("intensity_factor")
    np_val = meta.get("normalized_power")
    if tss is not None:
        print(f"Training Stress Score   : {tss:.1f}")
    if if_val is not None:
        print(f"Intensity Factor        : {if_val:.3f}")
    if np_val is not None:
        print(f"Normalized Power        : {np_val} W")
    print()

    print("Heart Rate:")
    if hr_points == 0:
        print("  No heart rate data.")
    else:
        s = df.dropna(subset=["heart_rate"]).set_index("timestamp")["heart_rate"]
        b20 = best_average(s, 20)
        b60 = best_average(s, 60)
        print(f"  Average               : {s.mean():.1f} bpm")
        print(
            f"  20-min best average   : {b20:.1f} bpm"
            if not pd.isna(b20)
            else "  20-min best average   : N/A"
        )
        print(
            f"  60-min best average   : {b60:.1f} bpm"
            if not pd.isna(b60)
            else "  60-min best average   : N/A"
        )
    print()

    print(f"Power  [{power_source}]:")
    if power_points == 0:
        print("  No power data found.")
    else:
        if ap_series is not None:
            s = ap_series.dropna()
        else:
            df_p = df.copy()
            df_p.loc[df_p["power"] == 0, "power"] = float("nan")
            s = df_p.dropna(subset=["power"]).set_index("timestamp")["power"]
        b20 = best_average(s, 20)
        b60 = best_average(s, 60)
        print(f"  Average               : {s.mean():.1f} W")
        print(
            f"  20-min best average   : {b20:.1f} W"
            if not pd.isna(b20)
            else "  20-min best average   : N/A"
        )
        print(
            f"  60-min best average   : {b60:.1f} W"
            if not pd.isna(b60)
            else "  60-min best average   : N/A"
        )

    if laps:
        print()
        print(f"Laps ({len(laps)} total)  [{lap_source}]:")
        _print_lap_table(laps, is_run, power_points > 0)

    if hr_points > 0:
        print()
        print("Heart Rate Duration Curve:")
        for pt in hdc:
            print(f"  {dur_label(pt['duration_s']):>6} : {pt['hr_bpm']:.0f} bpm")

    if power_points > 0:
        print()
        print(f"Power Duration Curve  [{power_source}]:")
        for pt in pdc:
            print(f"  {dur_label(pt['duration_s']):>6} : {pt['power_w']:.0f} W")

    if args.hr_zones:
        zones = load_hr_zones(args.hr_zones)
        print()
        print(f"Heart Rate Zone Distribution  [{Path(args.hr_zones).name}]:")
        print_zone_distribution(df, zones)


def _print_multisport_summary(fit_path: str, sessions: list[dict], combined: dict) -> None:
    real_sessions = [s for s in sessions if s.get("sport") != "transition"]
    transitions = [s for s in sessions if s.get("sport") == "transition"]

    print(f"Parsing: {fit_path}  [multisport activity — {len(real_sessions)} sports]\n")
    print(f"Sports                  : {', '.join(s['sport'] for s in real_sessions)}")
    print(f"Total duration          : {fmt_duration(combined['total_elapsed_time'])}")
    print(f"Total distance          : {fmt_distance(combined['total_distance'])}")
    if combined["total_calories"] is not None:
        print(f"Total calories          : {combined['total_calories']} kcal")
    if combined["total_ascent"] is not None:
        print(f"Total ascent            : {int(combined['total_ascent'])} m")
    if combined["total_descent"] is not None:
        print(f"Total descent           : {int(combined['total_descent'])} m")
    if combined["total_training_effect"] is not None:
        print(
            f"Aerobic TE (final)      : {combined['total_training_effect']:.1f}  "
            f"({combined['primary_benefit']})"
        )
    if combined["total_anaerobic_training_effect"] is not None:
        print(f"Anaerobic TE (final)    : {combined['total_anaerobic_training_effect']:.1f}")
    if combined["training_stress_score"] is not None:
        print(f"Combined TSS            : {combined['training_stress_score']:.1f}")
    if combined["avg_hr"] is not None:
        print(f"Avg HR (whole activity) : {combined['avg_hr']:.1f} bpm")
        print(f"Max HR                  : {combined['max_hr']:.0f} bpm")
    if not pd.isna(combined["b20_hr"]):
        print(f"20-min best HR          : {combined['b20_hr']:.1f} bpm")
    if not pd.isna(combined["b60_hr"]):
        print(f"60-min best HR          : {combined['b60_hr']:.1f} bpm")
    if transitions:
        total_transition_s = sum(s["total_elapsed_time"] or 0 for s in transitions)
        print(
            f"Transition time         : {fmt_duration(total_transition_s)}  "
            f"({len(transitions)} transition(s), excluded below)"
        )


def _resolve_segment_laps(
    fit_path: str,
    df: pd.DataFrame,
    df_segments: list[pd.DataFrame],
    sessions: list[dict],
    lap_distance_km: float | None,
) -> list[list[dict]]:
    if lap_distance_km is not None:
        return [synthetic_laps(seg_df, lap_distance_km) for seg_df in df_segments]
    return split_laps_by_session(parse_laps(fit_path, df), sessions)


def handle_multisport_console(
    fit_path: str,
    df: pd.DataFrame,
    sessions: list[dict],
    ap_series: "pd.Series | None",
    power_source: str,
    args,
) -> None:
    combined = compute_combined_meta(df, sessions)
    _print_multisport_summary(fit_path, sessions, combined)

    df_segments = split_records_by_session(df, sessions)
    laps_segments = _resolve_segment_laps(fit_path, df, df_segments, sessions, args.lap_distance)
    ap_segments = (
        split_series_by_session(ap_series, sessions)
        if ap_series is not None
        else [None] * len(sessions)
    )
    lap_source = (
        f"synthetic @ {args.lap_distance} km" if args.lap_distance is not None else "from FIT file"
    )

    for i, s in enumerate(sessions):
        if s.get("sport") == "transition" or df_segments[i].empty:
            continue
        seg_df = df_segments[i]
        seg_ap = ap_segments[i]
        seg_hdc = compute_hdc(seg_df)
        seg_pdc = compute_pdc(seg_df, seg_ap)
        seg_label = sport_label(s["sport"], s["sub_sport"])

        print()
        print("=" * 70)
        seg_dur = fmt_duration(s["total_elapsed_time"])
        print(f"Segment: {seg_label}   ({seg_dur}, start {s['start_time']})")
        print("=" * 70)
        print_activity_report(
            seg_df, laps_segments[i], s, seg_hdc, seg_pdc, seg_ap, power_source, args, lap_source
        )


def handle_multisport_html(
    fit_path: str,
    df: pd.DataFrame,
    sessions: list[dict],
    ap_series: "pd.Series | None",
    power_source: str,
    args,
) -> None:
    combined = compute_combined_meta(df, sessions)
    df_segments = split_records_by_session(df, sessions)
    laps_segments = _resolve_segment_laps(fit_path, df, df_segments, sessions, args.lap_distance)
    ap_segments = (
        split_series_by_session(ap_series, sessions)
        if ap_series is not None
        else [None] * len(sessions)
    )
    zones = load_hr_zones(args.hr_zones) if args.hr_zones else None

    out_path = args.output or (Path(fit_path).stem + "_report.html")
    stem = Path(out_path).stem
    parent = Path(out_path).parent

    segment_reports = []
    for i, s in enumerate(sessions):
        if s.get("sport") == "transition" or df_segments[i].empty:
            continue
        seg_df = df_segments[i]
        seg_hdc = compute_hdc(seg_df)
        seg_pdc = compute_pdc(seg_df, ap_segments[i])
        seg_html = build_html_report(
            fit_path,
            seg_df,
            laps_segments[i],
            s,
            seg_hdc,
            seg_pdc,
            power_source=power_source,
            hr_zones=zones,
        )
        segment_reports.append({"meta": s, "html": seg_html})

        if args.split_reports:
            seg_path = parent / f"{stem}_{len(segment_reports)}_{s['sport']}_report.html"
            seg_path.write_text(seg_html, encoding="utf-8")
            print(f"Segment report written to: {seg_path}")

    combined_html = build_multisport_html_report(fit_path, combined, segment_reports)
    Path(out_path).write_text(combined_html, encoding="utf-8")
    print(f"Combined report written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a Garmin FIT file and report HR and power statistics."
    )
    parser.add_argument("--fit-file-path", required=True, metavar="PATH")
    parser.add_argument(
        "--html-report", action="store_true", help="Generate a self-contained HTML report."
    )
    parser.add_argument("--output", metavar="PATH", help="Output path for HTML report.")
    parser.add_argument(
        "--pdc-json", action="store_true", help="Print power duration curve as JSON and exit."
    )
    parser.add_argument(
        "--hdc-json", action="store_true", help="Print HR duration curve as JSON and exit."
    )
    parser.add_argument(
        "--accumulated-power",
        action="store_true",
        help="Derive power from accumulated_power field.",
    )
    parser.add_argument("--hr-zones", metavar="PATH", help="Path to a JSON file defining HR zones.")
    parser.add_argument(
        "--lap-distance",
        type=float,
        metavar="KM",
        default=None,
        help=(
            "Override FIT file laps and generate synthetic laps at this fixed "
            "distance interval in kilometres (e.g. --lap-distance 5 creates "
            "one lap every 5 km). Useful for races recorded as a single lap."
        ),
    )
    parser.add_argument(
        "--split-reports",
        action="store_true",
        help=(
            "For multisport files with --html-report: also write a standalone "
            "full report per sport, in addition to the combined report."
        ),
    )
    args = parser.parse_args()

    if args.lap_distance is not None and args.lap_distance <= 0:
        print("ERROR: --lap-distance must be a positive number.", file=sys.stderr)
        sys.exit(1)

    fit_path = args.fit_file_path

    try:
        meta = get_session_meta(fit_path)
        df = parse_fit_to_dataframe(fit_path)
        sessions = get_all_sessions_meta(fit_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    ap_series = _resolve_power_series(df, args.accumulated_power)
    power_source = "accumulated_power" if ap_series is not None else "instantaneous power"

    if args.pdc_json:
        print(json.dumps(compute_pdc(df, ap_series)))
        return
    if args.hdc_json:
        print(json.dumps(compute_hdc(df)))
        return

    if is_multisport(sessions):
        if args.html_report:
            handle_multisport_html(fit_path, df, sessions, ap_series, power_source, args)
        else:
            handle_multisport_console(fit_path, df, sessions, ap_series, power_source, args)
        return

    hdc = compute_hdc(df)
    pdc = compute_pdc(df, ap_series)
    laps = _resolve_laps(fit_path, df, args.lap_distance)

    lap_source = (
        f"synthetic @ {args.lap_distance} km" if args.lap_distance is not None else "from FIT file"
    )

    if args.html_report:
        zones = load_hr_zones(args.hr_zones) if args.hr_zones else None
        html = build_html_report(
            fit_path,
            df,
            laps,
            meta,
            hdc,
            pdc,
            power_source=power_source,
            hr_zones=zones,
        )
        out_path = args.output or (Path(fit_path).stem + "_report.html")
        Path(out_path).write_text(html, encoding="utf-8")
        print(f"Report written to: {out_path}")
        return

    print_activity_report(
        df, laps, meta, hdc, pdc, ap_series, power_source, args, lap_source, fit_path=fit_path
    )


if __name__ == "__main__":
    main()
