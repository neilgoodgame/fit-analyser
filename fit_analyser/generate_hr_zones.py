#!/usr/bin/env python3
"""
Generate a heart rate zones JSON file from physiological parameters.

Three calculation methods are supported:

  max-hr      Percentage of maximum heart rate (Fox 1971 / Garmin default).
  karvonen    Percentage of heart rate reserve (Karvonen method).
  coggan      Anchored to lactate threshold heart rate (Coggan method).

Usage:
    poetry run generate-hr-zones \
        --resting-hr 45 \
        --lthr 155 \
        --max-hr 176 \
        --method coggan \
        --output coggan_zones.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX_HR_ZONES = [
    (
        "Z1",
        "Active Recovery",
        "Very light effort, warm-up and cool-down. Fully aerobic, no fatigue accumulation.",
        0.50,
        0.60,
    ),
    (
        "Z2",
        "Endurance",
        "Easy aerobic base work. Conversational pace, the bulk of long run and easy ride volume.",
        0.60,
        0.70,
    ),
    (
        "Z3",
        "Tempo",
        "Comfortably hard. Marathon to half-marathon effort. Sustained aerobic work.",
        0.70,
        0.80,
    ),
    (
        "Z4",
        "Threshold",
        "Lactate threshold effort. 10km to half-marathon race pace. Sustainable for 20-60 minutes.",
        0.80,
        0.90,
    ),
    (
        "Z5",
        "VO2max",
        "Very hard. 5km race pace and above. Sustainable only for short intervals of 3-8 minutes.",
        0.90,
        1.00,
    ),
]

KARVONEN_ZONES = [
    (
        "Z1",
        "Active Recovery",
        "Very light effort, warm-up and cool-down. Fully aerobic, no fatigue accumulation.",
        0.50,
        0.60,
    ),
    (
        "Z2",
        "Endurance",
        "Easy aerobic base work. Conversational pace, the bulk of long run and easy ride volume.",
        0.60,
        0.70,
    ),
    (
        "Z3",
        "Tempo",
        "Comfortably hard. Marathon to half-marathon effort. Sustained aerobic work.",
        0.70,
        0.80,
    ),
    (
        "Z4",
        "Threshold",
        "Lactate threshold effort. 10km to half-marathon race pace. Sustainable for 20-60 minutes.",
        0.80,
        0.90,
    ),
    (
        "Z5",
        "VO2max",
        "Very hard. 5km race pace and above. Sustainable only for short intervals of 3-8 minutes.",
        0.90,
        1.00,
    ),
]

COGGAN_ZONES = [
    (
        "Z1",
        "Active Recovery",
        "Very light effort, warm-up and cool-down. Fully aerobic, no fatigue accumulation.",
        0.00,
        0.68,
    ),
    (
        "Z2",
        "Endurance",
        "Easy aerobic base work. Conversational pace, the bulk of long run and easy ride volume.",
        0.68,
        0.83,
    ),
    (
        "Z3",
        "Tempo",
        "Comfortably hard. Marathon to half-marathon effort. Sustained aerobic work.",
        0.84,
        0.94,
    ),
    (
        "Z4",
        "Threshold",
        "Lactate threshold effort. 10km to half-marathon race pace. Sustainable for 20-60 minutes.",
        0.95,
        1.05,
    ),
    (
        "Z5",
        "VO2max",
        "Very hard. 5km race pace and above. Sustainable only for short intervals of 3-8 minutes.",
        1.06,
        9.99,
    ),
]


def calc_max_hr_zones(rhr: int, lthr: int, mhr: int) -> dict:
    zones = {}
    for i, (name, label, desc, lo_pct, hi_pct) in enumerate(MAX_HR_ZONES):
        lo = round(mhr * lo_pct)
        hi = round(mhr * hi_pct) - 1 if i < len(MAX_HR_ZONES) - 1 else mhr
        zones[name] = {"min": lo, "max": hi, "description": f"{label} — {desc}"}
    return zones


def calc_karvonen_zones(rhr: int, lthr: int, mhr: int) -> dict:
    hrr = mhr - rhr
    zones = {}
    for i, (name, label, desc, lo_pct, hi_pct) in enumerate(KARVONEN_ZONES):
        lo = round(rhr + hrr * lo_pct)
        hi = round(rhr + hrr * hi_pct) - 1 if i < len(KARVONEN_ZONES) - 1 else mhr
        zones[name] = {"min": lo, "max": hi, "description": f"{label} — {desc}"}
    return zones


def calc_coggan_zones(rhr: int, lthr: int, mhr: int) -> dict:
    zones = {}
    prev_max = None
    for i, (name, label, desc, lo_pct, hi_pct) in enumerate(COGGAN_ZONES):
        lo = (prev_max + 1) if prev_max is not None else (round(lthr * lo_pct) if lo_pct > 0 else 0)
        hi = min(round(lthr * hi_pct), mhr) if hi_pct < 9 else mhr
        if i < len(COGGAN_ZONES) - 1:
            hi = hi - 1
        prev_max = hi
        zones[name] = {"min": lo, "max": hi, "description": f"{label} — {desc}"}
    return zones


METHODS = {
    "max-hr": (calc_max_hr_zones, "Percentage of maximum heart rate (Fox 1971 / Garmin default)"),
    "karvonen": (calc_karvonen_zones, "Percentage of heart rate reserve — Karvonen method"),
    "coggan": (calc_coggan_zones, "Anchored to lactate threshold heart rate — Coggan method"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a heart rate zones JSON file from physiological parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  max-hr     Zones as % of max HR (50/60/70/80/90/100%).
  karvonen   Zones as % of heart rate reserve (HRR = max HR - resting HR).
  coggan     Zones anchored to LTHR using Coggan boundaries (68/83/94/105%).

Example:
  generate-hr-zones --resting-hr 45 --lthr 155 --max-hr 176 --method coggan
        """,
    )
    parser.add_argument("--resting-hr", required=True, type=int, metavar="BPM")
    parser.add_argument("--lthr", required=True, type=int, metavar="BPM")
    parser.add_argument("--max-hr", required=True, type=int, metavar="BPM")
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--output", metavar="PATH")
    parser.add_argument("--preview", action="store_true", help="Print zones without writing file.")
    args = parser.parse_args()

    if args.resting_hr >= args.lthr:
        print("ERROR: resting HR must be less than LTHR.", file=sys.stderr)
        sys.exit(1)
    if args.lthr >= args.max_hr:
        print("ERROR: LTHR must be less than max HR.", file=sys.stderr)
        sys.exit(1)

    calc_fn, method_desc = METHODS[args.method]
    zones = calc_fn(args.resting_hr, args.lthr, args.max_hr)

    print(f"\nMethod   : {args.method} — {method_desc}")
    print(f"Resting  : {args.resting_hr} bpm")
    print(f"LTHR     : {args.lthr} bpm")
    print(f"Max HR   : {args.max_hr} bpm")
    print(f"HRR      : {args.max_hr - args.resting_hr} bpm\n")
    print(f"  {'Zone':<6} {'Range':<14} {'Description'}")
    print("  " + "-" * 70)
    for name, vals in zones.items():
        hi_label = str(vals["max"]) if vals["max"] < 999 else "max"
        rng = f"{vals['min']}-{hi_label} bpm"
        desc = vals["description"].split("—")[0].strip()
        print(f"  {name:<6} {rng:<14} {desc}")

    if args.preview:
        return

    out_path = args.output or f"{args.method}_zones.json"
    Path(out_path).write_text(
        json.dumps(zones, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nZones written to: {out_path}")


if __name__ == "__main__":
    main()
