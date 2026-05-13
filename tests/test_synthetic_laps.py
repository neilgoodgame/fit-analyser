"""Tests for synthetic_laps() and --lap-distance CLI option."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from fit_analyser.parser import synthetic_laps

FIXTURES = Path(__file__).parent / "fixtures"
MARATHON  = FIXTURES / "running_outdoor_marathon.fit"
CYCLING   = FIXTURES / "cycling_indoor.fit"
TREADMILL = FIXTURES / "running_treadmill.fit"


# ── Helpers ─────────────────────────────────────────────────────────────────────────────

def make_df(total_km: float, hr_start: int = 130, power: float = 250.0) -> pd.DataFrame:
    """Build a minimal record DataFrame with 1-second resolution."""
    n = int(total_km * 1000)
    t0 = datetime(2024, 5, 11, 8, 0, 0)
    timestamps = [t0 + timedelta(seconds=i) for i in range(n)]
    return pd.DataFrame({
        "timestamp":  timestamps,
        "distance":   [float(i) for i in range(n)],
        "heart_rate": [hr_start + i // 600 for i in range(n)],
        "power":      [power] * n,
    })


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fit_analyser.cli", *args],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )


# ── synthetic_laps unit tests ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticLaps:

    def test_exact_laps_no_remainder(self):
        """10 km at 5 km intervals → exactly 2 laps."""
        df = make_df(10.0)
        laps = synthetic_laps(df, 5.0)
        assert len(laps) == 2

    def test_partial_final_lap_included(self):
        """12 km at 5 km intervals → 2 full + 1 partial = 3 laps."""
        df = make_df(12.0)
        laps = synthetic_laps(df, 5.0)
        assert len(laps) == 3

    def test_lap_distances_correct(self):
        df = make_df(10.0)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            assert abs(lap["distance_m"] - 5000) < 10

    def test_lap_durations_positive(self):
        df = make_df(10.0)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            assert lap["duration_s"] > 0

    def test_lap_has_required_keys(self):
        df = make_df(10.0)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            for key in ["duration_s", "distance_m", "avg_hr", "avg_power", "start_time", "end_time"]:
                assert key in lap

    def test_avg_hr_computed(self):
        df = make_df(10.0, hr_start=140)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            assert lap["avg_hr"] is not None
            assert 100 < lap["avg_hr"] < 220

    def test_avg_power_computed(self):
        df = make_df(10.0, power=250.0)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            assert lap["avg_power"] is not None
            assert abs(lap["avg_power"] - 250.0) < 5

    def test_no_distance_returns_empty(self):
        df = make_df(10.0)
        df["distance"] = float("nan")
        laps = synthetic_laps(df, 5.0)
        assert laps == []

    def test_sub_km_interval(self):
        """1 km at 0.2 km intervals → 5 laps."""
        df = make_df(1.0)
        laps = synthetic_laps(df, 0.2)
        assert len(laps) == 5

    def test_interval_larger_than_activity(self):
        """5 km activity with 10 km interval → 1 lap covering full distance."""
        df = make_df(5.0)
        laps = synthetic_laps(df, 10.0)
        assert len(laps) == 1
        assert laps[0]["distance_m"] < 5100

    def test_start_end_times_ascending(self):
        df = make_df(15.0)
        laps = synthetic_laps(df, 5.0)
        for lap in laps:
            assert lap["end_time"] > lap["start_time"]

    def test_consecutive_lap_times_contiguous(self):
        """Each lap starts where the previous one ended (within 2 seconds)."""
        df = make_df(15.0)
        laps = synthetic_laps(df, 5.0)
        for i in range(1, len(laps)):
            gap = (laps[i]["start_time"] - laps[i-1]["end_time"]).total_seconds()
            assert abs(gap) <= 2

    def test_total_distance_preserved(self):
        """Sum of lap distances ≈ total activity distance."""
        df = make_df(42.195)
        laps = synthetic_laps(df, 5.0)
        total = sum(lap["distance_m"] for lap in laps)
        assert abs(total - 42195) < 100

    def test_marathon_fixture_5km_laps(self, marathon_df):
        """Marathon fixture (~42 km) → 9 laps at 5 km (8 full + 1 partial)."""
        laps = synthetic_laps(marathon_df, 5.0)
        assert len(laps) == 9

    def test_marathon_fixture_1km_laps(self, marathon_df):
        laps = synthetic_laps(marathon_df, 1.0)
        assert len(laps) == 43  # 42 full + 1 partial

    def test_marathon_fixture_10km_laps(self, marathon_df):
        laps = synthetic_laps(marathon_df, 10.0)
        assert len(laps) == 5  # 4 full + 1 partial

    def test_indoor_cycling_no_gps_still_works(self, cycling_df):
        """Indoor cycling has virtual distance — synthetic laps should still work."""
        laps = synthetic_laps(cycling_df, 5.0)
        assert len(laps) > 0

    def test_treadmill_5km_laps(self, treadmill_df):
        laps = synthetic_laps(treadmill_df, 5.0)
        assert len(laps) > 0
        for lap in laps:
            assert lap["duration_s"] > 0


# ── CLI integration tests ─────────────────────────────────────────────────────────────────────────────

class TestCliLapDistance:

    def test_lap_distance_overrides_fit_laps(self):
        """Marathon FIT has 5 laps; --lap-distance 5 should produce 9."""
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "5")
        assert result.returncode == 0
        assert "9 total" in result.stdout

    def test_lap_distance_1km_marathon(self):
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "1")
        assert result.returncode == 0
        assert "43 total" in result.stdout

    def test_lap_distance_10km_marathon(self):
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "10")
        assert result.returncode == 0
        assert "5 total" in result.stdout

    def test_lap_source_label_shown(self):
        """CLI output should indicate laps are synthetic."""
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "5")
        assert "synthetic" in result.stdout.lower()

    def test_without_lap_distance_uses_fit_laps(self):
        """Without --lap-distance the original 5 FIT laps should appear."""
        result = run_cli("--fit-file-path", str(MARATHON))
        assert result.returncode == 0
        assert "5 total" in result.stdout

    def test_invalid_lap_distance_exits_nonzero(self):
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "-1")
        assert result.returncode != 0

    def test_zero_lap_distance_exits_nonzero(self):
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "0")
        assert result.returncode != 0

    def test_lap_distance_with_cycling(self):
        result = run_cli("--fit-file-path", str(CYCLING), "--lap-distance", "5")
        assert result.returncode == 0

    def test_lap_distance_pace_shown_for_running(self):
        result = run_cli("--fit-file-path", str(MARATHON), "--lap-distance", "5")
        assert "/km" in result.stdout

    def test_lap_distance_combined_with_accumulated_power(self):
        result = run_cli(
            "--fit-file-path", str(CYCLING),
            "--lap-distance", "5",
            "--accumulated-power",
        )
        assert result.returncode == 0
