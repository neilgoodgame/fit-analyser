"""Tests for fit_analyser.metrics."""

import math
import pytest
import pandas as pd
from datetime import datetime, timedelta

from fit_analyser.metrics import (
    best_average,
    compute_curve,
    compute_hdc,
    compute_pdc,
    compute_heat_stress,
)
from fit_analyser.constants import CURVE_DURATIONS


def make_series(values: list[float], freq_s: int = 1) -> pd.Series:
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    index = [t0 + timedelta(seconds=i * freq_s) for i in range(len(values))]
    return pd.Series(values, index=pd.DatetimeIndex(index))


def make_df_with_hsi(hsi_values: list[float]) -> pd.DataFrame:
    t0 = datetime(2024, 1, 1, 8, 0, 0)
    timestamps = [t0 + timedelta(seconds=i) for i in range(len(hsi_values))]
    return pd.DataFrame({"timestamp": timestamps, "heat_strain_index": hsi_values})


class TestBestAverage:
    def test_constant_series(self):
        s = make_series([150.0] * 3600)
        result = best_average(s, 60)
        assert abs(result - 150.0) < 0.5

    def test_peak_window_detected(self):
        values = [130.0] * 3600 + [160.0] * 1800
        s = make_series(values)
        result = best_average(s, 30)
        assert result >= 158.0

    def test_activity_shorter_than_window(self):
        s = make_series([150.0] * 600)
        result = best_average(s, 20)
        assert math.isnan(result)

    def test_exactly_at_window_length(self):
        s = make_series([155.0] * 1200)
        result = best_average(s, 20)
        assert not math.isnan(result)
        assert abs(result - 155.0) < 0.5

    def test_handles_gaps_with_ffill(self):
        values = [150.0] * 300 + [float("nan")] * 3 + [150.0] * 300
        s = make_series(values)
        result = best_average(s, 5)
        assert not math.isnan(result)


class TestComputeCurve:
    def test_returns_list_of_dicts(self):
        s = make_series([250.0] * 7200)
        result = compute_curve(s, "power_w")
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_result_has_correct_keys(self):
        s = make_series([250.0] * 7200)
        result = compute_curve(s, "power_w")
        for r in result:
            assert "duration_s" in r
            assert "power_w" in r

    def test_durations_are_subset_of_constants(self):
        s = make_series([250.0] * 7200)
        result = compute_curve(s, "power_w")
        durations = [r["duration_s"] for r in result]
        assert all(d in CURVE_DURATIONS for d in durations)

    def test_short_activity_fewer_points(self):
        s = make_series([250.0] * 300)
        result = compute_curve(s, "power_w")
        max_dur = max(r["duration_s"] for r in result)
        assert max_dur <= 300

    def test_curve_is_non_increasing(self):
        s = make_series([250.0] * 7200)
        result = compute_curve(s, "power_w")
        powers = [r["power_w"] for r in result]
        for i in range(1, len(powers)):
            assert powers[i] <= powers[i - 1] + 1.0


class TestComputePdc:
    def test_cycling_returns_curve(self, cycling_df):
        result = compute_pdc(cycling_df)
        assert len(result) > 0

    def test_cycling_values_plausible(self, cycling_df):
        result = compute_pdc(cycling_df)
        for r in result:
            assert 50 < r["power_w"] < 2000

    def test_cycling_5s_gt_60min(self, cycling_df):
        result = compute_pdc(cycling_df)
        by_dur = {r["duration_s"]: r["power_w"] for r in result}
        if 5 in by_dur and 3600 in by_dur:
            assert by_dur[5] >= by_dur[3600]

    def test_marathon_cleaned_power(self, marathon_df):
        result = compute_pdc(marathon_df)
        for r in result:
            assert r["power_w"] < 1000

    def test_treadmill_has_power(self, treadmill_df):
        result = compute_pdc(treadmill_df)
        assert len(result) > 0

    def test_empty_power_returns_empty(self):
        from datetime import datetime, timedelta
        t0 = datetime(2024, 1, 1)
        df = pd.DataFrame({
            "timestamp": [t0 + timedelta(seconds=i) for i in range(100)],
            "power": [float("nan")] * 100,
        })
        result = compute_pdc(df)
        assert result == []


class TestComputeHdc:
    def test_cycling_returns_curve(self, cycling_df):
        result = compute_hdc(cycling_df)
        assert len(result) > 0

    def test_values_are_bpm(self, cycling_df):
        result = compute_hdc(cycling_df)
        for r in result:
            assert 40 < r["hr_bpm"] < 220

    def test_marathon_hr_plausible(self, marathon_df):
        result = compute_hdc(marathon_df)
        by_dur = {r["duration_s"]: r["hr_bpm"] for r in result}
        assert by_dur[5] <= 220
        assert by_dur.get(3600, 200) <= 200


class TestComputeHeatStress:
    def test_no_hsi_returns_none(self):
        df = pd.DataFrame({"timestamp": [], "heat_strain_index": []})
        assert compute_heat_stress(df) is None

    def test_all_zeros_returns_none(self):
        df = make_df_with_hsi([0.0] * 59)
        assert compute_heat_stress(df) is None

    def test_returns_dict_with_required_keys(self):
        hsi = [0.0] * 1200 + [2.0] * 3600
        df = make_df_with_hsi(hsi)
        result = compute_heat_stress(df)
        assert result is not None
        for key in ["hsi_min", "active_min", "init_min", "peak_60s", "zones"]:
            assert key in result

    def test_init_period_excluded(self):
        hsi = [0.0] * 600 + [2.0] * 3600
        df = make_df_with_hsi(hsi)
        result = compute_heat_stress(df)
        assert result is not None
        assert abs(result["init_min"] - 10.0) < 0.5

    def test_hsi_min_approximately_correct(self):
        hsi = [0.0] * 600 + [2.0] * 3600
        df = make_df_with_hsi(hsi)
        result = compute_heat_stress(df)
        assert result is not None
        assert 100 < result["hsi_min"] < 130

    def test_zones_cover_all_four(self):
        hsi = [0.0] * 600 + [2.0] * 3600
        df = make_df_with_hsi(hsi)
        result = compute_heat_stress(df)
        assert result is not None
        assert len(result["zones"]) == 4
        assert [z["zone"] for z in result["zones"]] == [1, 2, 3, 4]

    def test_cycling_returns_valid_result(self, cycling_df):
        result = compute_heat_stress(cycling_df)
        assert result is not None
        assert result["hsi_min"] > 0
        assert result["active_min"] > 30

    def test_marathon_no_heat_stress_without_hrm_pro(self, marathon_df):
        result = compute_heat_stress(marathon_df)
        assert result is None

    def test_dominant_zone_is_zone2_for_moderate_activity(self, cycling_df):
        result = compute_heat_stress(cycling_df)
        assert result is not None
        dominant = max(result["zones"], key=lambda z: z["mins"])
        assert dominant["zone"] in (2, 3)
