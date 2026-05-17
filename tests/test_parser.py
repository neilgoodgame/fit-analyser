"""Tests for fit_analyser.parser."""

import math
import pytest
import pandas as pd

from fit_analyser.parser import (
    decode_lr_balance,
    get_session_meta,
    parse_fit_to_dataframe,
    parse_laps,
    training_effect_label,
)


class TestTrainingEffectLabel:
    def test_none_returns_none(self):
        assert training_effect_label(None) is None

    def test_zero_is_no_benefit(self):
        assert training_effect_label(0.0) == "No Benefit"

    def test_below_one_is_no_benefit(self):
        assert training_effect_label(0.9) == "No Benefit"

    def test_one_is_minor_benefit(self):
        assert training_effect_label(1.0) == "Minor Benefit"

    def test_two_is_maintaining(self):
        assert training_effect_label(2.0) == "Maintaining"

    def test_three_is_improving(self):
        assert training_effect_label(3.0) == "Improving"

    def test_four_is_highly_improving(self):
        assert training_effect_label(4.0) == "Highly Improving"

    def test_four_point_two_is_highly_improving(self):
        assert training_effect_label(4.2) == "Highly Improving"

    def test_five_is_overreaching(self):
        assert training_effect_label(5.0) == "Overreaching"


class TestDecodeLrBalance:
    def test_none_returns_none(self):
        assert decode_lr_balance(None) is None

    def test_balanced_value(self):
        assert decode_lr_balance(180) == 52.0

    def test_left_dominant(self):
        assert decode_lr_balance(179) == 51.0

    def test_right_dominant(self):
        assert decode_lr_balance(173) == 45.0

    def test_string_input(self):
        assert decode_lr_balance("180") == 52.0

    def test_invalid_string_returns_none(self):
        assert decode_lr_balance("not_a_number") is None

    def test_zero_returns_zero(self):
        assert decode_lr_balance(0) == 0.0


class TestGetSessionMeta:
    def test_cycling_sport(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["sport"] == "cycling"

    def test_cycling_sub_sport(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["sub_sport"] == "indoor_cycling"

    def test_running_sport(self, treadmill_fit):
        meta = get_session_meta(treadmill_fit)
        assert meta["sport"] == "running"

    def test_running_sub_sport(self, treadmill_fit):
        meta = get_session_meta(treadmill_fit)
        assert meta["sub_sport"] == "treadmill"

    def test_has_total_distance(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["total_distance"] is not None
        assert meta["total_distance"] > 0

    def test_has_calories(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["total_calories"] is not None
        assert meta["total_calories"] > 0

    def test_has_start_time(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["start_time"] != ""

    def test_marathon_has_elevation(self, marathon_fit):
        meta = get_session_meta(marathon_fit)
        assert meta["total_ascent"] is not None
        assert meta["total_ascent"] > 0
        assert meta["total_descent"] is not None

    def test_indoor_no_elevation(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert "total_ascent" in meta

    def test_cycling_has_aerobic_te(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["total_training_effect"] is not None
        assert 0.0 <= meta["total_training_effect"] <= 5.0

    def test_cycling_has_anaerobic_te(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["total_anaerobic_training_effect"] is not None
        assert 0.0 <= meta["total_anaerobic_training_effect"] <= 5.0

    def test_cycling_has_primary_benefit(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["primary_benefit"] is not None
        assert meta["primary_benefit"] in {
            "No Benefit", "Minor Benefit", "Maintaining",
            "Improving", "Highly Improving", "Overreaching",
        }

    def test_primary_benefit_matches_aerobic_te(self, cycling_fit):
        meta = get_session_meta(cycling_fit)
        assert meta["primary_benefit"] == training_effect_label(meta["total_training_effect"])

    def test_missing_file_returns_empty(self):
        meta = get_session_meta("/nonexistent/path.fit")
        assert meta == {}


class TestParseFitToDataframe:
    def test_returns_dataframe(self, cycling_df):
        assert isinstance(cycling_df, pd.DataFrame)

    def test_required_columns_present(self, cycling_df):
        required = [
            "timestamp", "heart_rate", "power", "left_pct",
            "core_temperature", "skin_temperature", "heat_strain_index",
            "stryd_temp", "stryd_humidity", "altitude", "distance",
        ]
        for col in required:
            assert col in cycling_df.columns, f"Missing column: {col}"

    def test_sorted_by_timestamp(self, cycling_df):
        ts = cycling_df["timestamp"]
        assert (ts.diff().dropna() >= pd.Timedelta(0)).all()

    def test_cycling_has_hr(self, cycling_df):
        assert cycling_df["heart_rate"].notna().sum() > 0

    def test_cycling_has_power(self, cycling_df):
        assert cycling_df["power"].notna().sum() > 0

    def test_cycling_no_stryd_fields(self, cycling_df):
        assert cycling_df["stryd_temp"].isna().all()
        assert cycling_df["stryd_humidity"].isna().all()

    def test_treadmill_has_stryd_power(self, treadmill_df):
        assert treadmill_df["power"].notna().sum() > 0

    def test_treadmill_has_stryd_ambient(self, treadmill_df):
        assert treadmill_df["stryd_temp"].notna().sum() > 0
        assert treadmill_df["stryd_humidity"].notna().sum() > 0

    def test_treadmill_no_elevation(self, treadmill_df):
        assert treadmill_df["altitude"].isna().all()

    def test_marathon_has_elevation(self, marathon_df):
        assert marathon_df["altitude"].notna().sum() > 0
        assert marathon_df["distance"].notna().sum() > 0

    def test_marathon_elevation_range_plausible(self, marathon_df):
        alt = marathon_df["altitude"].dropna()
        assert alt.min() > 0
        assert alt.max() < 500
        assert alt.max() - alt.min() > 50

    def test_power_outlier_removal(self, marathon_df):
        power = marathon_df["power"].dropna()
        q75 = power.quantile(0.75)
        q25 = power.quantile(0.25)
        ceiling = q75 + 3 * (q75 - q25)
        assert (power <= ceiling).all()

    def test_cycling_no_left_right_balance_absent(self, treadmill_df):
        assert treadmill_df["left_pct"].isna().all()

    def test_cycling_has_left_right_balance(self, cycling_df):
        assert cycling_df["left_pct"].notna().sum() > 0

    def test_cycling_lr_balance_range(self, cycling_df):
        lr = cycling_df["left_pct"].dropna()
        assert (lr >= 20).all()
        assert (lr <= 80).all()

    def test_invalid_fit_raises(self):
        with pytest.raises((ValueError, Exception)):
            parse_fit_to_dataframe("/nonexistent/path.fit")

    def test_numeric_columns_are_numeric(self, cycling_df):
        numeric_cols = ["heart_rate", "power", "left_pct", "core_temperature"]
        for col in numeric_cols:
            assert pd.api.types.is_float_dtype(cycling_df[col]) or \
                   pd.api.types.is_integer_dtype(cycling_df[col])


class TestParseLaps:
    def test_cycling_lap_count(self, cycling_fit, cycling_df):
        laps = parse_laps(cycling_fit, cycling_df)
        assert len(laps) == 8

    def test_treadmill_lap_count(self, treadmill_fit, treadmill_df):
        laps = parse_laps(treadmill_fit, treadmill_df)
        assert len(laps) == 4

    def test_marathon_lap_count(self, marathon_fit, marathon_df):
        laps = parse_laps(marathon_fit, marathon_df)
        assert len(laps) == 5

    def test_lap_has_required_keys(self, cycling_fit, cycling_df):
        laps = parse_laps(cycling_fit, cycling_df)
        for lap in laps:
            assert "duration_s" in lap
            assert "distance_m" in lap
            assert "avg_hr" in lap
            assert "avg_power" in lap

    def test_lap_durations_positive(self, cycling_fit, cycling_df):
        laps = parse_laps(cycling_fit, cycling_df)
        for lap in laps:
            assert lap["duration_s"] > 0

    def test_cycling_lap_power_plausible(self, cycling_fit, cycling_df):
        laps = parse_laps(cycling_fit, cycling_df)
        for lap in laps:
            if lap["avg_power"] is not None:
                assert 50 < lap["avg_power"] < 600

    def test_marathon_lap_power_cleaned(self, marathon_fit, marathon_df):
        laps = parse_laps(marathon_fit, marathon_df)
        for lap in laps:
            if lap["avg_power"] is not None:
                assert lap["avg_power"] < 600

    def test_laps_without_df_use_fit_values(self, cycling_fit):
        laps = parse_laps(cycling_fit)
        assert len(laps) == 8
        assert all("avg_power" in lap for lap in laps)
