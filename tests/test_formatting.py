"""Tests for fit_analyser.formatting."""

from fit_analyser.formatting import dur_label, fmt_distance, fmt_duration, fmt_pace


class TestFmtDuration:
    def test_seconds_only(self):
        assert fmt_duration(45) == "00:45"

    def test_minutes_and_seconds(self):
        assert fmt_duration(125) == "02:05"

    def test_exact_hour(self):
        assert fmt_duration(3600) == "1:00:00"

    def test_over_an_hour(self):
        assert fmt_duration(4282) == "1:11:22"

    def test_rounding(self):
        assert fmt_duration(59.6) == "01:00"

    def test_none_returns_na(self):
        assert fmt_duration(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_duration(float("nan")) == "N/A"


class TestFmtDistance:
    def test_km(self):
        assert fmt_distance(10000) == "10.00 km"

    def test_sub_km(self):
        assert fmt_distance(500) == "0.50 km"

    def test_zero_is_indoor(self):
        assert fmt_distance(0) == "indoor"

    def test_none_returns_na(self):
        assert fmt_distance(None) == "N/A"

    def test_nan_returns_na(self):
        assert fmt_distance(float("nan")) == "N/A"

    def test_marathon_distance(self):
        result = fmt_distance(42195)
        assert result == "42.20 km"


class TestFmtPace:
    def test_5_min_per_km(self):
        assert fmt_pace(3000, 10000) == "5:00 /km"

    def test_sub_4_pace(self):
        assert fmt_pace(2340, 10000) == "3:54 /km"

    def test_zero_distance_returns_na(self):
        assert fmt_pace(600, 0) == "N/A"

    def test_none_distance_returns_na(self):
        assert fmt_pace(600, None) == "N/A"

    def test_none_duration_returns_na(self):
        assert fmt_pace(None, 10000) == "N/A"

    def test_nan_distance_returns_na(self):
        assert fmt_pace(600, float("nan")) == "N/A"


class TestDurLabel:
    def test_seconds(self):
        assert dur_label(5) == "5s"
        assert dur_label(30) == "30s"

    def test_minutes(self):
        assert dur_label(60) == "1min"
        assert dur_label(300) == "5min"
        assert dur_label(1800) == "30min"

    def test_hours(self):
        assert dur_label(3600) == "1hr"
        assert dur_label(7200) == "2hr"
