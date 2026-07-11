"""Unit tests for report.py — _ts helper and build_html_report."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from fit_analyser.metrics import compute_hdc, compute_pdc
from fit_analyser.parser import parse_laps
from fit_analyser.report import _ts, build_html_report, build_multisport_html_report

_FIT_DUMMY = "activity.fit"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 1300,
    *,
    has_power: bool = True,
    has_hr: bool = True,
    has_lr: bool = False,
    has_core: bool = False,
    has_hsi: bool = False,
    has_stryd: bool = False,
    has_elevation: bool = False,
    avg_left: float = 50.0,
) -> pd.DataFrame:
    t0 = pd.Timestamp("2025-01-01 08:00:00")
    timestamps = [t0 + datetime.timedelta(seconds=i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "power": pd.Series([200.0] * n if has_power else [np.nan] * n),
            "heart_rate": pd.Series([150.0] * n if has_hr else [np.nan] * n),
            "left_pct": pd.Series([avg_left] * n if has_lr else [np.nan] * n),
            "core_temperature": pd.Series([37.5] * n if has_core else [np.nan] * n),
            "skin_temperature": pd.Series([np.nan] * n),
            "heat_strain_index": pd.Series([2.0] * n if has_hsi else [np.nan] * n),
            "stryd_temp": pd.Series([20.0] * n if has_stryd else [np.nan] * n),
            "stryd_humidity": pd.Series([60.0] * n if has_stryd else [np.nan] * n),
            "altitude": pd.Series(
                [100.0 + i * 0.1 for i in range(n)] if has_elevation else [np.nan] * n
            ),
            "distance": pd.Series([i * 3.0 for i in range(n)] if has_elevation else [np.nan] * n),
        }
    )


def _make_meta(sport: str = "cycling") -> dict:
    return {
        "sport": sport,
        "sub_sport": "",
        "total_distance": 20000,
        "total_calories": 500,
        "start_time": "2025-01-01 08:00:00",
        "total_ascent": None,
        "total_descent": None,
        "total_training_effect": 3.5,
        "total_anaerobic_training_effect": 1.0,
        "primary_benefit": "Improving",
        "training_stress_score": 80.0,
        "intensity_factor": 0.75,
        "normalized_power": 210,
    }


def _laps() -> list[dict]:
    return [{"duration_s": 120, "distance_m": 500, "avg_hr": 150, "avg_power": 200}]


def _hdc() -> list[dict]:
    return [{"duration_s": 60, "hr_bpm": 155}, {"duration_s": 120, "hr_bpm": 150}]


def _pdc() -> list[dict]:
    return [{"duration_s": 60, "power_w": 250}, {"duration_s": 120, "power_w": 220}]


def _render(**kwargs) -> str:
    """Render a report with defaults overridable via kwargs."""
    df = kwargs.pop("df", _make_df())
    laps = kwargs.pop("laps", _laps())
    meta = kwargs.pop("meta", _make_meta())
    hdc = kwargs.pop("hdc", _hdc())
    pdc = kwargs.pop("pdc", _pdc())
    return build_html_report(_FIT_DUMMY, df, laps, meta, hdc, pdc, **kwargs)


# ---------------------------------------------------------------------------
# _ts
# ---------------------------------------------------------------------------


class TestTs:
    def test_rounds_to_one_decimal_by_default(self):
        assert _ts(pd.Series([1.234, 2.567])) == [1.2, 2.6]

    def test_nan_becomes_none(self):
        result = _ts(pd.Series([1.0, float("nan")]))
        assert result[1] is None

    def test_custom_decimals(self):
        assert _ts(pd.Series([1.2345]), decimals=2) == [1.23]

    def test_empty_series(self):
        assert _ts(pd.Series([], dtype=float)) == []


# ---------------------------------------------------------------------------
# build_html_report — structure
# ---------------------------------------------------------------------------


class TestHtmlStructure:
    def test_returns_string(self):
        assert isinstance(_render(), str)

    def test_valid_html_skeleton(self):
        html = _render()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_filename_in_output(self):
        assert "activity" in _render()


# ---------------------------------------------------------------------------
# build_html_report — sport icons and labels
# ---------------------------------------------------------------------------


class TestSportIcon:
    def test_cycling_icon(self):
        assert "🚴" in _render(meta=_make_meta("cycling"))

    def test_running_icon(self):
        assert "🏃" in _render(meta=_make_meta("running"))

    def test_unknown_sport_icon(self):
        assert "🏋️" in _render(meta=_make_meta("swimming"))


# ---------------------------------------------------------------------------
# build_html_report — feature sections
# ---------------------------------------------------------------------------


class TestFeatureSections:
    def test_hr_section_present_with_hr(self):
        assert "hrTrace" in _render(df=_make_df(has_hr=True))

    def test_hr_section_absent_without_hr(self):
        assert "hrTrace" not in _render(df=_make_df(has_hr=False))

    def test_power_section_present_with_power(self):
        assert "pwrTrace" in _render(df=_make_df(has_power=True))

    def test_power_section_absent_without_power(self):
        assert "pwrTrace" not in _render(df=_make_df(has_power=False))

    def test_elevation_section_present(self):
        assert "elevChart" in _render(df=_make_df(has_elevation=True))

    def test_elevation_section_absent(self):
        assert "elevChart" not in _render(df=_make_df(has_elevation=False))

    def test_stryd_ambient_present(self):
        assert "strydAmbientTrace" in _render(df=_make_df(has_stryd=True))

    def test_stryd_ambient_absent(self):
        assert "strydAmbientTrace" not in _render(df=_make_df(has_stryd=False))

    def test_lr_section_present(self):
        assert "lrTrace" in _render(df=_make_df(has_lr=True))

    def test_lr_section_absent(self):
        assert "lrTrace" not in _render(df=_make_df(has_lr=False))

    def test_ftp_estimate_shown_with_power(self):
        assert "FTP" in _render(df=_make_df(has_power=True))

    def test_ftp_estimate_absent_without_power(self):
        assert "FTP" not in _render(df=_make_df(has_power=False))


# ---------------------------------------------------------------------------
# build_html_report — balance note
# ---------------------------------------------------------------------------


class TestBalanceNote:
    def test_balanced(self):
        assert "balanced" in _render(df=_make_df(has_lr=True, avg_left=50.0))

    def test_left_dominant(self):
        assert "left-dominant" in _render(df=_make_df(has_lr=True, avg_left=55.0))

    def test_right_dominant(self):
        assert "right-dominant" in _render(df=_make_df(has_lr=True, avg_left=45.0))


# ---------------------------------------------------------------------------
# build_html_report — HR zones
# ---------------------------------------------------------------------------


_ZONES = [
    {"name": "Z1", "min": 100, "max": 135, "description": "Recovery — easy"},
    {"name": "Z2", "min": 136, "max": 155, "description": "Aerobic — base"},
]


class TestHrZones:
    def test_zone_section_present_with_hr(self):
        html = _render(df=_make_df(has_hr=True), hr_zones=_ZONES)
        assert "zoneBarChart" in html
        assert "Z1" in html
        assert "Z2" in html

    def test_zone_section_absent_without_hr(self):
        html = _render(df=_make_df(has_hr=False), hr_zones=_ZONES)
        assert "zoneBarChart" not in html

    def test_zone_section_absent_when_zones_none(self):
        html = _render(df=_make_df(has_hr=True), hr_zones=None)
        assert "zoneBarChart" not in html

    def test_zone_description_stripped_at_dash(self):
        html = _render(df=_make_df(has_hr=True), hr_zones=_ZONES)
        assert "Recovery" in html
        assert "easy" not in html


# ---------------------------------------------------------------------------
# build_html_report — real FIT data
# ---------------------------------------------------------------------------


class TestRealData:
    def test_cycling_renders(self, cycling_fit, cycling_df, cycling_meta):
        laps = parse_laps(cycling_fit)
        html = build_html_report(
            cycling_fit,
            cycling_df,
            laps,
            cycling_meta,
            compute_hdc(cycling_df),
            compute_pdc(cycling_df),
        )
        assert "<!DOCTYPE html>" in html
        assert len(html) > 5000

    def test_cycling_has_power_and_hr(self, cycling_fit, cycling_df, cycling_meta):
        laps = parse_laps(cycling_fit)
        html = build_html_report(
            cycling_fit,
            cycling_df,
            laps,
            cycling_meta,
            compute_hdc(cycling_df),
            compute_pdc(cycling_df),
        )
        assert "pwrTrace" in html
        assert "hrTrace" in html

    def test_treadmill_renders(self, treadmill_fit, treadmill_df, treadmill_meta):
        laps = parse_laps(treadmill_fit)
        html = build_html_report(
            treadmill_fit,
            treadmill_df,
            laps,
            treadmill_meta,
            compute_hdc(treadmill_df),
            compute_pdc(treadmill_df),
        )
        assert "<!DOCTYPE html>" in html

    def test_marathon_has_elevation(self, marathon_fit, marathon_df, marathon_meta):
        laps = parse_laps(marathon_fit)
        html = build_html_report(
            marathon_fit,
            marathon_df,
            laps,
            marathon_meta,
            compute_hdc(marathon_df),
            compute_pdc(marathon_df),
        )
        assert "elevChart" in html

    def test_cycling_with_hr_zones(self, cycling_fit, cycling_df, cycling_meta):
        laps = parse_laps(cycling_fit)
        zones = [
            {"name": "Z1", "min": 100, "max": 135, "description": "Recovery — easy"},
            {"name": "Z2", "min": 136, "max": 155, "description": "Aerobic — base"},
            {"name": "Z3", "min": 156, "max": 175, "description": "Tempo — hard"},
        ]
        html = build_html_report(
            cycling_fit,
            cycling_df,
            laps,
            cycling_meta,
            compute_hdc(cycling_df),
            compute_pdc(cycling_df),
            hr_zones=zones,
        )
        assert "zoneBarChart" in html
        assert "Z1" in html


def _make_combined_meta() -> dict:
    return {
        "start_time": "2025-01-01 08:00:00",
        "total_elapsed_time": 7200,
        "total_distance": 40000,
        "total_calories": 1200,
        "total_ascent": 100,
        "total_descent": 100,
        "total_training_effect": 3.8,
        "total_anaerobic_training_effect": 0.5,
        "primary_benefit": "Base",
        "training_stress_score": 60.0,
        "avg_hr": 130.5,
        "max_hr": 165.0,
        "b20_hr": 145.2,
        "b60_hr": 138.0,
    }


class TestBuildMultisportHtmlReport:
    def _segment_reports(self):
        run_meta = _make_meta(sport="running")
        run_meta["sub_sport"] = "generic"
        bike_meta = _make_meta(sport="cycling")
        bike_meta["sub_sport"] = "indoor_cycling"
        run_df = _make_df(has_lr=False)
        bike_df = _make_df(has_lr=True)
        return [
            {
                "meta": run_meta,
                "html": build_html_report(
                    _FIT_DUMMY, run_df, [], run_meta, compute_hdc(run_df), compute_pdc(run_df)
                ),
            },
            {
                "meta": bike_meta,
                "html": build_html_report(
                    _FIT_DUMMY, bike_df, [], bike_meta, compute_hdc(bike_df), compute_pdc(bike_df)
                ),
            },
        ]

    def _build(self):
        return build_multisport_html_report(
            _FIT_DUMMY, _make_combined_meta(), self._segment_reports()
        )

    def test_has_doctype_and_closing_tag(self):
        html = self._build()
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_embeds_one_iframe_per_segment(self):
        html = self._build()
        assert html.count("<iframe") == 2

    def test_generic_sub_sport_falls_back_to_sport_name(self):
        html = self._build()
        assert "Running" in html
        assert "Indoor Cycling" in html

    def test_combined_summary_stats_present(self):
        html = self._build()
        assert "Training Stress Score" in html
        assert "Avg Heart Rate" in html
        assert "Base" in html

    def test_embedded_segment_html_is_escaped_not_raw(self):
        html = self._build()
        # The two segment reports' own <!DOCTYPE html> tags must be escaped inside
        # srcdoc="...", leaving exactly one raw top-level <!DOCTYPE html>.
        assert html.count("<!DOCTYPE html>") == 1
        assert html.count("&lt;!DOCTYPE html&gt;") == 2

    def test_no_segments_still_renders(self):
        html = build_multisport_html_report(_FIT_DUMMY, _make_combined_meta(), [])
        assert "<!DOCTYPE html>" in html
        assert "<iframe" not in html
