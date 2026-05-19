"""Unit tests for cli.py helper functions and uncovered CLI paths."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fit_analyser.cli import (
    _resolve_laps,
    _resolve_power_series,
    load_hr_zones,
    main,
    print_zone_distribution,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fit_analyser.cli", *args],
        capture_output=True,
        text=True,
    )


def _write_zones(tmp_path: Path, data: dict) -> str:
    p = tmp_path / "zones.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


_VALID_ZONES = {
    "Z1": {"min": 100, "max": 135, "description": "Recovery — easy"},
    "Z2": {"min": 136, "max": 155, "description": "Aerobic — base"},
    "Z3": {"min": 156, "max": 175, "description": "Tempo — hard"},
}


# ---------------------------------------------------------------------------
# load_hr_zones — happy path
# ---------------------------------------------------------------------------


class TestLoadHrZonesValid:
    def test_returns_list(self, tmp_path):
        zones = load_hr_zones(_write_zones(tmp_path, _VALID_ZONES))
        assert isinstance(zones, list)
        assert len(zones) == 3

    def test_sorted_by_min(self, tmp_path):
        shuffled = {"Z3": _VALID_ZONES["Z3"], "Z1": _VALID_ZONES["Z1"], "Z2": _VALID_ZONES["Z2"]}
        zones = load_hr_zones(_write_zones(tmp_path, shuffled))
        assert zones[0]["name"] == "Z1"
        assert zones[1]["name"] == "Z2"
        assert zones[2]["name"] == "Z3"

    def test_zone_keys_present(self, tmp_path):
        zones = load_hr_zones(_write_zones(tmp_path, _VALID_ZONES))
        for z in zones:
            assert {"name", "min", "max", "description"} <= z.keys()

    def test_missing_description_defaults_to_empty(self, tmp_path):
        data = {"Z1": {"min": 100, "max": 135}}
        zones = load_hr_zones(_write_zones(tmp_path, data))
        assert zones[0]["description"] == ""


# ---------------------------------------------------------------------------
# load_hr_zones — error paths
# ---------------------------------------------------------------------------


class TestLoadHrZonesErrors:
    def test_missing_file_exits(self):
        with pytest.raises(SystemExit) as exc:
            load_hr_zones("/no/such/file.json")
        assert exc.value.code == 1

    def test_invalid_json_exits(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            load_hr_zones(str(p))
        assert exc.value.code == 1

    def test_zone_missing_min_exits(self, tmp_path):
        data = {"Z1": {"max": 135, "description": "Easy"}}
        with pytest.raises(SystemExit) as exc:
            load_hr_zones(_write_zones(tmp_path, data))
        assert exc.value.code == 1

    def test_zone_missing_max_exits(self, tmp_path):
        data = {"Z1": {"min": 100, "description": "Easy"}}
        with pytest.raises(SystemExit) as exc:
            load_hr_zones(_write_zones(tmp_path, data))
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# print_zone_distribution
# ---------------------------------------------------------------------------


def _make_hr_df(hr_values: list[float]) -> pd.DataFrame:
    import datetime

    t0 = pd.Timestamp("2025-01-01 08:00:00")
    return pd.DataFrame(
        {
            "timestamp": [t0 + datetime.timedelta(seconds=i) for i in range(len(hr_values))],
            "heart_rate": pd.Series(hr_values, dtype=float),
        }
    )


_ZONES_LIST = [
    {"name": "Z1", "min": 100, "max": 135, "description": "Recovery — easy"},
    {"name": "Z2", "min": 136, "max": 155, "description": "Aerobic — base"},
    {"name": "Z3", "min": 156, "max": 175, "description": "Tempo — hard"},
]


class TestPrintZoneDistribution:
    def test_prints_zone_table(self, capsys):
        df = _make_hr_df([120.0] * 60 + [145.0] * 60 + [160.0] * 60)
        print_zone_distribution(df, _ZONES_LIST)
        out = capsys.readouterr().out
        assert "Z1" in out
        assert "Z2" in out
        assert "Z3" in out

    def test_prints_avg_and_max_hr(self, capsys):
        df = _make_hr_df([150.0] * 120)
        print_zone_distribution(df, _ZONES_LIST)
        out = capsys.readouterr().out
        assert "Avg HR" in out
        assert "Max HR" in out

    def test_prints_dominant_zone(self, capsys):
        df = _make_hr_df([120.0] * 200 + [160.0] * 10)
        print_zone_distribution(df, _ZONES_LIST)
        out = capsys.readouterr().out
        assert "Dominant zone" in out
        assert "Z1" in out

    def test_no_hr_data_prints_message(self, capsys):
        df = _make_hr_df([float("nan")] * 60)
        print_zone_distribution(df, _ZONES_LIST)
        out = capsys.readouterr().out
        assert "No heart rate data" in out

    def test_description_truncated_at_dash(self, capsys):
        df = _make_hr_df([120.0] * 60)
        print_zone_distribution(df, _ZONES_LIST)
        out = capsys.readouterr().out
        assert "Recovery" in out
        assert "easy" not in out


# ---------------------------------------------------------------------------
# CLI — --hr-zones in console mode (not --html-report)
# ---------------------------------------------------------------------------


class TestCliHrZonesConsole:
    def test_hr_zones_console_shows_distribution(self, cycling_fit, tmp_path):
        zones_path = _write_zones(tmp_path, _VALID_ZONES)
        result = run_cli("--fit-file-path", cycling_fit, "--hr-zones", zones_path)
        assert result.returncode == 0
        assert "Heart Rate Zone Distribution" in result.stdout
        assert "Z1" in result.stdout

    def test_hr_zones_console_shows_dominant_zone(self, cycling_fit, tmp_path):
        zones_path = _write_zones(tmp_path, _VALID_ZONES)
        result = run_cli("--fit-file-path", cycling_fit, "--hr-zones", zones_path)
        assert "Dominant zone" in result.stdout

    def test_hr_zones_missing_file_exits_nonzero(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit, "--hr-zones", "/no/such/zones.json")
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# _resolve_power_series — accumulated power warning (lines 119-124)
# ---------------------------------------------------------------------------


def _make_console_df(*, has_hr: bool = True, has_power: bool = True, n: int = 60) -> pd.DataFrame:
    t0 = pd.Timestamp("2025-01-01 08:00:00")
    return pd.DataFrame(
        {
            "timestamp": [t0 + datetime.timedelta(seconds=i) for i in range(n)],
            "heart_rate": pd.Series([150.0] * n if has_hr else [np.nan] * n),
            "power": pd.Series([200.0] * n if has_power else [np.nan] * n),
            "accumulated_power": pd.Series([np.nan] * n),
            "distance": pd.Series([i * 3.0 for i in range(n)]),
        }
    )


class TestResolvePowerSeries:
    def test_returns_none_when_not_requested(self):
        df = _make_console_df()
        assert _resolve_power_series(df, use_accumulated=False) is None

    def test_warns_and_returns_none_when_no_accumulated_data(self, capsys):
        df = _make_console_df()  # accumulated_power is all NaN
        result = _resolve_power_series(df, use_accumulated=True)
        assert result is None
        assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# _resolve_laps — synthetic laps warning (line 142)
# ---------------------------------------------------------------------------


class TestResolveLaps:
    def test_warns_when_no_distance_data(self, capsys):
        df = _make_console_df()
        df["distance"] = np.nan  # strip distance so synthetic_laps returns []
        result = _resolve_laps("fake.fit", df, lap_distance_km=1.0)
        assert result == []
        assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# main() — no HR / no power console output (lines 267, 287)
# ---------------------------------------------------------------------------

_MINIMAL_META = {
    "sport": "cycling",
    "sub_sport": "",
    "total_distance": 10000,
    "total_calories": 300,
    "start_time": "2025-01-01 08:00:00",
    "total_ascent": None,
    "total_descent": None,
    "total_training_effect": None,
    "total_anaerobic_training_effect": None,
    "primary_benefit": None,
    "training_stress_score": None,
    "intensity_factor": None,
    "normalized_power": None,
}


class TestMainConsoleEdgeCases:
    def test_no_hr_prints_message(self, capsys):
        df = _make_console_df(has_hr=False)
        with (
            patch("sys.argv", ["cli", "--fit-file-path", "fake.fit"]),
            patch("fit_analyser.cli.get_session_meta", return_value=_MINIMAL_META),
            patch("fit_analyser.cli.parse_fit_to_dataframe", return_value=df),
            patch("fit_analyser.cli.parse_laps", return_value=[]),
            patch("fit_analyser.cli.compute_hdc", return_value=[]),
            patch("fit_analyser.cli.compute_pdc", return_value=[]),
        ):
            main()
        assert "No heart rate data" in capsys.readouterr().out

    def test_no_power_prints_message(self, capsys):
        df = _make_console_df(has_power=False)
        with (
            patch("sys.argv", ["cli", "--fit-file-path", "fake.fit"]),
            patch("fit_analyser.cli.get_session_meta", return_value=_MINIMAL_META),
            patch("fit_analyser.cli.parse_fit_to_dataframe", return_value=df),
            patch("fit_analyser.cli.parse_laps", return_value=[]),
            patch("fit_analyser.cli.compute_hdc", return_value=[]),
            patch("fit_analyser.cli.compute_pdc", return_value=[]),
        ):
            main()
        assert "No power data found" in capsys.readouterr().out
