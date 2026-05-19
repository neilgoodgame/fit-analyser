"""Tests for fit_analyser.generate_hr_zones."""

import json
import subprocess
import sys

import pytest

from fit_analyser.generate_hr_zones import (
    calc_coggan_zones,
    calc_karvonen_zones,
    calc_max_hr_zones,
)

RHR = 47
LTHR = 156
MHR = 169

CALC_FNS = [calc_max_hr_zones, calc_karvonen_zones, calc_coggan_zones]


def _zones(fn):
    return fn(RHR, LTHR, MHR)


# ---------------------------------------------------------------------------
# Shared structural properties for all methods
# ---------------------------------------------------------------------------


class TestZoneStructure:
    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_returns_five_zones(self, fn):
        assert list(_zones(fn).keys()) == ["Z1", "Z2", "Z3", "Z4", "Z5"]

    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_each_zone_has_required_keys(self, fn):
        for zone in _zones(fn).values():
            assert {"min", "max", "description"} <= zone.keys()

    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_min_less_than_max_in_every_zone(self, fn):
        for zone in _zones(fn).values():
            assert zone["min"] < zone["max"]

    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_last_zone_max_equals_max_hr(self, fn):
        assert _zones(fn)["Z5"]["max"] == MHR

    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_zones_are_contiguous(self, fn):
        zones = list(_zones(fn).values())
        for i in range(len(zones) - 1):
            assert zones[i]["max"] + 1 == zones[i + 1]["min"], (
                f"Gap between zone {i + 1} and {i + 2}: {zones[i]['max']} -> {zones[i + 1]['min']}"
            )

    @pytest.mark.parametrize("fn", CALC_FNS)
    def test_description_contains_zone_label(self, fn):
        labels = ["Active Recovery", "Endurance", "Tempo", "Threshold", "VO2max"]
        for zone, label in zip(_zones(fn).values(), labels):
            assert label in zone["description"]


# ---------------------------------------------------------------------------
# Coggan-specific values (anchored to LTHR=156, MHR=169)
# ---------------------------------------------------------------------------


class TestCogganZones:
    def test_z1_starts_at_zero(self):
        assert calc_coggan_zones(RHR, LTHR, MHR)["Z1"]["min"] == 0

    def test_z1_max(self):
        # 68% of LTHR 156 = 106.08 -> round = 106, -1 = 105
        assert calc_coggan_zones(RHR, LTHR, MHR)["Z1"]["max"] == 105

    def test_z2_max(self):
        # 83% of 156 = 129.48 -> round = 129, -1 = 128
        assert calc_coggan_zones(RHR, LTHR, MHR)["Z2"]["max"] == 128

    def test_z4_max(self):
        # 105% of 156 = 163.8 -> round = 164, -1 = 163
        assert calc_coggan_zones(RHR, LTHR, MHR)["Z4"]["max"] == 163

    def test_z5_capped_at_max_hr(self):
        # 106% of 156 = 165.36, but MHR=169 is the ceiling
        zones = calc_coggan_zones(RHR, LTHR, MHR)
        assert zones["Z5"]["max"] == MHR

    def test_z5_capped_when_mhr_below_coggan_upper(self):
        # When MHR is well below where Z5 would extend, it should still cap at MHR
        zones = calc_coggan_zones(40, 155, 170)
        assert zones["Z5"]["max"] == 170


# ---------------------------------------------------------------------------
# Max-HR-specific values
# ---------------------------------------------------------------------------


class TestMaxHrZones:
    def test_z1_min(self):
        # 50% of 169 = 84.5 -> round-half-to-even = 84
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z1"]["min"] == 84

    def test_z1_max(self):
        # 60% of 169 = 101.4 -> round = 101, -1 = 100
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z1"]["max"] == 100

    def test_z2_max(self):
        # 70% of 169 = 118.3 -> round = 118, -1 = 117
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z2"]["max"] == 117

    def test_z3_max(self):
        # 80% of 169 = 135.2 -> round = 135, -1 = 134
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z3"]["max"] == 134

    def test_z4_max(self):
        # 90% of 169 = 152.1 -> round = 152, -1 = 151
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z4"]["max"] == 151

    def test_z5_min(self):
        assert calc_max_hr_zones(RHR, LTHR, MHR)["Z5"]["min"] == 152

    def test_resting_hr_not_used(self):
        # max-hr method is independent of resting HR
        assert calc_max_hr_zones(40, LTHR, MHR) == calc_max_hr_zones(60, LTHR, MHR)

    def test_lthr_not_used(self):
        # max-hr method is independent of LTHR
        assert calc_max_hr_zones(RHR, 140, MHR) == calc_max_hr_zones(RHR, 170, MHR)


# ---------------------------------------------------------------------------
# Karvonen-specific values
# ---------------------------------------------------------------------------


class TestKarvononZones:
    def test_z1_min(self):
        # HRR=122; 50% of HRR = 61; 47+61 = 108
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z1"]["min"] == 108

    def test_z1_max(self):
        # 60% of 122 = 73.2; 47+73.2 = 120.2 -> round = 120, -1 = 119
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z1"]["max"] == 119

    def test_z2_max(self):
        # 70% of 122 = 85.4; 47+85.4 = 132.4 -> round = 132, -1 = 131
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z2"]["max"] == 131

    def test_z3_max(self):
        # 80% of 122 = 97.6; 47+97.6 = 144.6 -> round = 145, -1 = 144
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z3"]["max"] == 144

    def test_z4_max(self):
        # 90% of 122 = 109.8; 47+109.8 = 156.8 -> round = 157, -1 = 156
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z4"]["max"] == 156

    def test_z5_min(self):
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z5"]["min"] == 157

    def test_z5_max_equals_max_hr(self):
        assert calc_karvonen_zones(RHR, LTHR, MHR)["Z5"]["max"] == MHR

    def test_hrr_affects_zone_width(self):
        # Higher resting HR (smaller HRR) should compress zone widths
        narrow = calc_karvonen_zones(70, LTHR, MHR)
        wide = calc_karvonen_zones(30, LTHR, MHR)
        narrow_z1_width = narrow["Z1"]["max"] - narrow["Z1"]["min"]
        wide_z1_width = wide["Z1"]["max"] - wide["Z1"]["min"]
        assert wide_z1_width > narrow_z1_width

    def test_lthr_not_used(self):
        # karvonen method is independent of LTHR
        assert calc_karvonen_zones(RHR, 140, MHR) == calc_karvonen_zones(RHR, 170, MHR)


# ---------------------------------------------------------------------------
# CLI: validation and output
# ---------------------------------------------------------------------------


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "fit_analyser.generate_hr_zones", *args],
        capture_output=True,
        text=True,
    )


class TestCli:
    def test_preview_prints_zones(self):
        result = run_cli(
            "--resting-hr",
            "47",
            "--lthr",
            "156",
            "--max-hr",
            "169",
            "--method",
            "coggan",
            "--preview",
        )
        assert result.returncode == 0
        assert "Z1" in result.stdout
        assert "Z5" in result.stdout

    def test_resting_hr_gte_lthr_exits_nonzero(self):
        result = run_cli(
            "--resting-hr",
            "156",
            "--lthr",
            "156",
            "--max-hr",
            "169",
            "--method",
            "coggan",
            "--preview",
        )
        assert result.returncode != 0
        assert "resting HR" in result.stderr

    def test_lthr_gte_max_hr_exits_nonzero(self):
        result = run_cli(
            "--resting-hr",
            "47",
            "--lthr",
            "169",
            "--max-hr",
            "169",
            "--method",
            "coggan",
            "--preview",
        )
        assert result.returncode != 0
        assert "LTHR" in result.stderr

    def test_output_writes_valid_json(self, tmp_path):
        out = tmp_path / "zones.json"
        result = run_cli(
            "--resting-hr",
            "47",
            "--lthr",
            "156",
            "--max-hr",
            "169",
            "--method",
            "coggan",
            "--output",
            str(out),
        )
        assert result.returncode == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert set(data.keys()) == {"Z1", "Z2", "Z3", "Z4", "Z5"}

    def test_default_output_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = run_cli(
            "--resting-hr", "47", "--lthr", "156", "--max-hr", "169", "--method", "max-hr"
        )
        assert result.returncode == 0
        assert (tmp_path / "max-hr_zones.json").exists()
