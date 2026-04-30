"""Integration tests -- CLI and HTML report generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fit_analyser.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCliConsoleOutput:
    def test_cycling_runs_successfully(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert result.returncode == 0

    def test_cycling_shows_sport(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "cycling" in result.stdout.lower()

    def test_cycling_shows_hr(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Heart Rate" in result.stdout
        assert "bpm" in result.stdout

    def test_cycling_shows_power(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Power" in result.stdout
        assert " W" in result.stdout

    def test_cycling_shows_laps(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Laps" in result.stdout
        assert "8 total" in result.stdout

    def test_running_shows_pace(self, treadmill_fit):
        result = run_cli("--fit-file-path", treadmill_fit)
        assert "/km" in result.stdout

    def test_marathon_shows_5_laps(self, marathon_fit):
        result = run_cli("--fit-file-path", marathon_fit)
        assert "5 total" in result.stdout

    def test_missing_file_exits_nonzero(self):
        result = run_cli("--fit-file-path", "/no/such/file.fit")
        assert result.returncode != 0


class TestCliJsonOutput:
    def test_pdc_json_is_valid(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit, "--pdc-json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list) and len(data) > 0

    def test_pdc_json_has_required_keys(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit, "--pdc-json")
        data = json.loads(result.stdout)
        for point in data:
            assert "duration_s" in point
            assert "power_w" in point

    def test_hdc_json_is_valid(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit, "--hdc-json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list) and len(data) > 0

    def test_marathon_pdc_values_sane(self, marathon_fit):
        result = run_cli("--fit-file-path", marathon_fit, "--pdc-json")
        data = json.loads(result.stdout)
        for point in data:
            assert point["power_w"] < 1000


class TestHtmlReport:
    def test_cycling_generates_report(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        result = run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        assert result.returncode == 0
        assert Path(out).exists()
        content = Path(out).read_text()
        assert len(content) > 5000
        Path(out).unlink()

    def test_html_report_is_valid_html(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        Path(out).unlink()

    def test_cycling_report_has_power_section(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "Power" in content and "pwrTrace" in content
        Path(out).unlink()

    def test_cycling_report_has_lr_balance(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "lrTrace" in content
        Path(out).unlink()

    def test_treadmill_report_no_elevation(self, treadmill_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", treadmill_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "elevChart" not in content
        Path(out).unlink()

    def test_marathon_report_has_elevation(self, marathon_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", marathon_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "elevChart" in content
        Path(out).unlink()

    def test_treadmill_report_has_stryd_ambient(self, treadmill_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", treadmill_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "strydAmbientTrace" in content
        Path(out).unlink()

    def test_cycling_report_no_stryd_ambient(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "strydAmbientTrace" not in content
        Path(out).unlink()

    def test_reports_contain_heat_stress_section(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "Cumulative heat stress" in content and "HSI" in content
        Path(out).unlink()
