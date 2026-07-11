"""Integration tests -- CLI and HTML report generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


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

    def test_cycling_shows_aerobic_te(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Aerobic TE" in result.stdout

    def test_cycling_shows_anaerobic_te(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Anaerobic TE" in result.stdout

    def test_cycling_shows_normalized_power(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Normalized Power" in result.stdout

    def test_cycling_shows_tss(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Training Stress Score" in result.stdout

    def test_cycling_shows_intensity_factor(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "Intensity Factor" in result.stdout

    def test_cycling_shows_primary_benefit(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert any(
            label in result.stdout
            for label in [
                "Transitioning",
                "Base",
                "Tempo",
                "Threshold",
                "VO2 Max",
                "Anaerobic",
                "Overspeed",
                "No Benefit",
                "Minor Benefit",
                "Maintaining",
                "Improving",
                "Highly Improving",
                "Overreaching",
            ]
        )

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

    def test_report_has_normalized_power(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "Normalized Power" in content
        Path(out).unlink()

    def test_report_has_tss_and_if(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "Training Stress Score" in content
        assert "Intensity Factor" in content
        Path(out).unlink()

    def test_report_has_training_effect_section(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert "Training effect" in content
        assert "Aerobic TE" in content
        assert "Anaerobic TE" in content
        assert "Primary Benefit" in content
        Path(out).unlink()

    def test_report_training_effect_shows_valid_benefit(self, cycling_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        run_cli("--fit-file-path", cycling_fit, "--html-report", "--output", out)
        content = Path(out).read_text()
        assert any(
            label in content
            for label in [
                "Transitioning",
                "Base",
                "Tempo",
                "Threshold",
                "VO2 Max",
                "Anaerobic",
                "Overspeed",
                "No Benefit",
                "Minor Benefit",
                "Maintaining",
                "Improving",
                "Highly Improving",
                "Overreaching",
            ]
        )
        Path(out).unlink()


class TestMultisportConsole:
    def test_runs_successfully(self, multisport_fit):
        result = run_cli("--fit-file-path", multisport_fit)
        assert result.returncode == 0

    def test_detects_both_sports(self, multisport_fit):
        result = run_cli("--fit-file-path", multisport_fit)
        assert "multisport activity" in result.stdout.lower()
        assert "Segment: Running" in result.stdout
        assert "Segment: Indoor Cycling" in result.stdout

    def test_transition_segment_excluded(self, multisport_fit):
        result = run_cli("--fit-file-path", multisport_fit)
        assert "Segment: Transition" not in result.stdout
        assert "Transition time" in result.stdout

    def test_combined_summary_has_te_tss_hr(self, multisport_fit):
        result = run_cli("--fit-file-path", multisport_fit)
        assert "Aerobic TE (final)" in result.stdout
        assert "Combined TSS" in result.stdout
        assert "Avg HR (whole activity)" in result.stdout

    def test_each_segment_has_own_laps_and_curves(self, multisport_fit):
        result = run_cli("--fit-file-path", multisport_fit)
        assert result.stdout.count("Heart Rate Duration Curve:") == 2
        assert result.stdout.count("Power Duration Curve") == 2

    def test_single_sport_file_not_treated_as_multisport(self, cycling_fit):
        result = run_cli("--fit-file-path", cycling_fit)
        assert "multisport" not in result.stdout.lower()


class TestMultisportHtmlReport:
    def test_generates_combined_report(self, multisport_fit):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            out = f.name
        result = run_cli("--fit-file-path", multisport_fit, "--html-report", "--output", out)
        assert result.returncode == 0
        content = Path(out).read_text()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        assert "Multisport Activity" in content
        assert content.count("<iframe") == 2
        Path(out).unlink()

    def test_no_split_files_written_by_default(self, multisport_fit):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = str(Path(tmp_dir) / "report.html")
            run_cli("--fit-file-path", multisport_fit, "--html-report", "--output", out)
            written = list(Path(tmp_dir).iterdir())
            assert written == [Path(out)]

    def test_split_reports_writes_standalone_files(self, multisport_fit):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = str(Path(tmp_dir) / "report.html")
            result = run_cli(
                "--fit-file-path",
                multisport_fit,
                "--html-report",
                "--split-reports",
                "--output",
                out,
            )
            assert result.returncode == 0
            written = sorted(p.name for p in Path(tmp_dir).iterdir())
            assert "report.html" in written
            assert any("running" in name for name in written)
            assert any("cycling" in name for name in written)
            assert len(written) == 3

    def test_split_report_is_standalone_single_sport_report(self, multisport_fit):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = str(Path(tmp_dir) / "report.html")
            run_cli(
                "--fit-file-path",
                multisport_fit,
                "--html-report",
                "--split-reports",
                "--output",
                out,
            )
            running_file = next(p for p in Path(tmp_dir).iterdir() if "running" in p.name)
            content = running_file.read_text()
            assert "<!DOCTYPE html>" in content
            assert "hrTrace" in content
            assert "<iframe" not in content
