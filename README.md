# fit-analyser

[![CI](https://github.com/neilgoodgame/fit-analyser/actions/workflows/ci.yml/badge.svg)](https://github.com/neilgoodgame/fit-analyser/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/neilgoodgame/fit-analyser/branch/main/graph/badge.svg)](https://codecov.io/gh/neilgoodgame/fit-analyser)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Garmin FIT file analysis tool. Parses `.fit` files exported from Garmin Connect and produces:

- Heart rate and power statistics (average, 20-min best, 60-min best)
- Per-lap summary with pace (running) or power (cycling)
- Heart rate and power duration curves
- Left/right power balance (dual-sided power meters)
- Core/skin temperature and heat strain index with cumulative heat stress
- Elevation profile (GPS outdoor activities)
- Stryd ambient temperature and humidity
- Self-contained dark-themed HTML report
- Heart rate zone distribution chart (with custom zone JSON file)
- HR zones generator script (max-hr, karvonen, coggan methods)

Handles both Garmin-native power (cycling) and Stryd running power, including automatic filtering of Stryd firmware power glitches.

---

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)

---

## Installation

```bash
git clone https://github.com/neilgoodgame/fit-analyser.git
cd fit-analyser
poetry install
```

---

## Usage

```bash
# Console summary
poetry run fit-analyser --fit-file-path activity.fit

# Generate HTML report
poetry run fit-analyser --fit-file-path activity.fit --html-report

# With HR zone distribution
poetry run fit-analyser --fit-file-path activity.fit --hr-zones coggan_zones.json --html-report

# Use accumulated power (cleaner for ERG sessions with dropouts)
poetry run fit-analyser --fit-file-path activity.fit --accumulated-power --html-report

# Export duration curves as JSON
poetry run fit-analyser --fit-file-path activity.fit --pdc-json
poetry run fit-analyser --fit-file-path activity.fit --hdc-json

# Generate HR zones JSON file
poetry run generate-hr-zones --resting-hr 45 --lthr 155 --max-hr 176 --method coggan
poetry run generate-hr-zones --resting-hr 45 --lthr 155 --max-hr 176 --method karvonen --preview
```

---

## Pre-commit hooks

[pre-commit](https://pre-commit.com/) is configured to run `ruff` (lint + auto-fix) and `ruff-format` before every commit.

Install the hooks after cloning:

```bash
poetry run pre-commit install
```

Run manually against all files:

```bash
poetry run pre-commit run --all-files
```

---

## Running tests

```bash
# All tests with coverage
poetry run pytest

# Specific module
poetry run pytest tests/test_metrics.py -v

# Skip integration tests (faster)
poetry run pytest tests/test_parser.py tests/test_metrics.py tests/test_formatting.py -v
```

---

## Docker

```bash
# Build
docker build -t fit-analyser .

# Run (mount a directory containing your FIT files)
docker run --rm \
  -v /path/to/your/fits:/data \
  -v /path/to/output:/output \
  fit-analyser \
  --fit-file-path /data/activity.fit \
  --html-report \
  --output /output/report.html
```

---

## Project structure

```
fit_analyser/
├── __init__.py
├── cli.py               # Argument parsing and console output
├── constants.py         # CURVE_DURATIONS, HEAT_ZONES
├── formatting.py        # fmt_duration, fmt_distance, fmt_pace, dur_label
├── generate_hr_zones.py # HR zones JSON generator (max-hr / karvonen / coggan)
├── metrics.py           # best_average, compute_pdc/hdc, compute_heat_stress
├── parser.py            # parse_fit_to_dataframe, parse_laps, get_session_meta
└── report.py            # build_html_report (HTML template)

tests/
├── conftest.py          # Shared fixtures (session-scoped DataFrames)
├── fixtures/            # Real FIT files used as test data
├── test_formatting.py   # Unit tests for formatting helpers
├── test_integration.py  # End-to-end CLI and HTML report tests
├── test_metrics.py      # Unit + integration tests for metrics
└── test_parser.py       # Unit + integration tests for parser

coggan_zones.json        # Example HR zones file (Coggan LTHR-based)
```

---

## Supported devices / sensors

| Sensor | Power field | Notes |
|--------|-------------|-------|
| Garmin cycling power meter | `power` | Native FIT field |
| Stryd running footpod | `Power` | Developer field; IQR spike filter applied |
| Garmin HRM-Pro | `heart_rate`, `core_temperature`, `skin_temperature`, `heat_strain_index` | |
| Dual-sided power meter | `left_right_balance` | Decoded from Garmin uint8 encoding |
| GPS | `enhanced_altitude`, `distance` | Elevation profile for outdoor activities |
| Stryd ambient | `Stryd Temperature`, `Stryd Humidity` | Developer fields |
