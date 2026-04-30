"""Shared constants."""

CURVE_DURATIONS = [5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600]

HEAT_ZONES = [
    {
        "zone": 1,
        "label": "No Heat Strain",
        "lo": 0.0,
        "hi": 1.0,
        "color": "#34c98a",
        "perf": "Optimal Performance",
    },
    {
        "zone": 2,
        "label": "Moderate Heat Strain",
        "lo": 1.0,
        "hi": 3.0,
        "color": "#f5c842",
        "perf": "Potential Performance Decline",
    },
    {
        "zone": 3,
        "label": "High Heat Strain",
        "lo": 3.0,
        "hi": 7.0,
        "color": "#f0823a",
        "perf": "Performance Decline",
    },
    {
        "zone": 4,
        "label": "Extremely High Heat Strain",
        "lo": 7.0,
        "hi": 10.1,
        "color": "#e8394a",
        "perf": "Dangerous",
    },
]
