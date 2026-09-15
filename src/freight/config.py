from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

EQUIP_MAP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}

# Tree model carries most of the non-linear signal; Huber anchors extreme residuals.
BLEND_TREE = 0.85
BLEND_HUBER = 0.15

# Winsorize rare extreme posted rates when fitting (keeps evaluation unclipped).
RATE_CLIP_Q = 0.997

HOLDOUT_START = "2025-10-01"

TREE_FEATURES = [
    "distance",
    "log_dist",
    "sqrt_dist",
    "weight",
    "w_per_mile",
    "market_index",
    "quote_signal",
    "mq",
    "month",
    "dow",
    "doy",
    "week",
    "dom",
    "is_weekend",
    "equip_num",
    "pickup_code",
    "delivery_code",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "lat_diff",
    "lon_diff",
    "d_m",
    "d_q",
    "d_equip",
]

HUBER_FEATURES = [
    "distance",
    "d_m",
    "d_q",
    "d_reefer",
    "d_flat",
    "weight",
    "doy",
    "dow",
    "market_index",
    "quote_signal",
]
