from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EQUIP_MAP


def load_csv(path, parse_dates: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if parse_dates and "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues without inventing values.

    - Negative weights are treated as sign errors (absolute value).
    - Dates are parsed to datetime.
    """
    out = frame.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "weight" in out.columns:
        out["weight"] = out["weight"].abs()
    return out


def fit_imputer(train: pd.DataFrame) -> dict[str, float]:
    return {
        "weight": float(train["weight"].median()),
        "market_index": float(train["market_index"].median()),
        "quote_signal": float(train["quote_signal"].median()),
    }


def apply_imputer(frame: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for column, value in medians.items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    return out


def fit_city_encodings(train: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "pickup": {c: i for i, c in enumerate(sorted(train["pickup"].unique()))},
        "delivery": {c: i for i, c in enumerate(sorted(train["delivery"].unique()))},
    }


def add_features(
    frame: pd.DataFrame,
    city_encodings: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["month"] = out["date"].dt.month
    out["dow"] = out["date"].dt.dayofweek
    out["doy"] = out["date"].dt.dayofyear
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["dom"] = out["date"].dt.day
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["equip_num"] = out["equipment"].map(EQUIP_MAP).astype(float)

    if city_encodings is not None:
        out["pickup_code"] = out["pickup"].map(city_encodings["pickup"]).fillna(-1)
        out["delivery_code"] = (
            out["delivery"].map(city_encodings["delivery"]).fillna(-1)
        )
    else:
        out["pickup_code"] = -1.0
        out["delivery_code"] = -1.0

    out["log_dist"] = np.log1p(out["distance"])
    out["sqrt_dist"] = np.sqrt(out["distance"])
    out["w_per_mile"] = out["weight"] / (out["distance"] + 1.0)
    out["mq"] = out["market_index"] * out["quote_signal"]
    out["d_m"] = out["distance"] * out["market_index"]
    out["d_q"] = out["distance"] * out["quote_signal"]
    out["d_equip"] = out["distance"] * out["equip_num"]
    out["d_reefer"] = out["distance"] * (out["equipment"] == "Reefer").astype(float)
    out["d_flat"] = out["distance"] * (out["equipment"] == "Flatbed").astype(float)
    out["lat_diff"] = (out["pickup_lat"] - out["delivery_lat"]).abs()
    out["lon_diff"] = (out["pickup_lon"] - out["delivery_lon"]).abs()
    return out


def prepare(
    frame: pd.DataFrame,
    medians: dict[str, float],
    city_encodings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    return add_features(apply_imputer(clean_frame(frame), medians), city_encodings)
