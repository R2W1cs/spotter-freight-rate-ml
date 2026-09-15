from __future__ import annotations

import numpy as np
import pandas as pd


def city_coords(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Stable lat/lon lookup — each city has a single coordinate pair in this dataset."""
    coords: dict[str, tuple[float, float]] = {}
    pick = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    deliv = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    for city, row in pick.iterrows():
        coords[str(city)] = (float(row["pickup_lat"]), float(row["pickup_lon"]))
    for city, row in deliv.iterrows():
        coords.setdefault(
            str(city), (float(row["delivery_lat"]), float(row["delivery_lon"]))
        )
    return coords


def forecast_daily_series(
    train: pd.DataFrame, column: str, dates: pd.DatetimeIndex
) -> pd.Series:
    """Forecast a global daily series (e.g. market_index) for future dates.

    Combines day-of-year seasonality, day-of-week adjustment, and a short
    residual replay from the most recent 31 observed days, then recenters
    toward the latest market level.
    """
    daily = (
        train.dropna(subset=[column]).groupby("date")[column].median().sort_index()
    )
    hist = daily.reset_index()
    hist["doy"] = hist["date"].dt.dayofyear
    hist["dow"] = hist["date"].dt.dayofweek
    doy_mean = hist.groupby("doy")[column].mean()
    dow_mean = hist.groupby("dow")[column].mean()
    overall = float(daily.mean())
    recent = daily.tail(31)
    recent_level = float(recent.mean())
    recent_resid = recent - recent.mean()

    seasonal = []
    for doy in dates.dayofyear:
        if doy in doy_mean.index:
            seasonal.append(float(doy_mean.loc[doy]))
        else:
            # Map late-year doy onto late-summer/fall history when Dec is unseen.
            proxy = int(((doy - 335) % 61) + 274)
            seasonal.append(float(doy_mean.get(proxy, overall)))
    seasonal = pd.Series(seasonal, index=dates, dtype=float)

    dow_adj = pd.Series(
        [float(dow_mean.get(d, overall)) - overall for d in dates.dayofweek],
        index=dates,
        dtype=float,
    )
    replay = np.resize(recent_resid.values, len(dates))
    level_gap = recent_level - float(seasonal.mean())
    forecast = seasonal + 0.55 * level_gap + dow_adj + 0.85 * replay
    return forecast.clip(
        lower=float(daily.quantile(0.01)), upper=float(daily.quantile(0.99))
    )


def forecast_quote_signal(train: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Lane-aware quote forecast for Lexington → Fort Wayne Dry Van."""
    mask = (
        (train["pickup"] == "Lexington")
        & (train["delivery"] == "Fort Wayne")
        & (train["equipment"] == "Dry Van")
    )
    lane = train.loc[mask, "quote_signal"].dropna()
    base = float(lane.median() if len(lane) >= 5 else train["quote_signal"].median())

    daily = train.groupby("date")["quote_signal"].median().sort_index()
    recent = daily.tail(31)
    recent_resid = recent - recent.mean()
    dow_mean = (
        daily.reset_index()
        .assign(dow=lambda d: d["date"].dt.dayofweek)
        .groupby("dow")["quote_signal"]
        .mean()
    )
    overall = float(daily.mean())
    dow_adj = pd.Series(
        [float(dow_mean.get(d, overall)) - overall for d in dates.dayofweek],
        index=dates,
        dtype=float,
    )
    replay = np.resize(recent_resid.values, len(dates))
    return pd.Series(base + dow_adj.values + 0.9 * replay, index=dates).clip(
        lower=0.8, upper=3.2
    )


def build_december_features(train: pd.DataFrame, template: pd.DataFrame) -> pd.DataFrame:
    """Enrich the fixed December template with coords + forecasted market signals."""
    coords = city_coords(train)
    pickup_lat, pickup_lon = coords["Lexington"]
    delivery_lat, delivery_lon = coords["Fort Wayne"]
    dates = pd.DatetimeIndex(pd.to_datetime(template["date"]))

    out = template.copy()
    out["date"] = dates
    out["pickup_lat"] = pickup_lat
    out["pickup_lon"] = pickup_lon
    out["delivery_lat"] = delivery_lat
    out["delivery_lon"] = delivery_lon
    out["market_index"] = forecast_daily_series(train, "market_index", dates).values
    out["quote_signal"] = forecast_quote_signal(train, dates).values
    return out
