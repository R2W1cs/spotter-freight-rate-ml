from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import (
    BLEND_HUBER,
    BLEND_TREE,
    HOLDOUT_START,
    HUBER_FEATURES,
    RATE_CLIP_Q,
    TREE_FEATURES,
)
from .features import fit_city_encodings, fit_imputer, prepare


def detect_tree_backend() -> tuple[str, type | None]:
    """Prefer XGBoost CUDA → XGBoost CPU → sklearn HistGradientBoosting."""
    try:
        from xgboost import XGBRegressor

        probe = XGBRegressor(
            n_estimators=8,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        probe.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
        print("Tree backend: XGBoost (CUDA)")
        return "xgboost_cuda", XGBRegressor
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA XGBoost unavailable ({exc}); trying CPU fallbacks...")

    try:
        from xgboost import XGBRegressor

        print("Tree backend: XGBoost (CPU hist)")
        return "xgboost_cpu", XGBRegressor
    except Exception as exc:  # noqa: BLE001
        print(f"XGBoost unavailable ({exc}); using HistGradientBoosting")
        return "hgb", None


def make_tree_model(backend: str, xgb_cls: type | None) -> Any:
    if backend.startswith("xgboost") and xgb_cls is not None:
        device = "cuda" if backend == "xgboost_cuda" else "cpu"
        return xgb_cls(
            n_estimators=1600,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.2,
            reg_alpha=0.1,
            objective="reg:absoluteerror",
            tree_method="hist",
            device=device,
            early_stopping_rounds=50,
            random_state=42,
            n_jobs=0 if device == "cuda" else -1,
            verbosity=0,
        )
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=10,
        learning_rate=0.04,
        max_iter=800,
        l2_regularization=0.05,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )


def fit_models(
    train: pd.DataFrame, backend: str, xgb_cls: type | None
) -> tuple[Any, HuberRegressor, dict]:
    medians = fit_imputer(train)
    city_encodings = fit_city_encodings(train)
    train_f = prepare(train, medians, city_encodings)

    x_all = train_f[TREE_FEATURES]
    y_all = np.log1p(train_f["posted_rate"])

    cutoff = train_f["date"].quantile(0.9)
    train_mask = train_f["date"] <= cutoff
    x_tr, y_tr = x_all.loc[train_mask], y_all.loc[train_mask]
    x_va, y_va = x_all.loc[~train_mask], y_all.loc[~train_mask]

    tree = make_tree_model(backend, xgb_cls)
    if backend.startswith("xgboost"):
        tree.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
        best_iter = int(
            getattr(tree, "best_iteration", tree.n_estimators) or tree.n_estimators
        )
    else:
        tree.fit(x_all, y_all)
        best_iter = int(tree.n_iter_)

    huber = HuberRegressor(max_iter=2000, epsilon=1.35)
    huber.fit(train_f[HUBER_FEATURES], train_f["posted_rate"])

    meta = {
        "medians": medians,
        "city_encodings": city_encodings,
        "backend": backend,
        "best_iteration": best_iter,
    }
    return tree, huber, meta


def predict_rates(
    frame: pd.DataFrame,
    tree: Any,
    huber: HuberRegressor,
    medians: dict[str, float],
    city_encodings: dict[str, dict[str, int]],
) -> np.ndarray:
    prepared = prepare(frame, medians, city_encodings)
    tree_pred = np.expm1(tree.predict(prepared[TREE_FEATURES]))
    linear_pred = huber.predict(prepared[HUBER_FEATURES])
    blended = BLEND_TREE * tree_pred + BLEND_HUBER * linear_pred
    return np.maximum(blended, 1.0)


def evaluate_holdout(full: pd.DataFrame, backend: str, xgb_cls: type | None) -> dict:
    hi = full["posted_rate"].quantile(RATE_CLIP_Q)
    train = full[(full["date"] < HOLDOUT_START) & (full["posted_rate"] <= hi)].copy()
    test = full[full["date"] >= HOLDOUT_START].copy()
    tree, huber, meta = fit_models(train, backend, xgb_cls)
    pred = predict_rates(
        test, tree, huber, meta["medians"], meta["city_encodings"]
    )
    metrics = {
        "holdout": "2025-10",
        "split": "chronological: train Jan–Sep 2025, holdout Oct 2025",
        "backend": backend,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "mae": float(mean_absolute_error(test["posted_rate"], pred)),
        "rmse": float(mean_squared_error(test["posted_rate"], pred) ** 0.5),
        "mape": float(
            np.mean(np.abs((test["posted_rate"] - pred) / test["posted_rate"])) * 100
        ),
        "best_iteration": meta["best_iteration"],
        "blend": {"tree": BLEND_TREE, "huber": BLEND_HUBER},
    }
    print(
        f"October holdout [{backend}] — MAE=${metrics['mae']:.2f}  "
        f"RMSE=${metrics['rmse']:.2f}  MAPE={metrics['mape']:.2f}%"
    )
    return metrics
