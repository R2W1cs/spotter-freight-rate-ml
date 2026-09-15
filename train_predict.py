"""Train the freight-rate model and write submission prediction files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from freight.config import DATA, MODELS, RATE_CLIP_Q, REPORTS  # noqa: E402
from freight.december import build_december_features  # noqa: E402
from freight.features import clean_frame, load_csv  # noqa: E402
from freight.model import detect_tree_backend, evaluate_holdout, fit_models, predict_rates  # noqa: E402


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    backend, xgb_cls = detect_tree_backend()
    full = clean_frame(load_csv(DATA / "train_test.csv"))
    metrics = evaluate_holdout(full, backend, xgb_cls)

    # Final fit on all labeled data (still winsorize extreme rates for stability).
    hi = full["posted_rate"].quantile(RATE_CLIP_Q)
    train_all = full[full["posted_rate"] <= hi].copy()
    tree, huber, meta = fit_models(train_all, backend, xgb_cls)

    artifact = {
        "tree": tree,
        "huber": huber,
        "medians": meta["medians"],
        "city_encodings": meta["city_encodings"],
        "backend": backend,
        "metrics": metrics,
    }
    joblib.dump(artifact, MODELS / "rate_model.joblib")

    validation = clean_frame(load_csv(DATA / "validation.csv"))
    val_pred = predict_rates(
        validation, tree, huber, meta["medians"], meta["city_encodings"]
    )
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    merged = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": val_pred}),
        on="load_id",
        how="left",
    )
    if merged["predicted_rate"].isna().any():
        raise SystemExit("Missing predictions for some validation load_id values")
    out_path = ROOT / "validation_predictions.csv"
    merged.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(merged):,} rows)")

    december_path = DATA / "december_chart_inputs.csv"
    december_tmpl = pd.read_csv(december_path)[
        ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    ].copy()
    # If predicted_rate was filled by a prior run, start from blank target column.
    december_tmpl["predicted_rate"] = pd.NA
    december_feat = build_december_features(full, december_tmpl)
    dec_pred = predict_rates(
        december_feat, tree, huber, meta["medians"], meta["city_encodings"]
    )
    december_out = december_tmpl.copy()
    december_out["predicted_rate"] = dec_pred
    december_out.to_csv(december_path, index=False)
    print(
        f"Wrote {december_path} "
        f"(rate range ${december_out['predicted_rate'].min():.2f}"
        f"–${december_out['predicted_rate'].max():.2f})"
    )

    (REPORTS / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
