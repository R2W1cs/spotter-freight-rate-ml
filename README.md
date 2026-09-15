# Freight Rate Prediction — Spotter ML Engineer Assessment

End-to-end solution for predicting truckload `posted_rate`.

**Author:** Rayen Oueslati  
**Holdout (Oct 2025):** MAE **$108.79**, MAPE **5.15%** (GPU XGBoost + Huber blend)  
**Scorer:** official `score.py` validates 12,000 predictions and writes `scorer_results/candidate_december.png`

See `Freight_Rate_ML_Assessment.pdf` for the brief and `reports/Freight_Rate_Prediction_Report.pdf` for the write-up.

---

## What this delivers

| Deliverable | Location |
|-------------|----------|
| Validation predictions (12,000 rows) | `validation_predictions.csv` |
| December fixed-lane predictions | `data/december_chart_inputs.csv` |
| Official December chart | `scorer_results/candidate_december.png` |
| Technical report (PDF) | `reports/Freight_Rate_Prediction_Report.pdf` |

---

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/eda.py
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

GPU note: training prefers **XGBoost CUDA** (`tree_method='hist'`, `device='cuda'`). If CUDA is unavailable it falls back to XGBoost CPU, then sklearn `HistGradientBoostingRegressor`.

---

## Approach (short)

1. **Chronological split** — train Jan–Sep 2025, hold out Oct 2025. Random splits would leak future market conditions.
2. **Cleaning** — absolute value on negative weights; median impute `weight` / `market_index` (fit on train only); winsorize extreme rates when fitting.
3. **Features** — distance transforms, equipment, geo, calendar effects, and economic interactions (`distance × market_index`, `distance × quote_signal`).
4. **Model** — XGBoost on `log1p(posted_rate)` with MAE objective, blended with Huber regression.
5. **December chart** — fixed Lexington → Fort Wayne Dry Van lane; forecast missing market/quote from history, then predict daily rates (non-flat by design).

---

## Project layout

```
data/
  train_test.csv
  validation.csv
  validation_predictions_template.csv
  december_chart_inputs.csv
src/freight/
  config.py      # paths, feature lists, blend weights
  features.py    # clean, impute, engineer
  model.py       # GPU/CPU backends, train, predict
  december.py    # fixed-lane enrichment + market/quote forecast
scripts/eda.py
train_predict.py
score.py                     # provided — do not modify
validation_predictions.csv
reports/
  Freight_Rate_Prediction_Report.pdf
  holdout_metrics.json
  figures/
scorer_results/candidate_december.png
requirements.txt
```

---

## Design choices worth reviewing

- **Why not random CV?** Validation is later in time; chronological holdout matches deployment.
- **Why log target?** Absolute dollar error grows with distance; log compresses that scale.
- **Why blend Huber?** A linear mile-rate skeleton is economically plausible; the tree captures residuals/interactions.
- **Why forecast December market/quote?** The template omits them; a constant impute yields an unrealistically flat chart when only date varies.

---

## Requirements

```
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
scikit-learn>=1.4,<2
joblib>=1.3,<2
xgboost>=2.0,<4
```
