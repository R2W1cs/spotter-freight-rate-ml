"""Exploratory analysis figures used in the assessment report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from freight.config import DATA, FIGURES  # noqa: E402
from freight.features import clean_frame, load_csv  # noqa: E402


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#9DAFB3",
            "axes.grid": True,
            "grid.color": "#E6EEF0",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    df = clean_frame(load_csv(DATA / "train_test.csv"))
    df["rpm"] = df["posted_rate"] / df["distance"]

    # 1) Rate vs distance by equipment
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    for equip, color in zip(
        ["Dry Van", "Flatbed", "Reefer"], ["#064A56", "#C45C26", "#2F6F4E"]
    ):
        sample = df[df["equipment"] == equip].sample(
            n=min(2500, (df["equipment"] == equip).sum()), random_state=42
        )
        ax.scatter(
            sample["distance"],
            sample["posted_rate"],
            s=8,
            alpha=0.25,
            label=equip,
            color=color,
        )
    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Posted rate ($)")
    ax.set_title("Posted rate scales primarily with distance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "rate_vs_distance.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Daily market index
    daily = df.groupby("date").agg(
        market=("market_index", "median"),
        rpm=("rpm", "median"),
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=160)
    ax.plot(daily.index, daily["market"], color="#064A56", lw=1.8)
    ax.set_title("Market index has clear seasonal structure")
    ax.set_ylabel("Median market_index")
    fig.tight_layout()
    fig.savefig(FIGURES / "market_index_daily.png", bbox_inches="tight")
    plt.close(fig)

    # 3) Equipment RPM
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
    order = ["Dry Van", "Flatbed", "Reefer"]
    stats = df.groupby("equipment")["rpm"].median().reindex(order)
    ax.bar(order, stats.values, color=["#064A56", "#C45C26", "#2F6F4E"])
    ax.set_ylabel("Median $/mile")
    ax.set_title("Equipment premium: Reefer > Flatbed > Dry Van")
    fig.tight_layout()
    fig.savefig(FIGURES / "equipment_rpm.png", bbox_inches="tight")
    plt.close(fig)

    # 4) Data quality summary (from raw CSV for missing/negative counts)
    raw = pd.read_csv(DATA / "train_test.csv")
    summary = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "missing_weight": int(raw["weight"].isna().sum()),
        "missing_market_index": int(raw["market_index"].isna().sum()),
        "negative_weight": int((raw["weight"] < 0).sum()),
        "cities": int(df["pickup"].nunique()),
        "corr_distance_rate": float(df["distance"].corr(df["posted_rate"])),
        "corr_market_rpm": float(daily["market"].corr(daily["rpm"])),
    }
    (FIGURES / "eda_summary.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    print("Wrote EDA figures to", FIGURES)
    print(summary)


if __name__ == "__main__":
    main()
