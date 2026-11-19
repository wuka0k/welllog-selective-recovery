"""Create downstream decision summary and figure for FORCE classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parents[2] / "outputs"
FEATURES = ["GR", "RHOB", "NPHI", "DTC"]
ACCEPTED = ["RHOB", "NPHI", "DTC"]


def main() -> None:
    raw = pd.read_csv(OUT / "welllog_downstream_force_raw.csv")
    summary = pd.read_csv(OUT / "welllog_downstream_force_summary.csv")
    paired = pd.read_csv(OUT / "welllog_downstream_force_paired.csv")

    def avg(method: str, metric: str, features: list[str]) -> float:
        sub = raw[
            (raw.method == method) & raw.feature.isin(features)
        ]
        return float(sub.groupby("seed")[metric].mean().mean())

    decision = {
        "dataset": "FORCE 2020 lithology",
        "split": "10 random well-disjoint train/calibration/test splits",
        "classifier": "XGBoost multiclass",
        "classes": 11,
        "accepted_features": ACCEPTED,
        "deferred_features": ["GR"],
        "recovery_feature_coverage": 0.75,
        "oracle_macro_f1": float(
            summary[
                (summary.method == "oracle")
            ].macro_f1_mean.iloc[0]
        ),
        "oracle_accuracy": float(
            summary[
                (summary.method == "oracle")
            ].accuracy_mean.iloc[0]
        ),
        "accepted_feature_mean": {
            "raw_missing": {
                "accuracy": avg("raw_missing", "accuracy", ACCEPTED),
                "macro_f1": avg("raw_missing", "macro_f1", ACCEPTED),
                "weighted_f1": avg("raw_missing", "weighted_f1", ACCEPTED),
            },
            "mean_fill": {
                "accuracy": avg("mean_fill", "accuracy", ACCEPTED),
                "macro_f1": avg("mean_fill", "macro_f1", ACCEPTED),
                "weighted_f1": avg("mean_fill", "weighted_f1", ACCEPTED),
            },
            "cross_log": {
                "accuracy": avg("cross_log_all", "accuracy", ACCEPTED),
                "macro_f1": avg("cross_log_all", "macro_f1", ACCEPTED),
                "weighted_f1": avg("cross_log_all", "weighted_f1", ACCEPTED),
            },
        },
        "inference": {
            "GR_cross_log_r2_range": [
                float(raw[raw.feature == "GR"].cross_log_r2.min()),
                float(raw[raw.feature == "GR"].cross_log_r2.max()),
            ],
            "accepted_cross_log_r2_range": [
                float(raw[raw.feature.isin(ACCEPTED)].cross_log_r2.min()),
                float(raw[raw.feature.isin(ACCEPTED)].cross_log_r2.max()),
            ],
        },
        "paired_tests": paired.to_dict(orient="records"),
    }
    (OUT / "welllog_downstream_force_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), constrained_layout=True)
    methods = ["raw_missing", "mean_fill", "cross_log_all", "oracle"]
    labels = ["Raw missing", "Mean fill", "Cross-log", "Oracle"]
    colors = ["#D9E2E8", "#9DB2BF", "#386C7A", "#123B46"]
    for ax, feature in zip(axes, FEATURES):
        values = []
        errors = []
        for method in methods:
            if method == "oracle":
                sub = raw[raw.method == method]
            else:
                sub = raw[
                    (raw.feature == feature) & (raw.method == method)
                ]
            values.append(float(sub.macro_f1.mean()))
            errors.append(float(sub.macro_f1.std()))
        ax.bar(
            np.arange(len(methods)),
            values,
            yerr=errors,
            capsize=3,
            color=colors,
        )
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 0.36)
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.25)
        if feature == "GR":
            ax.text(0.5, 0.33, "cross-log deferred", ha="center", fontsize=8)
        else:
            ax.text(0.5, 0.33, "accept: R²≥0.5", ha="center", fontsize=8)
    axes[0].set_ylabel("Macro-F1")
    fig.suptitle("FORCE 2020 downstream lithology audit (10 well splits)")
    fig.savefig(OUT / "fig_welllog_force_downstream.png", dpi=190)
    plt.close(fig)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
