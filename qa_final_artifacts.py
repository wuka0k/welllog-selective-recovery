"""Recompute headline claims from saved artifacts and fail on drift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
ACCEPTED = ["RHOB", "NPHI", "DTC"]
TOL = 1e-5


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), expected, atol=TOL, rtol=0):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def main() -> None:
    summary = pd.read_csv(OUT / "welllog_recovery_summary_full5fold.csv")
    paired = pd.read_csv(OUT / "welllog_recovery_paired_full5fold.csv")
    raw = pd.read_csv(OUT / "welllog_recovery_raw_full5fold.csv")
    downstream = pd.read_csv(OUT / "welllog_downstream_force_raw.csv")

    expected = {
        ("taranaki", "SAITS", "selective_mae_mean"): 0.730231,
        ("taranaki", "NoCrossExpert", "selective_mae_mean"): 0.690619,
        ("taranaki", "Proposed", "selective_mae_mean"): 0.349780,
        ("taranaki", "Proposed", "coverage_mean"): 0.894745,
        ("taranaki", "Proposed", "outside_rate_mean"): 0.102292,
        ("taranaki", "Proposed", "defer_mean"): 0.346667,
        ("geolink", "SAITS", "selective_mae_mean"): 0.602945,
        ("geolink", "NoCrossExpert", "selective_mae_mean"): 0.766420,
        ("geolink", "Proposed", "selective_mae_mean"): 0.335542,
        ("geolink", "Proposed", "coverage_mean"): 0.891833,
        ("geolink", "Proposed", "outside_rate_mean"): 0.112958,
        ("geolink", "Proposed", "defer_mean"): 0.5,
        ("teapot", "Proposed", "defer_mean"): 1.0,
    }
    for (dataset, method, metric), value in expected.items():
        row = summary[
            (summary.dataset == dataset)
            & (summary.condition == "profile")
            & (summary.method == method)
        ]
        assert len(row) == 1, (dataset, method, len(row))
        close(row.iloc[0][metric], value, f"{dataset}/{method}/{metric}")

    for dataset, baseline, delta, holm in [
        ("taranaki", "SAITS", -0.380451, 9.536743e-7),
        ("geolink", "SAITS", -0.267403, 9.536743e-7),
        ("taranaki", "NoCrossExpert", -0.340838, 9.536743e-7),
        ("geolink", "NoCrossExpert", -0.430878, 9.536743e-7),
    ]:
        row = paired[
            (paired.dataset == dataset)
            & (paired.condition == "profile")
            & (paired.baseline == baseline)
            & (paired.metric == "selective_mae")
        ]
        assert len(row) == 1
        close(row.iloc[0]["mean_delta"], delta, f"{dataset}/{baseline}/delta")
        close(row.iloc[0]["holm_p"], holm, f"{dataset}/{baseline}/holm")
        assert int(row.iloc[0]["n_fold_seed_pairs"]) == 25
        assert int(row.iloc[0]["folds_favor_proposed"]) == 5
        assert int(row.iloc[0]["folds_total"]) == 5

    proposed_profile = raw[
        raw.action.isin(["accept", "recover", "defer"])
        & (raw.condition == "profile")
    ]
    assert set(proposed_profile.action) <= {"recover", "defer"}
    assert not (
        (proposed_profile.accept_or_recover == 1) & (proposed_profile.action != "recover")
    ).any()
    proposed_local = raw[
        raw.action.isin(["accept", "recover", "defer"])
        & (raw.condition != "profile")
    ]
    assert set(proposed_local.action) <= {"accept", "defer"}
    assert not (
        (proposed_local.accept_or_recover == 1) & (proposed_local.action != "accept")
    ).any()

    accepted_rows = downstream[
        (downstream.method == "cross_log_all")
        & downstream.feature.isin(ACCEPTED)
    ]
    assert len(accepted_rows) == 30
    downstream_means = {
        "cross_log_recovery": {
            metric: float(accepted_rows[metric].mean())
            for metric in ["accuracy", "macro_f1", "weighted_f1"]
        }
    }
    for method in ["raw_missing", "mean_fill"]:
        selected = downstream[
            (downstream.method == method) & downstream.feature.isin(ACCEPTED)
        ]
        downstream_means[method] = {
            metric: float(selected[metric].mean())
            for metric in ["accuracy", "macro_f1", "weighted_f1"]
        }
    oracle = downstream[downstream.method == "oracle"]
    downstream_means["oracle"] = {
        metric: float(oracle[metric].mean())
        for metric in ["accuracy", "macro_f1", "weighted_f1"]
    }
    for method, values in {
        "raw_missing": (0.382427, 0.134416, 0.418763),
        "mean_fill": (0.356467, 0.189709, 0.416737),
        "cross_log_recovery": (0.475968, 0.243755, 0.537356),
        "oracle": (0.498940, 0.265476, 0.561239),
    }.items():
        for metric, value in zip(
            ["accuracy", "macro_f1", "weighted_f1"], values
        ):
            close(downstream_means[method][metric], value, f"{method}/{metric}")

    r2 = {
        feature: [
            float(downstream[(downstream.method == "cross_log_all") & (downstream.feature == feature)].cross_log_r2.min()),
            float(downstream[(downstream.method == "cross_log_all") & (downstream.feature == feature)].cross_log_r2.max()),
        ]
        for feature in ["GR", "RHOB", "NPHI", "DTC"]
    }
    assert r2["GR"][1] < 0.5
    assert all(r2[feature][0] >= 0.5 for feature in ACCEPTED)

    result = {
        "status": "PASS",
        "checks": {
            "fivefold_headline_metrics": True,
            "paired_wilcoxon_holm": True,
            "action_estimator_consistency": True,
            "downstream_real_label_metrics": True,
            "identifiability_boundary": True,
        },
        "downstream_accepted_feature_means": downstream_means,
        "cross_log_r2_ranges": r2,
        "oracle_macro_f1_gap": (
            downstream_means["oracle"]["macro_f1"]
            - downstream_means["cross_log_recovery"]["macro_f1"]
        ),
        "coverage_definition": (
            "Empirical fraction of all masked test points whose error is inside the "
            "calibrated interval; defer cases remain in the denominator."
        ),
        "outside_rate_definition": (
            "Among released pattern-feature groups, the fraction whose errors lie "
            "outside the calibrated interval. This is not a binary correctness error."
        ),
    }
    (OUT / "welllog_q2_numeric_qa.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
