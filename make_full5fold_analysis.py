"""Aggregate five-fold, five-seed 100-epoch recovery experiments."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from make_recovery_analysis import holm, weighted_mean


OUT = Path(__file__).resolve().parents[2] / "outputs"
ACTIONS = ["accept", "recover", "defer"]


def fold_of(path: Path) -> int:
    match = re.search(r"fold([0-4])", path.name)
    if match:
        return int(match.group(1))
    if "mid100" in path.name:
        return 0
    raise ValueError(f"cannot infer fold from {path}")


def load_raw() -> pd.DataFrame:
    paths = [
        OUT / "welllog_recovery_raw_mid100.csv",
        *sorted(OUT.glob("welllog_recovery_raw_fold[1-4]_*.csv")),
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["fold"] = fold_of(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    assert set(raw.fold.unique()) == {0, 1, 2, 3, 4}
    assert set(raw.seed.unique()) == {0, 1, 2, 3, 4}
    assert set(raw.mask_rep.unique()) == {0, 1, 2, 3, 4}
    return raw


def condition_aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, fold, seed, rep, condition), block in raw.groupby(
        ["dataset", "fold", "seed", "mask_rep", "condition"]
    ):
        full = block[block.action.isin(ACTIONS)]
        accepted = full[full.accept_or_recover == 1]
        rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "mask_rep": rep,
                "condition": condition,
                "method": "Proposed",
                "mae": weighted_mean(full.mae, full.n_points),
                "selective_mae": weighted_mean(
                    accepted.selective_mae, accepted.n_points
                ),
                "coverage": weighted_mean(full.coverage, full.n_points),
                "outside_rate": weighted_mean(
                    accepted.outside_rate, accepted.n_points
                ),
                "defer": weighted_mean(full.defer, full.n_points),
                "normalized_width": weighted_mean(
                    full.normalized_width, full.n_points
                ),
            }
        )
        for action, method in [
            ("saits_point", "SAITS"),
            ("interval_only", "IntervalOnly"),
            ("width_only", "WidthOnly"),
            ("no_cross_expert", "NoCrossExpert"),
        ]:
            other = block[block.action == action]
            accepted_other = other[other.accept_or_recover == 1]
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "mask_rep": rep,
                    "condition": condition,
                    "method": method,
                    "mae": weighted_mean(other.mae, other.n_points),
                    "selective_mae": weighted_mean(
                        accepted_other.selective_mae,
                        accepted_other.n_points,
                    ),
                    "coverage": weighted_mean(other.coverage, other.n_points),
                    "outside_rate": weighted_mean(
                        accepted_other.outside_rate,
                        accepted_other.n_points,
                    ),
                    "defer": weighted_mean(other.defer, other.n_points),
                    "normalized_width": weighted_mean(
                        other.normalized_width, other.n_points
                    ),
                }
            )
    result = pd.DataFrame(rows)
    # Primary unit: one fold-seed, with mask repetitions averaged out.
    return (
        result.groupby(
            ["dataset", "fold", "seed", "condition", "method"], as_index=False
        )[
            [
                "mae",
                "selective_mae",
                "coverage",
                "outside_rate",
                "defer",
                "normalized_width",
            ]
        ]
        .mean()
    )


def paired_tests(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["mae", "selective_mae", "coverage", "outside_rate", "defer"]
    for dataset in unit.dataset.unique():
        for condition in unit.condition.unique():
            proposed = unit[
                (unit.dataset == dataset)
                & (unit.condition == condition)
                & (unit.method == "Proposed")
            ]
            for baseline in ["SAITS", "IntervalOnly", "WidthOnly", "NoCrossExpert"]:
                other = unit[
                    (unit.dataset == dataset)
                    & (unit.condition == condition)
                    & (unit.method == baseline)
                ]
                pair = proposed.merge(
                    other, on=["fold", "seed"], suffixes=("_p", "_b")
                )
                fold_deltas = (
                    pair.assign(
                        **{
                            metric: pair[f"{metric}_p"] - pair[f"{metric}_b"]
                            for metric in metrics
                        }
                    )
                    .groupby("fold")[metrics]
                    .mean()
                )
                for metric in metrics:
                    a = pair[f"{metric}_p"].to_numpy(float)
                    b = pair[f"{metric}_b"].to_numpy(float)
                    valid = np.isfinite(a) & np.isfinite(b)
                    a, b = a[valid], b[valid]
                    delta = a - b
                    if not len(a) or np.allclose(delta, 0):
                        p = 1.0
                    else:
                        try:
                            p = float(stats.wilcoxon(a, b).pvalue)
                        except ValueError:
                            p = 1.0
                    lower = int((delta < 0).sum())
                    upper = int((delta > 0).sum())
                    fd = fold_deltas[metric].dropna()
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "baseline": baseline,
                            "metric": metric,
                            "n_fold_seed_pairs": len(a),
                            "mean_delta": float(delta.mean()) if len(a) else np.nan,
                            "wilcoxon_p": p,
                            "cliffs_delta": (lower - upper)
                            / max(lower + upper, 1),
                            "folds_favor_proposed": int((fd < 0).sum()),
                            "folds_total": int(fd.size),
                        }
                    )
    result = pd.DataFrame(rows)
    result["holm_p"] = 1.0
    for _, loc in result.groupby(["dataset", "metric"]).groups.items():
        sub = result.loc[loc].sort_values("wilcoxon_p")
        result.loc[sub.index, "holm_p"] = holm(sub.wilcoxon_p.tolist())
    return result


def plot_profile_by_fold(unit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["taranaki", "geolink", "teapot"]):
        sub = unit[
            (unit.dataset == dataset) & (unit.condition == "profile")
        ]
        folds = sorted(sub.fold.unique())
        for method, color, marker in [
            ("SAITS", "#9DB2BF", "o"),
            ("NoCrossExpert", "#607D8B", "s"),
            ("Proposed", "#006D77", "D"),
        ]:
            means = []
            stds = []
            for fold in folds:
                x = sub[
                    (sub.fold == fold) & (sub.method == method)
                ].selective_mae.mean()
                means.append(x if np.isfinite(x) else np.nan)
                stds.append(
                    sub[
                        (sub.fold == fold) & (sub.method == method)
                    ].selective_mae.std()
                )
            ax.errorbar(
                np.asarray(folds) + {"SAITS": -0.12, "NoCrossExpert": 0, "Proposed": 0.12}[method],
                means,
                yerr=stds,
                marker=marker,
                capsize=3,
                label=method,
                color=color,
            )
        ax.set_title(dataset)
        ax.set_xlabel("fold")
        ax.set_ylabel("profile selective MAE")
        ax.set_xticks(folds)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Five-fold stability of profile recovery")
    fig.savefig(OUT / "fig_welllog_full5fold_profile.png", dpi=190)
    plt.close(fig)


def main() -> None:
    raw = load_raw()
    raw.to_csv(OUT / "welllog_recovery_raw_full5fold.csv", index=False)
    unit = condition_aggregate(raw)
    unit.to_csv(OUT / "welllog_recovery_condition_runs_full5fold.csv", index=False)
    summary = (
        unit.groupby(["dataset", "condition", "method"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            selective_mae_mean=("selective_mae", "mean"),
            selective_mae_std=("selective_mae", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            outside_rate_mean=("outside_rate", "mean"),
            outside_rate_std=("outside_rate", "std"),
            defer_mean=("defer", "mean"),
            defer_std=("defer", "std"),
        )
    )
    summary.to_csv(OUT / "welllog_recovery_summary_full5fold.csv", index=False)
    tests = paired_tests(unit)
    tests.to_csv(OUT / "welllog_recovery_paired_full5fold.csv", index=False)
    plot_profile_by_fold(unit)
    manifest = {
        "datasets": sorted(raw.dataset.unique().tolist()),
        "folds": sorted(raw.fold.unique().tolist()),
        "seeds": sorted(raw.seed.unique().tolist()),
        "masks_per_seed": int(raw.mask_rep.nunique()),
        "raw_rows": int(raw.shape[0]),
        "fold_seed_pairs_per_method_condition": int(
            len(
                unit[
                    (unit.dataset == "taranaki")
                    & (unit.condition == "profile")
                ]
            )
            / 5
        ),
        "significant_holm": int(
            ((tests.holm_p < 0.1) & (tests.n_fold_seed_pairs > 0)).sum()
        ),
    }
    (OUT / "welllog_recovery_manifest_full5fold.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(tests[tests.holm_p < 0.1].to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
