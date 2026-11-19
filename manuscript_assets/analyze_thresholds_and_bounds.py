"""Post-hoc sensitivity and irreducible-risk analysis from frozen outputs.

The sensitivity analysis changes only release thresholds. It does not create
new model predictions and is therefore not a replacement for a fresh
calibration run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT / "evidence"
MANIFESTS = EVIDENCE / "fold_manifests"
TABLES = PROJECT / "tables"


def manifest_runs() -> dict[tuple[str, int, int], dict]:
    runs = {}
    for path in sorted(MANIFESTS.glob("*.json")):
        fold = (
            0
            if path.name == "welllog_recovery_manifest_mid100.json"
            else int(re.search(r"fold([0-4])", path.name).group(1))
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload["runs"]:
            runs[(run["dataset"], fold, int(run["seed"]))] = run
    return runs


def main() -> None:
    raw = pd.read_csv(EVIDENCE / "welllog_recovery_raw_full5fold.csv")
    runs = manifest_runs()
    proposed = raw[
        raw.action.isin(["accept", "recover", "defer"])
        & (raw.condition == "profile")
    ].copy()
    proposed["fold"] = proposed["fold"].astype(int)
    proposed["seed"] = proposed["seed"].astype(int)
    proposed["feature"] = proposed["feature"].astype(int)

    rows = []
    for row in proposed.itertuples(index=False):
        run = runs[(row.dataset, int(row.fold), int(row.seed))]
        std = float(run["diagnostics"][str(int(row.feature))]["target_std"])
        radius = 0.5 * float(row.normalized_width) * std
        q_values = np.asarray(list(run["radii"].values()), dtype=float)
        rows.append(
            {
                **row._asdict(),
                "radius": radius,
                "r2": float(row.cross_log_r2),
                "q90_tau": float(np.quantile(q_values, 0.90)),
                "q95_tau": float(np.quantile(q_values, 0.95)),
                "q975_tau": float(np.quantile(q_values, 0.975)),
            }
        )
    profile = pd.DataFrame(rows)
    profile.to_csv(
        EVIDENCE / "welllog_profile_threshold_rows.csv", index=False
    )

    sensitivity_rows = []
    for tau_name in ["q90_tau", "q95_tau", "q975_tau"]:
        for kappa in [0.30, 0.40, 0.50, 0.60, 0.70]:
            released = profile[
                (profile.radius <= profile[tau_name])
                & (profile.r2 >= kappa)
            ].copy()
            for (dataset, fold, seed), block in profile.groupby(
                ["dataset", "fold", "seed"]
            ):
                chosen = released[
                    (released.dataset == dataset)
                    & (released.fold == fold)
                    & (released.seed == seed)
                ]
                total_points = block.n_points.sum()
                accepted_points = chosen.n_points.sum()
                selective_mae = (
                    np.average(chosen.mae, weights=chosen.n_points)
                    if len(chosen)
                    else np.nan
                )
                sensitivity_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "tau_rule": tau_name.replace("_tau", ""),
                        "kappa": kappa,
                        "release_fraction": (
                            accepted_points / total_points
                            if total_points
                            else 0.0
                        ),
                        "selective_mae": selective_mae,
                    }
                )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        EVIDENCE / "welllog_threshold_sensitivity_units.csv", index=False
    )
    sensitivity_summary = (
        sensitivity.groupby(
            ["dataset", "tau_rule", "kappa"], as_index=False
        )
        .agg(
            release_mean=("release_fraction", "mean"),
            release_std=("release_fraction", "std"),
            selective_mae_mean=("selective_mae", "mean"),
            selective_mae_std=("selective_mae", "std"),
        )
    )
    sensitivity_summary.to_csv(
        EVIDENCE / "welllog_threshold_sensitivity_summary.csv", index=False
    )

    bounds = (
        profile.groupby(["dataset", "feature"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_min=("r2", "min"),
            r2_max=("r2", "max"),
        )
    )
    bounds["min_normalized_rmse_mean"] = np.clip(
        np.sqrt(np.maximum(1.0 - bounds.r2_mean, 0.0)), 0.0, 1.0
    )
    bounds["min_normalized_rmse_worst"] = np.clip(
        np.sqrt(np.maximum(1.0 - bounds.r2_min, 0.0)), 0.0, 1.0
    )
    bounds["min_normalized_rmse_best"] = np.clip(
        np.sqrt(np.maximum(1.0 - bounds.r2_max, 0.0)), 0.0, 1.0
    )
    bounds.to_csv(
        EVIDENCE / "welllog_identifiability_bounds.csv", index=False
    )

    lines = [
        r"\begin{table}",
        r"\caption{Sensitivity of the profile release rule to the calibrated "
        r"radius quantile and cross-log identifiability threshold. Values are "
        r"means over fold-seed units; release fraction is point-weighted.}",
        r"\label{tab:sensitivity}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccc@{}}",
        r"\toprule",
        r"Dataset & Radius & $\kappa$ & Release & Selective MAE & SD \\",
        r"\midrule",
    ]
    table_grid = sensitivity_summary[
        sensitivity_summary.kappa.isin([0.3, 0.5, 0.7])
    ]
    for row in table_grid.itertuples(index=False):
        mae = (
            "--"
            if not np.isfinite(row.selective_mae_mean)
            else f"{row.selective_mae_mean:.4f}"
        )
        std = (
            "--"
            if not np.isfinite(row.selective_mae_std)
            else f"{row.selective_mae_std:.4f}"
        )
        lines.append(
            f"{row.dataset.title()} & {row.tau_rule} & {row.kappa:.2f} & "
            f"{row.release_mean * 100:.1f}\\% & "
            f"{mae} & {std} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    (TABLES / "table_s4_sensitivity.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    bound_lines = [
        r"\begin{table}",
        r"\caption{Cross-log identifiability and the squared-error "
        r"normalized-RMSE scale implied by the fitted conditional $R^2$ "
        r"for whole-profile recovery. The population-$R^2$ lower-bound "
        r"statement is conditional on the fitted conditional law.}",
        r"\label{tab:identifiability}",
        r"\centering",
        r"\footnotesize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Dataset & Channel & Mean $R^2$ & Lower bound & Range \\",
        r"\midrule",
    ]
    for row in bounds.itertuples(index=False):
        bound_lines.append(
            f"{row.dataset.title()} & {row.feature} & {row.r2_mean:.3f} & "
            f"{row.min_normalized_rmse_mean:.3f} & "
            f"{row.min_normalized_rmse_best:.3f}--"
            f"{row.min_normalized_rmse_worst:.3f} \\\\"
        )
    bound_lines.extend(
        [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    )
    (TABLES / "table4_identifiability.tex").write_text(
        "\n".join(bound_lines), encoding="utf-8"
    )
    print(sensitivity_summary.to_string(index=False))
    print(bounds.to_string(index=False))


if __name__ == "__main__":
    main()
