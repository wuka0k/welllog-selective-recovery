"""Generate submission figures and LaTeX tables from frozen evidence files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "evidence"
FIG = PROJECT / "figures"
TAB = PROJECT / "tables"

DATASETS = ["taranaki", "geolink", "teapot"]
DISPLAY = {
    "taranaki": "Taranaki",
    "geolink": "Geolink",
    "teapot": "Teapot",
}
METHODS = ["SAITS", "NoCrossExpert", "Proposed"]
METHOD_LABEL = {
    "SAITS": "SAITS",
    "NoCrossExpert": "Same gate, no expert",
    "Proposed": "Proposed",
}
METHOD_COLOR = {
    "SAITS": "#87939B",
    "NoCrossExpert": "#315B7D",
    "Proposed": "#C54A35",
}
CONDITIONS = ["single", "block20", "block100", "profile"]
COND_LABEL = {
    "single": "Point",
    "block20": "Block-20",
    "block100": "Block-100",
    "profile": "Profile",
}
ACCEPTED_FEATURES = ["RHOB", "NPHI", "DTC"]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def pm(mean: float, std: float | None, digits: int = 4) -> str:
    if not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def pfmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 1e-4:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / (10**exponent)
        return rf"${mantissa:.2f}\times10^{{{exponent}}}$"
    return f"{value:.4f}"


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=220)
    plt.close(fig)


def make_fig2(unit: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.1, 2.35), constrained_layout=True, sharex=True
    )
    for ax, dataset in zip(axes, DATASETS):
        subset = unit[
            (unit.dataset == dataset) & (unit.condition == "profile")
        ]
        folds = np.arange(1, 6)
        for offset, method in zip([-0.13, 0.0, 0.13], METHODS):
            means, errors = [], []
            for fold in folds:
                values = subset[
                    (subset.fold == fold - 1) & (subset.method == method)
                ].selective_mae.to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                means.append(values.mean() if values.size else np.nan)
                errors.append(values.std(ddof=1) if values.size > 1 else 0.0)
            ax.errorbar(
                folds + offset,
                means,
                yerr=errors,
                fmt="o",
                ms=3.2,
                capsize=1.8,
                elinewidth=0.65,
                color=METHOD_COLOR[method],
                label=METHOD_LABEL[method],
            )
        ax.set_title(DISPLAY[dataset], pad=3)
        ax.set_xlabel("Fold")
        ax.set_xticks(folds)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
        ax.set_axisbelow(True)
        if dataset == "teapot":
            ymax = float(
                subset[subset.method == "SAITS"].selective_mae.max()
            )
            ax.set_ylim(ymax * 0.88, ymax * 1.08)
            ax.text(
                0.5,
                0.08,
                "Proposed: all profiles deferred",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7.2,
                bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#B8B8B8", "lw": 0.5},
            )
    axes[0].set_ylabel("Selective MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        frameon=False,
    )
    save_figure(fig, "fig2_profile_recovery")


def make_fig3(unit: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.1, 2.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.2])
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])

    # (a) Profile selective MAE with the same release gate.
    x = np.arange(2)
    width = 0.22
    for index, method in enumerate(METHODS):
        means, errors = [], []
        for dataset in ["taranaki", "geolink"]:
            rows = unit[
                (unit.dataset == dataset)
                & (unit.condition == "profile")
                & (unit.method == method)
            ].selective_mae.to_numpy(dtype=float)
            rows = rows[np.isfinite(rows)]
            means.append(rows.mean())
            errors.append(rows.std(ddof=1))
        ax1.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=errors,
            capsize=1.5,
            color=METHOD_COLOR[method],
            edgecolor="white",
            linewidth=0.45,
            label=METHOD_LABEL[method],
        )
    ax1.set_xticks(x, ["Taranaki", "Geolink"])
    ax1.set_ylabel("Profile selective MAE")
    ax1.set_title("(a) Same-gate ablation", loc="left", pad=3)
    ax1.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax1.set_axisbelow(True)

    # (b) Coverage and interval-outside rate.
    metrics = ["coverage", "outside_rate"]
    metric_label = ["Coverage", "Outside interval"]
    bar_x = np.arange(2)
    width2 = 0.32
    for index, method in enumerate(["NoCrossExpert", "Proposed"]):
        values = []
        errors = []
        for metric in metrics:
            rows = unit[
                (unit.condition == "profile")
                & (unit.method == method)
                & unit.dataset.isin(["taranaki", "geolink"])
            ][metric].to_numpy(dtype=float)
            rows = rows[np.isfinite(rows)]
            values.append(rows.mean())
            errors.append(rows.std(ddof=1))
        ax2.bar(
            bar_x + (index - 0.5) * width2,
            values,
            width2,
            yerr=errors,
            capsize=1.5,
            color=METHOD_COLOR[method],
            edgecolor="white",
            linewidth=0.45,
        )
    ax2.set_xticks(bar_x, metric_label)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Rate")
    ax2.set_title("(b) Risk calibration", loc="left", pad=3)
    ax2.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax2.set_axisbelow(True)
    from matplotlib.patches import Patch


    # (c) Block-100 per-fold delta: positive means proposed is worse.
    fold_subset = unit[unit.condition == "block100"]
    for dataset, marker in [("taranaki", "o"), ("geolink", "s"), ("teapot", "^")]:
        deltas = []
        for fold in range(5):
            block = fold_subset[
                (fold_subset.dataset == dataset) & (fold_subset.fold == fold)
            ]
            proposed = block[block.method == "Proposed"].mae.mean()
            saits = block[block.method == "SAITS"].mae.mean()
            deltas.append(proposed - saits)
        ax3.plot(
            np.arange(1, 6),
            deltas,
            marker=marker,
            ms=3.2,
            color={
                "taranaki": METHOD_COLOR["Proposed"],
                "geolink": METHOD_COLOR["NoCrossExpert"],
                "teapot": METHOD_COLOR["SAITS"],
            }[dataset],
            label=DISPLAY[dataset],
        )
    ax3.axhline(0, color="#444444", linewidth=0.65, linestyle="--")
    ax3.set_xlabel("Fold")
    ax3.set_ylabel(r"$\Delta$MAE (Proposed $-$ SAITS)")
    ax3.set_xticks(np.arange(1, 6))
    ax3.set_title("(c) Block-100 boundary", loc="left", pad=3)
    ax3.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax3.set_axisbelow(True)
    ax3.legend(frameon=False, loc="best")
    fig.legend(
        handles=[
            Patch(color=METHOD_COLOR[method], label=METHOD_LABEL[method])
            for method in METHODS
        ],
        labels=[METHOD_LABEL[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=3,
        frameon=False,
    )
    save_figure(fig, "fig3_mechanism_boundary")


def downstream_seed_average(raw: pd.DataFrame, method: str) -> pd.Series:
    if method == "oracle":
        subset = raw[raw.method == method]
    else:
        subset = raw[
            (raw.method == method) & raw.feature.isin(ACCEPTED_FEATURES)
        ]
    return subset.groupby("seed").macro_f1.mean()


def make_fig4(raw: pd.DataFrame) -> None:
    averages = {
        "Raw missing": downstream_seed_average(raw, "raw_missing"),
        "Mean fill": downstream_seed_average(raw, "mean_fill"),
        "Recovered": downstream_seed_average(raw, "cross_log_all"),
        "Oracle": downstream_seed_average(raw, "oracle"),
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.55), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    ax = axes[0]
    labels = list(averages)
    x = np.arange(len(labels))
    for seed in range(10):
        y = [averages[label].loc[seed] for label in labels]
        ax.plot(
            x,
            y,
            color="#B5B5B5",
            linewidth=0.55,
            alpha=0.75,
            marker="o",
            ms=1.8,
        )
    means = [averages[label].mean() for label in labels]
    ax.plot(
        x,
        means,
        color="#C54A35",
        linewidth=1.8,
        marker="D",
        ms=4,
        label="Mean",
        zorder=5,
    )
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_title("(a) Paired well-disjoint splits", loc="left", pad=3)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    features = ACCEPTED_FEATURES
    x = np.arange(len(features))
    width = 0.34
    for index, (baseline, color, label) in enumerate(
        [
            ("raw_missing", METHOD_COLOR["SAITS"], "vs raw missing"),
            ("mean_fill", METHOD_COLOR["NoCrossExpert"], "vs mean fill"),
        ]
    ):
        deltas = []
        for feature in features:
            cross = raw[
                (raw.method == "cross_log_all") & (raw.feature == feature)
            ].set_index("seed").macro_f1
            base = raw[
                (raw.method == baseline) & (raw.feature == feature)
            ].set_index("seed").macro_f1
            deltas.append(float((cross - base).mean()))
        ax.bar(
            x + (index - 0.5) * width,
            deltas,
            width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )
    ax.axhline(0, color="#444444", linewidth=0.6)
    ax.set_xticks(x, features)
    ax.set_ylabel(r"$\Delta$Macro-F1")
    ax.set_title("(b) Accepted profiles", loc="left", pad=3)
    ax.grid(axis="y", color="#E1E1E1", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, "fig4_downstream")


def table1() -> None:
    rows = [
        ("Teapot", "Boundary", "67 / 19", "3", "5", "5", "5"),
        ("Taranaki", "Positive", "224 / 61", "3", "5", "5", "5"),
        ("Geolink", "Positive", "141 / 36", "4", "5", "5", "5"),
        ("FORCE 2020", "Label audit", "58 / 19 / 21", "4", "10", "--", "--"),
    ]
    body = []
    for dataset, role, wells, logs, folds, seeds, masks in rows:
        body.append(
            f"{dataset} & {role} & {wells} & {logs} & {folds} & {seeds} & {masks} \\\\"
        )
    text = rf"""\begin{{table}}
\caption{{Public datasets and evaluation roles. For Teapot, Taranaki and Geolink,
well counts are training/validation; for FORCE they are
training/calibration/test wells. Seeds and masks apply to the recovery benchmark;
the FORCE audit uses ten well-disjoint splits.}}
\label{{tab:data}}
\centering
\footnotesize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llccccc@{{}}}}
\toprule
Dataset & Role & Wells & Logs & Folds & Seeds & Masks \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    (TAB / "table1_data.tex").write_text(text, encoding="utf-8")


def table2(summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = []
    for dataset in DATASETS:
        for method in METHODS:
            row = summary[
                (summary.dataset == dataset)
                & (summary.condition == "profile")
                & (summary.method == method)
            ].iloc[0]
            test = paired[
                (paired.dataset == dataset)
                & (paired.condition == "profile")
                & (paired.baseline == "SAITS")
                & (paired.metric == "selective_mae")
            ].iloc[0]
            selective = pm(
                row.selective_mae_mean, row.selective_mae_std, 4
            )
            coverage = pm(row.coverage_mean, row.coverage_std, 4)
            outside = pm(row.outside_rate_mean, row.outside_rate_std, 4)
            defer = f"{row.defer_mean * 100:.1f}\\%"
            if method != "Proposed":
                delta = "--"
                holm = "--"
                folds = "--"
            else:
                if int(test.n_fold_seed_pairs) == 0:
                    delta = "--"
                    holm = "--"
                    folds = "--"
                else:
                    delta = f"{test.mean_delta:+.4f}"
                    holm = pfmt(float(test.holm_p))
                    folds = (
                        f"{int(test.folds_favor_proposed)}/"
                        f"{int(test.folds_total)}"
                    )
            if method == "Proposed" and dataset == "teapot":
                selective = "all deferred"
                outside = "--"
            lines.append(
                f"{DISPLAY[dataset]} & {METHOD_LABEL[method]} & {selective} & "
                f"{coverage} & {defer} & {outside} & {delta} & {holm} & {folds} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    text = rf"""\begin{{table}}
\caption{{Whole-profile missingness over five folds, five seeds and five masks.
Values are mean$\pm$standard deviation over fold-seed units. Coverage is the
full-test empirical interval coverage and includes deferred cases in its
denominator. Outside rate is the share of released pattern-feature groups whose
error lies outside the calibrated interval. Holm tests compare fold-seed
selective MAE with SAITS.}}
\label{{tab:profile}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4.2pt}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llccccccc@{{}}}}
\toprule
Dataset & Method & Selective MAE & Coverage & Defer & Outside & $\Delta$ vs SAITS & Holm $p$ & Folds \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    (TAB / "table2_profile.tex").write_text(text, encoding="utf-8")


def table3(raw: pd.DataFrame) -> None:
    method_names = {
        "raw_missing": "Raw missing",
        "mean_fill": "Mean fill",
        "cross_log_all": "Cross-log recovery",
        "oracle": "Oracle complete",
    }
    # Oracle appears once per split; the other methods are averaged over three
    # observability-accepted features within each split.
    per_seed = {}
    for method in method_names:
        if method == "oracle":
            sub = raw[raw.method == method]
            per_seed[method] = sub.groupby("seed")[
                ["accuracy", "macro_f1", "weighted_f1"]
            ].mean()
        else:
            sub = raw[
                (raw.method == method) & raw.feature.isin(ACCEPTED_FEATURES)
            ]
            per_seed[method] = sub.groupby("seed")[
                ["accuracy", "macro_f1", "weighted_f1"]
            ].mean()
    cross = per_seed["cross_log_all"]
    delta_raw = cross.macro_f1 - per_seed["raw_missing"].macro_f1
    delta_mean = cross.macro_f1 - per_seed["mean_fill"].macro_f1
    p_raw = float(stats.wilcoxon(cross.macro_f1, per_seed["raw_missing"].macro_f1).pvalue)
    p_mean = float(stats.wilcoxon(cross.macro_f1, per_seed["mean_fill"].macro_f1).pvalue)
    lines = []
    for method, label in method_names.items():
        table = per_seed[method]
        means = table.mean()
        stds = table.std(ddof=1)
        if method == "oracle":
            d_raw = d_mean = ptext_raw = ptext_mean = "--"
        else:
            if method == "cross_log_all":
                d_raw = f"{delta_raw.mean():+.4f}"
                d_mean = f"{delta_mean.mean():+.4f}"
                ptext_raw = pfmt(p_raw)
                ptext_mean = pfmt(p_mean)
            else:
                d_raw = d_mean = ptext_raw = ptext_mean = "--"
        lines.append(
            f"{label} & {pm(means.accuracy, stds.accuracy, 4)} & "
            f"{pm(means.macro_f1, stds.macro_f1, 4)} & "
            f"{pm(means.weighted_f1, stds.weighted_f1, 4)} & "
            f"{d_raw} & {ptext_raw} & {d_mean} & {ptext_mean} \\\\"
        )
    text = rf"""\begin{{table}}
\caption{{FORCE 2020 lithology audit over ten well-disjoint splits. Results
average the three profiles accepted by the observability gate (RHOB, NPHI and
DTC); GR is deferred in every split. Oracle uses complete four-log inputs.
$\Delta$ and paired Wilcoxon tests compare cross-log recovery with each
baseline over the ten splits.}}
\label{{tab:downstream}}
\centering
\footnotesize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lcccccccc@{{}}}}
\toprule
Input & Accuracy & Macro-F1 & Weighted-F1 &
$\Delta$ raw & $p$ raw & $\Delta$ mean & $p$ mean \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    (TAB / "table3_downstream.tex").write_text(text, encoding="utf-8")


def tables_s1(summary: pd.DataFrame) -> None:
    method_order = [
        "SAITS",
        "IntervalOnly",
        "WidthOnly",
        "NoCrossExpert",
        "Proposed",
    ]
    blocks = []
    for index, dataset in enumerate(DATASETS, start=1):
        closing = r"\clearpage" if index in (1, 3) else ""
        lines = []
        for condition in CONDITIONS:
            for method in method_order:
                row = summary[
                    (summary.dataset == dataset)
                    & (summary.condition == condition)
                    & (summary.method == method)
                ].iloc[0]
                lines.append(
                    f"{COND_LABEL[condition]} & {METHOD_LABEL.get(method, method)} & "
                    f"{pm(row.mae_mean, row.mae_std, 4)} & "
                    f"{pm(row.selective_mae_mean, row.selective_mae_std, 4)} & "
                    f"{pm(row.coverage_mean, row.coverage_std, 4)} & "
                    f"{row.defer_mean * 100:.1f}\\% & "
                    f"{pm(row.outside_rate_mean, row.outside_rate_std, 4)} \\\\"
                )
            lines.append(r"\addlinespace[1.5pt]")
        blocks.append(
            rf"""\begin{{table}}
\caption{{Complete condition-level result for {DISPLAY[dataset]} (S1{chr(96 + index)}).
MAE and selective MAE are reported with standard deviation over fold-seed units;
coverage includes deferred cases in its denominator.}}
\label{{tab:s1{chr(96 + index)}}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llccccc@{{}}}}
\toprule
Pattern & Method & MAE & Selective MAE & Coverage & Defer & Outside \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular*}}
\end{{table}}
{closing}
"""
        )
    (TAB / "table_s1_conditions.tex").write_text(
        "\n".join(blocks), encoding="utf-8"
    )


def tables_s2(paired: pd.DataFrame) -> None:
    selected = paired[
        (paired.metric == "selective_mae")
        & paired.baseline.isin(
            ["SAITS", "IntervalOnly", "WidthOnly", "NoCrossExpert"]
        )
    ].copy()
    blocks = []
    for index, dataset in enumerate(DATASETS, start=1):
        closing = r"\clearpage" if index == 2 else ""
        lines = []
        sub = selected[selected.dataset == dataset]
        for condition in CONDITIONS:
            for baseline in ["SAITS", "IntervalOnly", "WidthOnly", "NoCrossExpert"]:
                row = sub[
                    (sub.condition == condition) & (sub.baseline == baseline)
                ].iloc[0]
                if row.n_fold_seed_pairs == 0:
                    delta = "--"
                    p_raw = p_holm = "--"
                    folds = "--"
                else:
                    delta = f"{row.mean_delta:+.4f}"
                    p_raw = pfmt(float(row.wilcoxon_p))
                    p_holm = pfmt(float(row.holm_p))
                    folds = (
                        f"{int(row.folds_favor_proposed)}/"
                        f"{int(row.folds_total)}"
                    )
                lines.append(
                    f"{COND_LABEL[condition]} & {METHOD_LABEL.get(baseline, baseline)} & "
                    f"{int(row.n_fold_seed_pairs)} & {delta} & {p_raw} & "
                    f"{p_holm} & {float(row.cliffs_delta):.2f} & {folds} \\\\"
                )
            lines.append(r"\addlinespace[1.5pt]")
        blocks.append(
            rf"""\begin{{table}}
\caption{{Selective-MAE paired tests for {DISPLAY[dataset]} (S2{chr(96 + index)}).
Each test uses fold-seed units after averaging five masks; Holm correction is
within dataset and metric. Positive Cliff's delta denotes lower error for the
proposed method. The complete 240-test CSV is distributed with the code.}}
\label{{tab:s2{chr(96 + index)}}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3.6pt}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}llrccccc@{{}}}}
\toprule
Pattern & Baseline & $n$ & $\Delta$ & Wilcoxon $p$ & Holm $p$ & Cliff's $\delta$ & Folds \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular*}}
\end{{table}}
{closing}
"""
        )
    (TAB / "table_s2_paired.tex").write_text(
        "\n".join(blocks), encoding="utf-8"
    )


def table_s3() -> None:
    verdict = json.loads(
        (OUT / "welllog_q2_full5fold_verdict.json").read_text(encoding="utf-8")
    )
    runtime = verdict["runtime"]
    baseline_manifest = json.loads(
        (OUT / "welllog_profile_expert_baselines_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text = rf"""\begin{{table}}
\caption{{Compute budget of the frozen mechanism evaluation on one RTX 3090 Ti.}}
\label{{tab:compute}}
\centering
\footnotesize
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}lr@{{}}}}
\toprule
Quantity & Value \\
\midrule
Fold 0, three datasets & {runtime['fold0_three_datasets_seconds'] / 60:.1f} min \\
Fold 1--2 wall time & {runtime['fold1_2_wall_minutes_approx']} min \\
Fold 3--4 wall time & {runtime['fold3_4_wall_minutes_approx']} min \\
Complete five-fold wall time & {runtime['complete_wall_hours_approx']:.1f} h \\
GPU-task time & {runtime['gpu_task_hours_approx']:.1f} GPU-h \\
FORCE label audit & 83 s \\
Same-gate expert controls & {baseline_manifest['elapsed_seconds'] / 60:.1f} min \\
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    (TAB / "table_s3_compute.tex").write_text(text, encoding="utf-8")


def table5_expert_baselines() -> None:
    summary = pd.read_csv(
        OUT / "welllog_profile_expert_baselines_summary.csv"
    )
    lines = []
    for dataset in ["taranaki", "geolink"]:
        values = {}
        for method in ["Ridge", "RandomForest", "XGBoost", "HGB"]:
            row = summary[
                (summary.dataset == dataset) & (summary.method == method)
            ].iloc[0]
            values[method] = pm(row.mae_mean, row.mae_std, 4)
        lines.append(
            f"{DISPLAY[dataset]} & {values['Ridge']} & "
            f"{values['RandomForest']} & {values['XGBoost']} & "
            f"{values['HGB']} \\\\"
        )
    text = rf"""\begin{{table}}
\caption{{Same-gate profile-expert controls. All methods use the same fit
wells, sampled rows, release decisions and test masks. Values are
mean$\pm$standard deviation of selective MAE over 25 fold-seed units. HGB is
the expert used by the proposed operator.}}
\label{{tab:expert-baselines}}
\centering
\scriptsize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\toprule
Dataset & Ridge & Random Forest & XGBoost & HGB \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""
    (TAB / "table5_expert_baselines.tex").write_text(
        text, encoding="utf-8"
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    configure_style()
    unit = pd.read_csv(OUT / "welllog_recovery_condition_runs_full5fold.csv")
    summary = pd.read_csv(OUT / "welllog_recovery_summary_full5fold.csv")
    paired = pd.read_csv(OUT / "welllog_recovery_paired_full5fold.csv")
    downstream = pd.read_csv(OUT / "welllog_downstream_force_raw.csv")

    make_fig2(unit)
    make_fig3(unit)
    make_fig4(downstream)
    table1()
    table2(summary, paired)
    table3(downstream)
    tables_s1(summary)
    tables_s2(paired)
    table_s3()
    if (OUT / "welllog_profile_expert_baselines_summary.csv").exists():
        table5_expert_baselines()
    print("Generated figures:", sorted(p.name for p in FIG.iterdir()))
    print("Generated tables:", sorted(p.name for p in TAB.iterdir()))


if __name__ == "__main__":
    main()
