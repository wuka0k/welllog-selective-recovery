"""Selective conformal recovery experiment for the well-log benchmark.

The runner uses the authors' official SAITS factory, data split and masking
code, then adds a post-hoc recovery operator. It deliberately keeps the
backbone frozen so the ablation isolates the recovery mechanism.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT.parent / "well_log_imputation"
sys.path.insert(0, str(OFFICIAL))

from data import mask_X  # noqa: E402
from pypots.imputation import SAITS  # noqa: E402
from pypots.optim import Adam  # noqa: E402
from pypots.utils.random import set_random_seed  # noqa: E402


DATA_ROOT = (
    OFFICIAL
    / "preprocessed_data"
    / "pgeoprj"
    / "godeep"
    / "dados"
    / "l2_datasets"
    / "imputation-processed-datasets"
)
OUT = ROOT.parent.parent / "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    features: int
    seeds: int
    epochs: int
    patience: int
    masks_per_seed: int
    alpha: float
    fold: int = 0


CONFIGS = {
    "teapot": ExperimentConfig("teapot", 3, 5, 20, 5, 5, 0.10),
    "taranaki": ExperimentConfig("taranaki", 3, 5, 20, 5, 5, 0.10),
    "geolink": ExperimentConfig("geolink", 4, 5, 20, 5, 5, 0.10),
}


def load_fold(
    name: str, fold: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(DATA_ROOT / f"{name}_fold_{fold}_well_log_sliced_train.npy")
    val = np.load(DATA_ROOT / f"{name}_fold_{fold}_well_log_sliced_val.npy")
    meta_path = DATA_ROOT / f"{name}_fold_{fold}_well_log_slices_meta_train.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wells = np.asarray([row[0] for row in meta])
    return train.astype(np.float32), val.astype(np.float32), wells


def dataset_dict(
    data: np.ndarray,
    mode: str,
    *,
    feature: int | None = None,
    block_size: int = 20,
    n_points: int = 1,
    seed: int | None = None,
    fill: bool = False,
) -> dict[str, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    intact, masked, missing_mask, indicating_mask = mask_X(
        data,
        mode=mode,
        n_points=n_points,
        b_size=[block_size],
        feat=feature,
    )
    if fill:
        masked = masked.copy()
        masked[indicating_mask.astype(bool)] = np.nan
    return {
        "X": masked,
        "X_intact": intact,
        "indicating_mask": indicating_mask,
    }


def make_saits(n_features: int, cfg: ExperimentConfig, seed: int) -> SAITS:
    return SAITS(
        n_steps=256,
        n_features=n_features,
        n_layers=2,
        d_model=256,
        d_inner=128,
        n_heads=4,
        d_k=64,
        d_v=64,
        dropout=0.1,
        attn_dropout=0.1,
        diagonal_attention_mask=True,
        ORT_weight=1,
        MIT_weight=1,
        batch_size=32,
        epochs=cfg.epochs,
        patience=cfg.patience,
        optimizer=Adam(lr=1e-3),
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        saving_path=str(
            ROOT / "models" / f"{cfg.dataset}_fold{cfg.fold}_seed{seed}"
        ),
        model_saving_strategy="best",
    )


def finite_quantile(values: np.ndarray, coverage: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("inf")
    rank = math.ceil((values.size + 1) * coverage)
    rank = min(max(rank, 1), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def masked_errors(
    imputation: np.ndarray,
    test_set: dict[str, np.ndarray],
) -> np.ndarray:
    residual = np.abs(imputation - test_set["X_intact"])
    mask = test_set["indicating_mask"].astype(bool)
    return residual[mask]


def apply_cross_log_recovery(
    prediction: np.ndarray,
    observed: np.ndarray,
    feature: int,
    model: HistGradientBoostingRegressor,
) -> np.ndarray:
    recovered = prediction.copy()
    features = observed.shape[-1]
    others = [j for j in range(features) if j != feature]
    values = model.predict(observed[:, :, others].reshape(-1, features - 1))
    recovered[:, :, feature] = values.reshape(observed.shape[:2])
    return recovered


def fit_cross_log_experts(
    fit_data: np.ndarray,
    cal_data: np.ndarray,
    features: int,
    seed: int,
) -> tuple[list[HistGradientBoostingRegressor], dict[int, dict[str, float]]]:
    """Conditional cross-log recovery experts and held-out identifiability."""
    models = []
    result = {}
    flat_fit = fit_data.reshape(-1, features)
    flat_cal = cal_data.reshape(-1, features)
    rng = np.random.default_rng(seed + 91)
    if flat_fit.shape[0] > 30_000:
        flat_fit = flat_fit[
            rng.choice(flat_fit.shape[0], 30_000, replace=False)
        ]
    for feature in range(features):
        others = [j for j in range(features) if j != feature]
        model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed + feature,
        )
        model.fit(flat_fit[:, others], flat_fit[:, feature])
        pred = model.predict(flat_cal[:, others])
        r2 = r2_score(flat_cal[:, feature], pred)
        std = float(np.std(flat_fit[:, feature]))
        result[feature] = {
            "cross_log_r2": float(max(r2, -1.0)),
            "target_std": std,
            "condition_proxy": float(1.0 - max(r2, 0.0)),
        }
        models.append(model)
    return models, result


def run_seed(
    name: str,
    cfg: ExperimentConfig,
    seed: int,
    train: np.ndarray,
    val: np.ndarray,
    wells: np.ndarray,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed + 17076)
    unique_wells = np.unique(wells)
    well_order = rng.permutation(unique_wells)
    cut = int(0.8 * len(well_order))
    fit_wells = set(well_order[:cut])
    cal_wells = set(well_order[cut:])
    fit_index = np.asarray([w in fit_wells for w in wells])
    cal_index = np.asarray([w in cal_wells for w in wells])
    fit_data = train[fit_index]
    cal_data = train[cal_index]
    experts, diagnostics = fit_cross_log_experts(
        fit_data, cal_data, cfg.features, seed
    )

    set_random_seed(seed + 17076)
    model = make_saits(cfg.features, cfg, seed)
    train_set = dataset_dict(
        fit_data, "rand", n_points=1, block_size=20, seed=seed + 100
    )
    val_set = dataset_dict(
        val, "rand", n_points=1, block_size=20, seed=seed + 200, fill=True
    )
    train_start = time.perf_counter()
    model.fit(train_set=train_set, val_set=val_set)
    train_seconds = time.perf_counter() - train_start

    conditions = [
        ("single", -1, 0),
        ("block", 20, 0),
        ("block", 100, 0),
        ("profile", -1, 0),
    ]
    # Disambiguate the two block lengths in output tables.
    condition_names = ["single", "block20", "block100", "profile"]

    # Pattern/feature-specific split-conformal radii on held-out training wells.
    radii: dict[tuple[str, int], float] = {}
    calibration_counts: dict[str, int] = {}
    for cname, (mode, bsize, _) in zip(condition_names, conditions):
        for feature in range(cfg.features):
            errors = []
            for rep in range(2):
                test = dataset_dict(
                    cal_data,
                    mode,
                    feature=feature if mode == "profile" else feature,
                    block_size=bsize if mode == "block" else 20,
                    seed=seed * 1000 + rep * 37 + feature + 11,
                    fill=True,
                )
                pred = model.predict(test)["imputation"]
                if mode == "profile":
                    pred = apply_cross_log_recovery(
                        pred, cal_data, feature, experts[feature]
                    )
                errors.append(masked_errors(pred, test))
            pooled = np.concatenate(errors)
            radii[(cname, feature)] = finite_quantile(pooled, 1.0 - cfg.alpha)
            calibration_counts[f"{cname}:{feature}"] = int(pooled.size)
    # Rank recovery risk in the common normalized log units used by the
    # benchmark. Absolute conformal radii avoid deferring a low-amplitude log
    # merely because its channel variance is small.
    calibration_scores = np.array([radius for radius in radii.values()])
    tau_accept = float(np.quantile(calibration_scores, 0.65))
    tau_recover = float(np.quantile(calibration_scores, 0.95))

    rows = []
    prediction_seconds = 0.0
    with torch.no_grad():
        for rep in range(cfg.masks_per_seed):
            for cname, (mode, bsize, _) in zip(condition_names, conditions):
                for feature in range(cfg.features):
                    mask_seed = seed * 100_000 + rep * 1_000 + feature * 13
                    test = dataset_dict(
                        val,
                        mode,
                        feature=feature,
                        block_size=bsize if mode == "block" else 20,
                        seed=mask_seed,
                        fill=True,
                    )
                    begin = time.perf_counter()
                    pred_saits = model.predict(test)["imputation"]
                    prediction_seconds += time.perf_counter() - begin
                    pred = pred_saits
                    if mode == "profile":
                        pred = apply_cross_log_recovery(
                            pred, val, feature, experts[feature]
                        )
                    errors = masked_errors(pred, test)
                    errors_saits = masked_errors(pred_saits, test)
                    radius = radii[(cname, feature)]
                    std = diagnostics[feature]["target_std"]
                    width_norm = 2.0 * radius / max(std, 1e-8)

                    # The action name describes the estimator actually used,
                    # while the radius determines whether that estimate is
                    # released. Profile loss with weak cross-log predictability
                    # cannot be assigned a point-value recovery guarantee.
                    structural_defer = (
                        mode == "profile"
                        and diagnostics[feature]["cross_log_r2"] < 0.50
                    )
                    no_obs_defer = radius > tau_recover
                    recover_flag = (not structural_defer) and radius > tau_accept
                    accepted = not (structural_defer or no_obs_defer)
                    action = (
                        "defer"
                        if not accepted
                        else ("recover" if mode == "profile" else "accept")
                    )
                    risk_band = (
                        "defer"
                        if not accepted
                        else ("upper-release-band" if recover_flag else "lower-release-band")
                    )
                    outside = errors > radius

                    rows.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "mask_rep": rep,
                            "condition": cname,
                            "feature": feature,
                            "cross_log_r2": diagnostics[feature]["cross_log_r2"],
                            "normalized_width": width_norm,
                            "action": action,
                            "risk_band": risk_band,
                            "accept_or_recover": int(accepted),
                            "defer": int(not accepted),
                            "n_points": int(errors.size),
                            "mae": float(errors.mean()),
                            "selective_mae": float(errors.mean()) if accepted else np.nan,
                            "coverage": float((~outside).mean()),
                            # Legacy serialized name. This is the fraction of
                            # accepted pattern-feature groups whose errors lie
                            # outside the conformal interval, not a binary
                            # correctness label or decision error rate.
                            "false_recovery": float(outside.mean()) if accepted else np.nan,
                            "outside_rate": float(outside.mean()),
                        }
                    )

                    # Decision-only ablation: use SAITS point values but the
                    # same observability-aware gate as the full method.
                    outside_saits = errors_saits > radius
                    rows.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "mask_rep": rep,
                            "condition": cname,
                            "feature": feature,
                            "cross_log_r2": diagnostics[feature]["cross_log_r2"],
                            "normalized_width": width_norm,
                            "action": "no_cross_expert",
                            "risk_band": risk_band,
                            "accept_or_recover": int(accepted),
                            "defer": int(not accepted),
                            "n_points": int(errors_saits.size),
                            "mae": float(errors_saits.mean()),
                            "selective_mae": float(errors_saits.mean())
                            if accepted
                            else np.nan,
                            "coverage": float((~outside_saits).mean()),
                            "false_recovery": float(outside_saits.mean())
                            if accepted
                            else np.nan,
                            "outside_rate": float(outside_saits.mean()),
                        }
                    )

                    # Ablations use exactly the same predictions and masks.
                    fixed_accepted = True
                    width_based_accepted = not (structural_defer or no_obs_defer)
                    rows.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "mask_rep": rep,
                            "condition": cname,
                            "feature": feature,
                            "cross_log_r2": diagnostics[feature]["cross_log_r2"],
                            "normalized_width": width_norm,
                            "action": "interval_only",
                            "risk_band": "full-release",
                            "accept_or_recover": int(fixed_accepted),
                            "defer": int(not fixed_accepted),
                            "n_points": int(errors.size),
                            "mae": float(errors.mean()),
                            "selective_mae": float(errors.mean()) if fixed_accepted else np.nan,
                            "coverage": float((~outside).mean()),
                            "false_recovery": float(outside.mean())
                            if fixed_accepted
                            else np.nan,
                            "outside_rate": float(outside.mean()),
                        }
                    )
                    rows.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "mask_rep": rep,
                            "condition": cname,
                            "feature": feature,
                            "cross_log_r2": diagnostics[feature]["cross_log_r2"],
                            "normalized_width": width_norm,
                            "action": "width_only",
                            "risk_band": (
                                "defer"
                                if not width_based_accepted
                                else (
                                    "upper-release-band"
                                    if recover_flag
                                    else "lower-release-band"
                                )
                            ),
                            "accept_or_recover": int(width_based_accepted),
                            "defer": int(not width_based_accepted),
                            "n_points": int(errors.size),
                            "mae": float(errors.mean()),
                            "selective_mae": float(errors.mean())
                            if width_based_accepted
                            else np.nan,
                            "coverage": float((~outside).mean()),
                            "false_recovery": float(outside.mean())
                            if width_based_accepted
                            else np.nan,
                            "outside_rate": float(outside.mean()),
                        }
                    )
                    rows.append(
                        {
                            "dataset": name,
                            "seed": seed,
                            "mask_rep": rep,
                            "condition": cname,
                            "feature": feature,
                            "cross_log_r2": diagnostics[feature]["cross_log_r2"],
                            "normalized_width": width_norm,
                            "action": "saits_point",
                            "risk_band": "full-release",
                            "accept_or_recover": 1,
                            "defer": 0,
                            "n_points": int(errors_saits.size),
                            "mae": float(errors_saits.mean()),
                            "selective_mae": float(errors_saits.mean()),
                            "coverage": np.nan,
                            "false_recovery": np.nan,
                            "outside_rate": np.nan,
                        }
                    )

    meta = {
        "dataset": name,
        "seed": seed,
        "train_sequences": int(len(fit_data)),
        "calibration_sequences": int(len(cal_data)),
        "validation_sequences": int(len(val)),
        "train_seconds": train_seconds,
        "prediction_seconds": prediction_seconds,
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
        "radii": {f"{k[0]}:{k[1]}": v for k, v in radii.items()},
        "calibration_counts": calibration_counts,
        "tau_accept": tau_accept,
        "tau_recover": tau_recover,
        "diagnostics": {str(k): v for k, v in diagnostics.items()},
    }
    return rows, meta


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "action", "condition"]
    grouped = raw.groupby(keys, as_index=False).agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        selective_mae_mean=("selective_mae", "mean"),
        selective_mae_std=("selective_mae", "std"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        defer_mean=("defer", "mean"),
        false_recovery_mean=("false_recovery", "mean"),
        normalized_width_mean=("normalized_width", "mean"),
    )
    return grouped


def paired_tests(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae", "selective_mae", "coverage", "false_recovery"]
    rows = []
    for dataset in raw.dataset.unique():
        for condition in raw.condition.unique():
            full = raw[
                (raw.dataset == dataset)
                & (raw.condition == condition)
                & (raw.action.isin(["accept", "recover", "defer"]))
            ]
            for baseline in [
                "saits_point",
                "interval_only",
                "width_only",
                "no_cross_expert",
            ]:
                other = raw[
                    (raw.dataset == dataset)
                    & (raw.condition == condition)
                    & (raw.action == baseline)
                ]
                pair = full.merge(
                    other,
                    on=["seed", "mask_rep", "feature"],
                    suffixes=("_full", "_base"),
                )
                for metric in metrics:
                    a = pair[f"{metric}_full"].to_numpy(dtype=float)
                    b = pair[f"{metric}_base"].to_numpy(dtype=float)
                    valid = np.isfinite(a) & np.isfinite(b)
                    a, b = a[valid], b[valid]
                    if len(a) == 0 or np.allclose(a, b):
                        p = 1.0
                    else:
                        try:
                            p = float(stats.wilcoxon(a, b).pvalue)
                        except ValueError:
                            p = 1.0
                    delta = a - b
                    rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "baseline": baseline,
                            "metric": metric,
                            "n_pairs": len(a),
                            "mean_delta": float(delta.mean()) if len(a) else np.nan,
                            "wilcoxon_p": p,
                            "cliffs_delta": float(
                                ((delta < 0).sum() - (delta > 0).sum())
                                / max(((delta < 0).sum() + (delta > 0).sum()), 1)
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    result["holm_p"] = 1.0
    for _, loc in result.groupby(["dataset", "metric"]).groups.items():
        subset = result.loc[loc].sort_values("wilcoxon_p")
        vals = []
        running = 0.0
        for rank, (idx, row) in enumerate(subset.iterrows()):
            running = max(running, min(1.0, (len(subset) - rank) * row.wilcoxon_p))
            vals.append((idx, running))
        for idx, val in vals:
            result.loc[idx, "holm_p"] = val
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["teapot", "taranaki"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--masks-per-seed", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    all_rows = []
    metas = []
    overall_start = time.perf_counter()
    for name in args.datasets:
        base = CONFIGS[name]
        cfg = ExperimentConfig(
            base.dataset,
            base.features,
            args.seeds,
            args.epochs,
            args.patience,
            args.masks_per_seed,
            base.alpha,
            args.fold,
        )
        train, val, wells = load_fold(name, args.fold)
        print(f"{name}: train={train.shape}, val={val.shape}", flush=True)
        for seed in range(cfg.seeds):
            begin = time.perf_counter()
            rows, meta = run_seed(name, cfg, seed, train, val, wells)
            all_rows.extend(rows)
            metas.append(meta)
            print(
                name,
                seed,
                f"seconds={time.perf_counter()-begin:.1f}",
                f"official_train={meta['train_seconds']:.1f}",
                flush=True,
            )
    raw = pd.DataFrame(all_rows)
    suffix = f"_{args.suffix}" if args.suffix else ""
    raw.to_csv(OUT / f"welllog_recovery_raw{suffix}.csv", index=False)
    summary = aggregate(raw)
    summary.to_csv(OUT / f"welllog_recovery_summary{suffix}.csv", index=False)
    tests = paired_tests(raw)
    tests.to_csv(OUT / f"welllog_recovery_paired_tests{suffix}.csv", index=False)
    manifest = {
        "datasets": args.datasets,
        "seeds": args.seeds,
        "masks_per_seed": args.masks_per_seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "alpha": 0.10,
        "fold": args.fold,
        "elapsed_seconds": time.perf_counter() - overall_start,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "runs": metas,
    }
    (OUT / f"welllog_recovery_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
