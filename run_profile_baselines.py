"""Compare cross-log profile experts under the frozen release decisions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


HERE = Path(__file__).resolve()
LOCAL_EVIDENCE = HERE.parents[1] / "evidence"
DEFAULT_EVIDENCE = (
    LOCAL_EVIDENCE if LOCAL_EVIDENCE.exists() else HERE.parents[2] / "outputs"
)
OUT = Path(os.environ.get("WELLLOG_EVIDENCE_ROOT", DEFAULT_EVIDENCE))
DEFAULT_OFFICIAL = Path(
    r"C:\Users\HAO\Documents\Codex\2026-11-14\ftncbf-q2-execution"
    r"\work\well_log_imputation"
)
OFFICIAL = Path(os.environ.get("WELLLOG_BENCHMARK_ROOT", DEFAULT_OFFICIAL))
sys.path.insert(0, str(OFFICIAL))

from data import mask_X  # noqa: E402


DATA_ROOT = (
    OFFICIAL
    / "preprocessed_data"
    / "pgeoprj"
    / "godeep"
    / "dados"
    / "l2_datasets"
    / "imputation-processed-datasets"
)
CONFIGS = {"teapot": 3, "taranaki": 3, "geolink": 4}
ACCEPTED_ACTIONS = {"accept", "recover"}


@dataclass(frozen=True)
class Config:
    dataset: str
    features: int
    fold: int
    seed: int


def load_fold(name: str, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(DATA_ROOT / f"{name}_fold_{fold}_well_log_sliced_train.npy")
    val = np.load(DATA_ROOT / f"{name}_fold_{fold}_well_log_sliced_val.npy")
    meta = json.loads(
        (
            DATA_ROOT / f"{name}_fold_{fold}_well_log_slices_meta_train.json"
        ).read_text(encoding="utf-8")
    )
    wells = np.asarray([row[0] for row in meta])
    return train.astype(np.float32), val.astype(np.float32), wells


def profile_mask(
    data: np.ndarray, feature: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    intact, masked, _, indicating = mask_X(
        data, mode="profile", n_points=1, b_size=[20], feat=feature
    )
    masked = masked.copy()
    masked[indicating.astype(bool)] = np.nan
    return intact, masked, indicating


def make_model(method: str, seed: int):
    if method == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if method == "RandomForest":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=20,
            min_samples_leaf=4,
            max_features=1.0,
            n_jobs=-1,
            random_state=seed,
        )
    if method == "XGBoost":
        return XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if method == "HGB":
        return HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        )
    raise ValueError(method)


def accepted_map(raw: pd.DataFrame) -> dict[tuple[str, int, int, int], bool]:
    proposed = raw[
        raw.action.isin(["accept", "recover", "defer"])
        & (raw.condition == "profile")
    ]
    result = {}
    for row in proposed.itertuples():
        result[
            (row.dataset, int(row.fold), int(row.seed), int(row.feature))
        ] = row.action in ACCEPTED_ACTIONS
    return result


def run() -> None:
    begin = time.perf_counter()
    raw_existing = pd.read_csv(
        OUT / "welllog_recovery_raw_full5fold.csv"
    )
    accepted = accepted_map(raw_existing)
    rows = []
    methods = ["Ridge", "RandomForest", "XGBoost", "HGB"]
    for name, n_features in CONFIGS.items():
        for fold in range(5):
            train, val, wells = load_fold(name, fold)
            for seed in range(5):
                cfg = Config(name, n_features, fold, seed)
                rng = np.random.default_rng(seed + 17076)
                unique_wells = np.unique(wells)
                order = rng.permutation(unique_wells)
                cut = int(0.8 * len(order))
                fit_wells = set(order[:cut])
                fit_index = np.asarray([well in fit_wells for well in wells])
                fit_data = train[fit_index]
                flat_fit = fit_data.reshape(-1, cfg.features)
                sample_rng = np.random.default_rng(seed + 91)
                if flat_fit.shape[0] > 30_000:
                    flat_fit = flat_fit[
                        sample_rng.choice(
                            flat_fit.shape[0], 30_000, replace=False
                        )
                    ]
                for feature in range(cfg.features):
                    decisions = [
                        accepted.get((name, fold, seed, feature), False)
                        for _ in range(5)
                    ]
                    if not any(decisions):
                        continue
                    others = [j for j in range(cfg.features) if j != feature]
                    for method in methods:
                        model = make_model(method, seed + feature)
                        model.fit(flat_fit[:, others], flat_fit[:, feature])
                        for rep in range(5):
                            mask_seed = seed * 100_000 + rep * 1_000 + feature * 13
                            intact, masked, indicating = profile_mask(
                                val, feature, mask_seed
                            )
                            prediction = model.predict(
                                masked[:, :, others].reshape(
                                    -1, cfg.features - 1
                                )
                            ).reshape(masked.shape[:2])
                            selected = indicating[:, :, feature].astype(bool)
                            errors = np.abs(
                                prediction[selected]
                                - intact[:, :, feature][selected]
                            )
                            rows.append(
                                {
                                    "dataset": name,
                                    "fold": fold,
                                    "seed": seed,
                                    "mask_rep": rep,
                                    "feature": feature,
                                    "method": method,
                                    "accepted_by_proposed_gate": int(
                                        decisions[rep]
                                    ),
                                    "n_points": int(errors.size),
                                    "mae": float(errors.mean()),
                                }
                            )
                print(
                    f"{name} fold={fold} seed={seed} "
                    f"elapsed={time.perf_counter() - begin:.1f}s",
                    flush=True,
                )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "welllog_profile_expert_baselines_raw.csv", index=False)

    units = []
    for (dataset, fold, seed, method), block in raw[
        raw.accepted_by_proposed_gate == 1
    ].groupby(["dataset", "fold", "seed", "method"]):
        units.append(
            {
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "method": method,
                "mae": np.average(block.mae, weights=block.n_points),
            }
        )
    unit = pd.DataFrame(units)
    unit.to_csv(OUT / "welllog_profile_expert_baselines_units.csv", index=False)
    summary = (
        unit.groupby(["dataset", "method"], as_index=False)
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), n_units=("mae", "size"))
    )
    summary.to_csv(OUT / "welllog_profile_expert_baselines_summary.csv", index=False)

    paired = []
    for dataset in sorted(unit.dataset.unique()):
        base = unit[
            (unit.dataset == dataset) & (unit.method == "HGB")
        ].set_index(["fold", "seed"])
        for method in ["Ridge", "RandomForest", "XGBoost"]:
            other = unit[
                (unit.dataset == dataset) & (unit.method == method)
            ].set_index(["fold", "seed"])
            joined = base.join(other, lsuffix="_hgb", rsuffix="_other")
            delta = joined.mae_other - joined.mae_hgb
            p = (
                1.0
                if np.allclose(delta, 0)
                else float(stats.wilcoxon(delta).pvalue)
            )
            lower = int((delta < 0).sum())
            upper = int((delta > 0).sum())
            paired.append(
                {
                    "dataset": dataset,
                    "baseline": method,
                    "n_pairs": len(delta),
                    "mean_delta_vs_hgb": float(delta.mean()),
                    "wins_vs_hgb": lower,
                    "losses_vs_hgb": upper,
                    "wilcoxon_p": p,
                    "cliffs_delta": (lower - upper) / max(lower + upper, 1),
                }
            )
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(
        OUT / "welllog_profile_expert_baselines_paired.csv", index=False
    )
    manifest = {
        "comparison": "same proposed-gate accepted units and test masks",
        "methods": methods,
        "datasets": sorted(raw.dataset.unique().tolist()),
        "folds": [0, 1, 2, 3, 4],
        "seeds": [0, 1, 2, 3, 4],
        "masks_per_seed": 5,
        "elapsed_seconds": time.perf_counter() - begin,
        "note": (
            "Ridge, RandomForest and XGBoost are model-family controls. "
            "They use the same fit wells, release decisions and masks as HGB."
        ),
    }
    (OUT / "welllog_profile_expert_baselines_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(paired_frame.to_string(index=False))
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    run()
