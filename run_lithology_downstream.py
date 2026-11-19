"""Downstream lithology classification audit on the public FORCE 2020 data."""

from __future__ import annotations

import json
import hashlib
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent.parent / "outputs"
FORCE_ZIP = ROOT.parent / "force_data" / "train.zip"
LOGS = ["GR", "RHOB", "NPHI", "DTC"]
LABEL = "FORCE_2020_LITHOFACIES_LITHOLOGY"


def load_force() -> pd.DataFrame:
    with zipfile.ZipFile(FORCE_ZIP) as archive:
        data = pd.read_csv(archive.open("train.csv"), sep=";")
    data = data[data[LABEL] != 93000].copy()
    data = data.dropna(subset=LOGS + [LABEL])
    encoder = LabelEncoder()
    data[LABEL] = encoder.fit_transform(data[LABEL].astype(int))
    data.attrs["lithology_classes"] = encoder.classes_.tolist()
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_rows(
    data: pd.DataFrame, maximum: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(data) > maximum:
        data = data.sample(maximum, random_state=seed)
    labels = data[LABEL].astype(int).to_numpy()
    return data, labels


def fit_expert(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature: str,
    seed: int,
) -> tuple[HistGradientBoostingRegressor, dict]:
    others = [log for log in LOGS if log != feature]
    fit_rows = train.sample(min(30_000, len(train)), random_state=seed)
    cal_rows = calibration.sample(min(10_000, len(calibration)), random_state=seed)
    model = HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=seed,
    )
    model.fit(fit_rows[others], fit_rows[feature])
    cal_pred = model.predict(cal_rows[others])
    r2 = r2_score(cal_rows[feature], cal_pred)
    residual = np.abs(cal_rows[feature].to_numpy() - cal_pred)
    rank = math.ceil((len(residual) + 1) * 0.90)
    rank = min(max(rank, 1), len(residual))
    radius = float(np.partition(residual, rank - 1)[rank - 1])
    return model, {
        "feature": feature,
        "cross_log_r2": float(r2),
        "conformal_radius_90": radius,
    }


def metrics(y: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=range(n_classes), zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def main() -> None:
    start = time.perf_counter()
    data = load_force()
    wells = np.asarray(sorted(data.WELL.unique()))
    n_classes = int(data[LABEL].nunique())
    rows = []
    splits = []
    n_seeds = 10
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed + 4001)
        n = len(wells)
        # Guarantee that every rare class has at least one training well.
        train_wells = set()
        for label in sorted(data[LABEL].unique()):
            candidates = data.loc[data[LABEL] == label, "WELL"].unique()
            train_wells.add(str(rng.choice(candidates)))
        remaining = np.asarray([w for w in wells if w not in train_wells])
        order = rng.permutation(remaining)
        train_target = int(0.60 * n)
        add_train = max(0, train_target - len(train_wells))
        train_wells.update(order[:add_train])
        rest = order[add_train:]
        cal_count = max(1, int(0.20 * n))
        cal_wells = set(rest[:cal_count])
        test_wells = set(rest[cal_count:])
        train = data[data.WELL.isin(train_wells)]
        cal = data[data.WELL.isin(cal_wells)]
        test = data[data.WELL.isin(test_wells)]

        train_rows, y_train = sample_rows(train, 100_000, seed)
        test_rows, y_test = sample_rows(test, 20_000, seed)
        class_weight = compute_sample_weight(
            class_weight="balanced", y=y_train
        )
        classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=n_classes,
            tree_method="hist",
            device="cpu",
            n_jobs=-1,
            random_state=seed,
        )
        classifier.fit(
            train_rows[LOGS],
            y_train,
            sample_weight=class_weight,
        )

        oracle_pred = classifier.predict(test_rows[LOGS])
        oracle_metrics = metrics(y_test, oracle_pred, n_classes)
        rows.append(
            {
                "seed": seed,
                "feature": "none",
                "method": "oracle",
                "status": "accept",
                **oracle_metrics,
                "n_rows": len(test_rows),
            }
        )

        expert_meta = {}
        for feature in LOGS:
            expert, meta = fit_expert(train, cal, feature, seed)
            expert_meta[feature] = meta
            others = [log for log in LOGS if log != feature]

            raw = test_rows[LOGS].copy()
            raw[feature] = np.nan
            raw_pred = classifier.predict(raw)
            rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "method": "raw_missing",
                    "status": "accept",
                    **metrics(y_test, raw_pred, n_classes),
                    "n_rows": len(test_rows),
                }
            )

            mean_fill = test_rows[LOGS].copy()
            mean_fill[feature] = float(train_rows[feature].mean())
            mean_pred = classifier.predict(mean_fill)
            rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "method": "mean_fill",
                    "status": "accept",
                    **metrics(y_test, mean_pred, n_classes),
                    "n_rows": len(test_rows),
                }
            )

            recovered = test_rows[LOGS].copy()
            recovered[feature] = expert.predict(test_rows[others])
            recovered_pred = classifier.predict(recovered)
            recovered_metrics = metrics(y_test, recovered_pred, n_classes)
            residual = np.abs(
                recovered[feature].to_numpy()
                - test_rows[feature].to_numpy()
            )
            coverage = float((residual <= meta["conformal_radius_90"]).mean())
            accepted = meta["cross_log_r2"] >= 0.50
            rows.append(
                {
                    "seed": seed,
                    "feature": feature,
                    "method": "cross_log_all",
                    "status": "accept",
                    **recovered_metrics,
                    "n_rows": len(test_rows),
                    "recovery_coverage": coverage,
                    "cross_log_r2": meta["cross_log_r2"],
                    "accepted_by_observability": int(accepted),
                }
            )
            if accepted:
                rows.append(
                    {
                        "seed": seed,
                        "feature": feature,
                        "method": "proposed",
                        "status": "accept",
                        **recovered_metrics,
                        "n_rows": len(test_rows),
                        "recovery_coverage": coverage,
                        "cross_log_r2": meta["cross_log_r2"],
                        "accepted_by_observability": 1,
                    }
                )
        splits.append(
            {
                "seed": seed,
                "train_wells": len(train_wells),
                "calibration_wells": len(cal_wells),
                "test_wells": len(test_wells),
                "train_rows_sampled": len(train_rows),
                "test_rows_sampled": len(test_rows),
                "expert_metadata": expert_meta,
            }
        )
        print(seed, oracle_metrics, {k: round(v["cross_log_r2"], 3) for k, v in expert_meta.items()}, flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "welllog_downstream_force_raw.csv", index=False)
    summary = raw.groupby(["method", "feature"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        rows_per_seed=("n_rows", "first"),
        recovery_coverage_mean=("recovery_coverage", "mean"),
        cross_log_r2_mean=("cross_log_r2", "mean"),
    )
    summary.to_csv(OUT / "welllog_downstream_force_summary.csv", index=False)

    # Compare cross-log against raw missingness on the same seed/feature pairs.
    paired_rows = []
    for feature in LOGS:
        cross = raw[
            (raw.feature == feature) & (raw.method == "cross_log_all")
        ].set_index("seed")
        for baseline in ["raw_missing", "mean_fill"]:
            other = raw[
                (raw.feature == feature) & (raw.method == baseline)
            ].set_index("seed")
            for metric in ["accuracy", "macro_f1", "weighted_f1"]:
                delta = cross[metric].to_numpy() - other[metric].to_numpy()
                value = {
                    "feature": feature,
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(delta),
                    "mean_delta": float(delta.mean()),
                    "wins": int((delta > 0).sum()),
                }
                try:
                    from scipy.stats import wilcoxon

                    value["wilcoxon_p"] = (
                        float(wilcoxon(cross[metric], other[metric]).pvalue)
                        if not np.allclose(delta, 0)
                        else 1.0
                    )
                except ValueError:
                    value["wilcoxon_p"] = 1.0
                paired_rows.append(value)
    pd.DataFrame(paired_rows).to_csv(
        OUT / "welllog_downstream_force_paired.csv", index=False
    )
    manifest = {
        "dataset": "FORCE 2020 lithology",
        "source": (
            "https://github.com/bolgebrygg/"
            "Force-2020-Machine-Learning-competition/"
            "tree/master/lithology_competition/data"
        ),
        "archive_sha256": sha256(FORCE_ZIP),
        "seeds": n_seeds,
        "wells": len(wells),
        "rows_complete": len(data),
        "classes": n_classes,
        "splits": splits,
        "elapsed_seconds": time.perf_counter() - start,
        "note": (
            "Geolink raw labels were blocked by the Google Drive network; "
            "this is an independent FORCE label audit, not a Geolink label reproduction."
        ),
    }
    (OUT / "welllog_downstream_force_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps({"elapsed_seconds": manifest["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
