"""Measure SAITS and cross-log recovery inference latency."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from run_recovery_experiment import (
    CONFIGS,
    OUT,
    dataset_dict,
    fit_cross_log_experts,
    load_fold,
    make_saits,
)


def median_latency(fn, repeats: int = 8) -> dict[str, float]:
    fn()
    values = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        begin = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000.0)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    result = {}
    for name in ["taranaki", "geolink"]:
        train, _, wells = load_fold(name)
        cfg = CONFIGS[name]
        model = make_saits(cfg.features, cfg, 0)
        batch = dataset_dict(
            train[:64], "profile", feature=0, seed=11, fill=True
        )
        saits = median_latency(lambda: model.predict(batch)["imputation"])

        fit = train[: min(1500, len(train))]
        cal = train[min(1500, len(train)) : min(2000, len(train))]
        experts, _ = fit_cross_log_experts(fit, cal, cfg.features, 0)
        observed = train[:64, :, 1:]
        cross = median_latency(
            lambda: experts[0].predict(
                observed.reshape(-1, cfg.features - 1)
            )
        )
        result[name] = {
            "sequence_batch": 64,
            "steps_per_sequence": 256,
            "features": cfg.features,
            "saits": saits,
            "cross_log_hist_gradient_boosting": cross,
        }
    (OUT / "welllog_recovery_latency.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
