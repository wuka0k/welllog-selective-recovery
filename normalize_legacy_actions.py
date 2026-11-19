"""Normalize legacy action labels without changing any prediction or metric.

Earlier result files used ``accept``/``recover`` as risk bands even when the
numeric estimate came from the same estimator.  The paper defines actions by
the estimator actually used: non-profile local gaps use SAITS (``accept``),
whole-profile gaps use the cross-log expert (``recover``), and unreleased
estimates use ``defer``.  This script applies only that label correction to
existing raw outputs so a full retraining run is unnecessary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT = Path(__file__).resolve().parents[2] / "outputs"
PROPOSED_ACTIONS = {"accept", "recover", "defer"}


def normalize(path: Path) -> None:
    frame = pd.read_csv(path)
    proposed = frame.action.isin(PROPOSED_ACTIONS)
    released = proposed & (frame.defer == 0)
    frame.loc[released & (frame.condition == "profile"), "action"] = "recover"
    frame.loc[released & (frame.condition != "profile"), "action"] = "accept"
    frame.loc[proposed & (frame.defer == 1), "action"] = "defer"
    if "risk_band" not in frame.columns:
        frame["risk_band"] = "not_recorded_legacy"
    frame.to_csv(path, index=False)
    print(path.name, frame[proposed].action.value_counts().to_dict())


def main() -> None:
    paths = [
        OUT / "welllog_recovery_raw_mid100.csv",
        *sorted(OUT.glob("welllog_recovery_raw_fold[1-4]_*.csv")),
        OUT / "welllog_recovery_raw_full5fold.csv",
    ]
    for path in paths:
        normalize(path)


if __name__ == "__main__":
    main()
