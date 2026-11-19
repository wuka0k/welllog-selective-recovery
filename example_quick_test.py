"""Fast self-contained check of conformal calibration and recovery routing."""

from __future__ import annotations

import math


def conformal_radius(residuals: list[float], coverage: float) -> float:
    if not residuals:
        return math.inf
    ordered = sorted(float(value) for value in residuals)
    rank = math.ceil((len(ordered) + 1) * coverage)
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def recovery_action(
    radius: float,
    release_limit: float,
    profile_missing: bool,
    r2: float,
    identifiability_limit: float = 0.5,
) -> str:
    if radius > release_limit:
        return "defer"
    if profile_missing and r2 < identifiability_limit:
        return "defer"
    if profile_missing:
        return "recover"
    return "accept"


def main() -> None:
    residuals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    radius = conformal_radius(residuals, 0.90)
    assert radius == 0.9
    assert recovery_action(0.2, 0.8, False, 0.1) == "accept"
    assert recovery_action(0.2, 0.8, True, 0.8) == "recover"
    assert recovery_action(0.2, 0.8, True, 0.2) == "defer"
    assert recovery_action(0.9, 0.8, False, 0.8) == "defer"
    print("quick test passed")


if __name__ == "__main__":
    main()
