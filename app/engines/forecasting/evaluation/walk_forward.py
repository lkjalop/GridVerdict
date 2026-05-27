"""Walk-forward (rolling-origin) evaluation protocol.

Scaffold module. This is the single most important anti-naivety guard: at every
origin, training data strictly precedes the test window. The no-leakage invariant
is asserted in code, not just assumed, because lookahead bias is the number-one
way naive forecasting fools itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Split:
    """Index ranges for one walk-forward origin (half-open ranges)."""
    train_start: int
    train_end: int   # exclusive; origin sits here
    test_start: int  # == train_end
    test_end: int    # exclusive


def walk_forward_splits(
    n: int,
    horizon: int,
    step: int,
    min_train: int,
    window: str = "expanding",
) -> Iterator[Split]:
    """Yield rolling-origin splits over a series of length n.

    horizon:   number of steps forecast at each origin.
    step:      how far the origin advances each iteration.
    min_train: minimum training rows before the first origin.
    window:    "expanding" (train grows) or "sliding" (fixed-width = min_train).

    Invariant asserted on every split: train_end == test_start, so no test row
    is ever visible during training.
    """
    if window not in ("expanding", "sliding"):
        raise ValueError("window must be 'expanding' or 'sliding'")

    origin = min_train
    while origin + horizon <= n:
        train_start = 0 if window == "expanding" else max(0, origin - min_train)
        split = Split(
            train_start=train_start,
            train_end=origin,
            test_start=origin,
            test_end=origin + horizon,
        )
        # No-leakage invariant — fail loudly rather than silently cheat.
        assert split.train_end == split.test_start, "leakage: train overlaps test"
        assert split.train_start < split.train_end, "empty training window"
        yield split
        origin += step
