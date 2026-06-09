from __future__ import annotations

import math
import random
from collections import Counter


def laplace_noise(scale: float) -> float:
    u = random.random() - 0.5
    return -scale * math.copysign(math.log(1 - 2 * abs(u)), u)


def differentially_private_counts(
    dish_ids: list[int], epsilon: float = 1.0
) -> dict[int, float]:
    """Return noisy aggregate counts. Individual feedback is not exposed."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    scale = 1.0 / epsilon
    counts = Counter(dish_ids)
    return {
        dish_id: max(0.0, count + laplace_noise(scale))
        for dish_id, count in counts.items()
    }

