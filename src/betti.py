import numpy as np
from src.constants import BETTI_NOISE_FLOOR

def estimate_betti_number(intervals, add_one=False, floor=BETTI_NOISE_FLOOR):
    '''
    Estimate the Betti number from a list of intervals.
    '''
    lifetimes = np.array([d - b for b, d in intervals if np.isfinite(d)])
    lifetimes = np.sort(lifetimes[lifetimes > 1e-12])[::-1]
    base = 1 if add_one else 0

    if len(lifetimes) == 0:
        return base

    candidates = lifetimes[lifetimes > floor * lifetimes[0]]
    if len(candidates) <= 1:
        return base + len(candidates)

    ratios = candidates[:-1] / candidates[1:]
    gap_idx = int(np.argmax(ratios))
    return base + gap_idx + 1
