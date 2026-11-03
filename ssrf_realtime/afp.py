"""Instantaneous adiabatic fast passage (AFP) helpers.

AFP is a discrete population swap on selected bins: n+↔n0 at bin i and
n0↔n− at its mirror m = n−1−i. Continuous ss-RF rate equations are unchanged.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from .profile_editing import region_indices


def perform_afp_on_populations(
    rho_plus: np.ndarray,
    rho_zero: np.ndarray,
    rho_minus: np.ndarray,
    sweep: Sequence[int],
    efficiency: float = 1.0,
) -> None:
    """AFP sweep on per-bin populations (in place): n+↔n0 at i, n0↔n− at mirror."""
    n = len(rho_plus)
    eff = float(efficiency)
    if not np.isfinite(eff) or eff < 0.0 or eff > 1.0:
        raise ValueError("AFP efficiency must be finite and in [0, 1].")
    for i in sweep:
        i = int(i)
        if i < 0 or i >= n:
            raise ValueError(f"AFP sweep index {i} outside [0, {n - 1}].")
        m = n - 1 - i
        (rho_plus[i], rho_zero[i]) = (
            eff * rho_zero[i] + (1.0 - eff) * rho_plus[i],
            eff * rho_plus[i] + (1.0 - eff) * rho_zero[i],
        )
        if m == i:
            continue
        (rho_zero[m], rho_minus[m]) = (
            eff * rho_minus[m] + (1.0 - eff) * rho_zero[m],
            eff * rho_zero[m] + (1.0 - eff) * rho_minus[m],
        )


def mirror_safe_indices(indices: Iterable[int], n_bins: int) -> List[int]:
    """Keep at most one index from each mirror pair (order preserved).

    Passing both a bin and its mirror into the AFP sweep would apply the
    operator twice. The first occurrence wins; its mirror is blocked.
    """
    n = int(n_bins)
    if n < 1:
        raise ValueError("n_bins must be positive.")
    chosen: List[int] = []
    blocked = set()
    for raw in indices:
        i = int(raw)
        if i < 0 or i >= n or i in blocked:
            continue
        m = n - 1 - i
        chosen.append(i)
        blocked.add(i)
        blocked.add(m)
    return sorted(chosen)


def qneg_sweep_indices(q: np.ndarray, allowed: Optional[Iterable[int]] = None) -> List[int]:
    """Bins with Q < 0, keeping at most one index from each mirror pair.

    More-negative Q is preferred when both members of a mirror pair qualify.
    ``allowed`` optionally restricts the candidate set (e.g. an L/R window).
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError("q must be a 1-D array.")
    neg = np.flatnonzero(q < 0.0)
    if allowed is not None:
        allow = np.zeros(q.size, dtype=bool)
        idx = np.asarray(list(allowed), dtype=int)
        idx = idx[(idx >= 0) & (idx < q.size)]
        allow[idx] = True
        neg = neg[allow[neg]]
    if neg.size == 0:
        return []
    ordered = neg[np.argsort(q[neg])]  # more negative first
    return mirror_safe_indices(ordered, q.size)


def selective_window_indices(grid, left: float, right: float) -> List[int]:
    """Contiguous L/R window on ``grid``, made mirror-safe for AFP."""
    region = list(region_indices(grid, left, right))
    return mirror_safe_indices(region, len(np.asarray(grid)))
