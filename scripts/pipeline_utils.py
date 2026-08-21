"""Pure helpers for the hybrid pipeline: scale-aware config and quality gates.

These functions are intentionally side-effect free so the pipeline logic is
unit-testable without loading AnnData / the CellMarker database. Design
references: ``docs/superpowers/specs/2026-08-21-hybridscsinglemarker-skeleton-design.md``
§4 (defaults), §4.1 (scale-aware fusion policy), §5.2 (reference capping) and
§6.3 (flip-precision quality gate).
"""

from __future__ import annotations

import numpy as np

# Per-type reference cap tiers (calibrated for the 160k-cell pbmc50k reference):
# n_ref >= 120k -> 300, n_ref >= 50k -> 500, otherwise no cap.
CAP_TIERS: tuple[tuple[int, int], ...] = ((120_000, 300), (50_000, 500))

# Fusion is net-negative at >= 50k single-reference scale (measured flip
# precision 0.253 in the canonical experiments); see design §4.1.
SINGLE_REF_LAMBDA_ZERO_AT: int = 50_000


def auto_cap_for_ref(n_ref: int) -> int | None:
    """Per-type reference cap for a reference with ``n_ref`` cells.

    ``None`` means "no capping" (small/medium references run uncapped).
    """
    for threshold, cap in CAP_TIERS:
        if n_ref >= threshold:
            return cap
    return None


def flip_stats(
    base_labels, fused_labels, truth
) -> dict[str, int | float]:
    """Compare fused vs base predictions against ground truth.

    A *flip* is a cell whose fused label differs from the base label:

    - ``n_good``: flip, and the fused label equals the truth;
    - ``n_bad``: flip, and the fused label differs from the truth;
    - ``n_both_ok``: no flip, base label correct;
    - ``n_both_bad``: no flip, base label wrong;
    - ``flip_precision``: ``n_good / (n_good + n_bad)``, ``NaN`` when no flips.

    ``flip_precision < 0.5`` means the fusion has no net value at this scale
    (design §6.3: flipping fewer labels would reduce loss).
    """
    base = np.asarray(base_labels)
    fused = np.asarray(fused_labels)
    truth = np.asarray(truth)
    flipped = fused != base
    good = flipped & (fused == truth)
    bad = flipped & (fused != truth)
    both_ok = ~flipped & (base == truth)
    both_bad = ~flipped & (base != truth)
    denom = int(good.sum()) + int(bad.sum())
    precision = float(good.sum() / denom) if denom > 0 else float("nan")
    return {
        "n_good": int(good.sum()),
        "n_bad": int(bad.sum()),
        "n_both_ok": int(both_ok.sum()),
        "n_both_bad": int(both_bad.sum()),
        "flip_precision": precision,
    }


def default_lambda(*, n_ref: int, n_refs: int, method: str) -> float:
    """Fusion strength default (design §4.1).

    Single reference with >= 50k cells -> ``0.0`` (pure pysingle; measured
    net-negative fusion at that scale, overridable with ``--lambda`` /
    ``--force-fusion``). Everything else keeps the canonical ``0.3``.
    """
    if n_refs == 1 and n_ref >= SINGLE_REF_LAMBDA_ZERO_AT:
        return 0.0
    return 0.3
