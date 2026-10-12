"""CIPE — Counterfactual Image Perturbation Effect.

Single causally-grounded metric quantifying a medical VLM's actual visual
reliance on each (image, question) input. See `plan.md §1` for the
full pivot rationale and theoretical claims.

For one (image I, question q, model m), the per-counterfactual CIPE is

    CIPE_c(m, I, q) = clip_to_[0,1]( 1 − KL( P_m(a | I, q) ∥ P_m(a | I_cf, q) ) )

where `c` indexes a counterfactual type from `medtrust.augment.counterfactual`
({image_swap, anatomy_mask, text_only}). The aggregate CIPE for a model is
the mean over (sample, counterfactual) pairs; per-counterfactual breakdowns
are reported separately.

P_m(a | I, q) is approximated from the model's logprobs over a discrete
candidate-answer set (closed-form QA) or from a token-level logprob agreement
on the open-form answer (open-form QA — a slightly looser proxy). We start
with closed-form because the metric there is unambiguous; open-form
extension is future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CIPEResult:
    """Per-(sample, counterfactual) CIPE record."""
    sample_id: str
    variant: str         # one of {"image_swap", "anatomy_mask", "text_only"}
    kl: float            # KL( P(a|I) || P(a|I_cf) )  — non-negative
    cipe: float          # 1 − clip(kl, 0, 1)  ∈ [0, 1]
    p_original: list[float]
    p_counterfactual: list[float]


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Discrete KL divergence KL(p || q). Returns 0 if either dist is empty.

    Both inputs must already be normalized probability vectors; we add `eps`
    only to `q` before log to avoid log(0).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch p={p.shape} q={q.shape}")
    if p.size == 0:
        return 0.0
    # Filter to the support of p (KL is undefined where p > 0 but q = 0; we
    # use the eps-floor convention).
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def cipe_score(
    p_original: Sequence[float],
    p_counterfactual: Sequence[float],
    *,
    sample_id: str = "",
    variant: str = "",
) -> CIPEResult:
    """Compute CIPE for one (sample, counterfactual) pair.

    Both distributions must be over the same answer-candidate set in the
    same order. They will be re-normalized internally to guard against
    floating-point drift in the upstream logprob → prob conversion.
    """
    p = np.asarray(p_original, dtype=np.float64)
    q = np.asarray(p_counterfactual, dtype=np.float64)

    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError(f"non-positive mass: p_sum={p_sum}, q_sum={q_sum}")
    p = p / p_sum
    q = q / q_sum

    kl = _kl(p, q)
    cipe = float(1.0 - min(max(kl, 0.0), 1.0))
    return CIPEResult(
        sample_id=sample_id,
        variant=variant,
        kl=kl,
        cipe=cipe,
        p_original=p.tolist(),
        p_counterfactual=q.tolist(),
    )


def aggregate_cipe(results: Sequence[CIPEResult]) -> dict[str, float]:
    """Compute summary stats over a flat list of CIPE results.

    Returns:
      mean_cipe   — mean over all (sample, variant) pairs
      mean_cipe_image_swap, mean_cipe_anatomy_mask, mean_cipe_text_only
                  — per-variant means
      frac_high   — fraction with cipe > 0.7 (the "fails CIPE" rate)
      n           — count
    """
    if not results:
        return {
            "mean_cipe": float("nan"),
            "frac_high": float("nan"),
            "n": 0,
        }
    by_variant: dict[str, list[float]] = {}
    all_cipe: list[float] = []
    for r in results:
        all_cipe.append(r.cipe)
        by_variant.setdefault(r.variant, []).append(r.cipe)

    out: dict[str, float] = {
        "mean_cipe": float(np.mean(all_cipe)),
        "median_cipe": float(np.median(all_cipe)),
        "frac_high": float(np.mean([1.0 if c > 0.7 else 0.0 for c in all_cipe])),
        "n": len(all_cipe),
    }
    for v, cs in by_variant.items():
        out[f"mean_cipe_{v}"] = float(np.mean(cs))
        out[f"n_{v}"] = len(cs)
    return out


def cipe_from_logprobs(
    logprobs_original: Mapping[str, float],
    logprobs_counterfactual: Mapping[str, float],
    *,
    candidates: Sequence[str],
    sample_id: str = "",
    variant: str = "",
) -> CIPEResult:
    """Convenience wrapper: caller supplies per-candidate logprobs from each
    forward pass. We restrict to `candidates`, exp-normalize, and call
    cipe_score.

    Use this when your VLM adapter exposes `next_token_logprobs(prompt)`
    or equivalent. Candidates are the closed-form answer set, e.g.
    ["yes", "no"] or ["A", "B", "C", "D"] for multiple choice.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")
    lp_o = np.array([logprobs_original.get(c, -1e9) for c in candidates], dtype=np.float64)
    lp_c = np.array([logprobs_counterfactual.get(c, -1e9) for c in candidates], dtype=np.float64)
    # softmax (numerically stable)
    p_o = np.exp(lp_o - lp_o.max())
    p_c = np.exp(lp_c - lp_c.max())
    return cipe_score(p_o, p_c, sample_id=sample_id, variant=variant)
