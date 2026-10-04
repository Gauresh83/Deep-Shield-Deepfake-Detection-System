"""
Face-level fusion — combines however many of the face detectors are
currently available (Detector 1: face authenticity, Detector 2: AI-generated
image, Detector 3: face-swap) into a single verdict.

This is intentionally separate from backend/modules/fusion/engine.py, which
combines Face + Voice + NLP at the TOP level. This module only combines
detectors *within* the face module, producing one face_score/verdict that
engine.py keeps consuming exactly as before — engine.py needs no changes.

Written generically (a list of signals, not fixed auth/swap parameters) so
adding Detector 2 didn't require rewriting the combine logic — the same
function handles 2 or 3 (or more, later) detectors.

Combine rule (see PHASE1_ARCHITECTURE.md §5, extended to N detectors):
  - Any available detector that is confidently "fake" (or "ai_generated",
    confidence > 60) wins outright — UNLESS another detector is, at the
    same time, confidently "authentic" (confidence > 60). Detectors
    specialize in different fake-types and different domains, so one
    detector being wrongly-confident (e.g. due to a domain mismatch) must
    not be allowed to silently override another detector that is actually
    suited to the input and correctly confident the other way. When
    detectors disagree like this, fall through to the confidence-weighted
    average instead of letting either side blindly win.
  - If there is no such disagreement, the most confident "fake"-family
    detector wins outright (original rule, unchanged).
  - Otherwise, confidence-weighted average across whichever detectors
    actually produced a score (unavailable/NOT_LOADED detectors are
    skipped, not treated as neutral 50s that would dilute the result).
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class DetectorSignal:
    name: str                          # breakdown key, e.g. "face_authenticity"
    score: Optional[float]             # None if this detector didn't run/load
    verdict: Optional[str]
    confidence: Optional[float]
    model: str
    fake_value: str = "fake"           # this detector's word for "fake" (e.g. "ai_generated")


def fuse_detectors(
    signals: List[DetectorSignal],
    score_to_verdict_fn,
) -> Tuple[float, str, float, str, Dict[str, Any]]:
    """
    Returns (final_score, final_verdict, final_confidence, primary_reason, breakdown).

    score_to_verdict_fn: the calibrated 3-band verdict function to reuse for
    the weighted-average case (passed in rather than imported, so this
    module has no dependency on detector.py — avoids a circular import).
    """
    breakdown = {
        s.name: {"score": s.score, "verdict": s.verdict,
                  "confidence": s.confidence, "model": s.model}
        for s in signals
    }

    available = [s for s in signals if s.score is not None]

    if not available:
        return 50.0, "uncertain", 0.0, "none", breakdown

    if len(available) == 1:
        s = available[0]
        return s.score, s.verdict, s.confidence, s.name, breakdown

    # Any confident "fake"-family verdict wins outright — pick the most
    # confident one if several detectors qualify. BUT: if another detector
    # is simultaneously confidently "authentic", don't let a confident-fake
    # verdict blindly win — one detector's domain-mismatch error (e.g.
    # Detector 1 on compressed FF++ frames) must not override a detector
    # that is actually suited to this input and correctly confident.
    fake_candidates = [s for s in available
                        if s.verdict == s.fake_value and (s.confidence or 0) > 60]
    authentic_candidates = [s for s in available
                             if s.verdict == "authentic" and (s.confidence or 0) > 60]

    if fake_candidates and not authentic_candidates:
        winner = max(fake_candidates, key=lambda s: s.confidence or 0)
        return winner.score, "fake", winner.confidence, winner.name, breakdown

    # Otherwise, confidence-weighted average across available detectors only.
    # This also covers the disagreement case above (fake vs authentic both
    # confident) — it blends them instead of picking a winner blindly.
    total_conf = sum((s.confidence or 0) for s in available)
    if total_conf <= 0:
        combined = sum(s.score for s in available) / len(available)
    else:
        combined = sum(s.score * (s.confidence or 0) for s in available) / total_conf

    verdict, confidence = score_to_verdict_fn(combined)
    # Attribute the result to whichever available detector scored lowest
    # (most "fake-leaning") — informative even when the overall verdict
    # isn't "fake", since it shows which signal pulled the average down.
    primary = min(available, key=lambda s: s.score)
    return round(combined, 2), verdict, confidence, primary.name, breakdown


def fuse_face_detectors(
    auth_score: float, auth_verdict: str, auth_confidence: float, auth_model: str,
    swap_score: Optional[float], swap_verdict: Optional[str],
    swap_confidence: Optional[float], swap_model: str,
    score_to_verdict_fn,
) -> Tuple[float, str, float, str, Dict[str, Any]]:
    """
    Backward-compatible 2-detector wrapper (Detector 1 + Detector 3), kept so
    existing call sites don't break. New code should build a DetectorSignal
    list and call fuse_detectors() directly (see detector.py's per-face loop
    for the 3-detector example, which includes Detector 2 as well).
    """
    signals = [
        DetectorSignal("face_authenticity", auth_score, auth_verdict, auth_confidence, auth_model),
        DetectorSignal("face_swap", swap_score, swap_verdict, swap_confidence, swap_model),
    ]
    score, verdict, confidence, primary, breakdown = fuse_detectors(signals, score_to_verdict_fn)
    # Keep the original primary_reason vocabulary ("face_manipulation" instead
    # of "face_swap") for anything still reading the old field name.
    primary_reason = "face_manipulation" if primary == "face_swap" and verdict == "fake" else primary
    return score, verdict, confidence, primary_reason, breakdown