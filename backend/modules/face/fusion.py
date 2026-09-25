"""
backend/modules/face/fusion.py
================================
Face-level fusion — combines whichever Detector 1/2/3 results are
present in detector_breakdown into a single top-level score/verdict/
confidence, while every individual detector's raw output stays visible
in detector_breakdown itself (built by detector.py).

Why this exists as a separate module (see PHASE1_ARCHITECTURE.md §5):
detector.py's job is running detectors and building detector_breakdown;
fusion.py's job is turning that breakdown into ONE number the rest of
the app (fusion/engine.py, frontend) can read. Keeping this separate
means adding Detector 2 (ai_gen_detector.py, not trained yet) later is
a drop-in — nothing in this file's logic changes, it just needs to see
an "ai_generated" entry show up in the dict it's given.

Currently wired detectors (see detector.py):
  - face_authenticity      (Detector 1, EfficientNet-B4 / heuristic)
  - face_swap_manipulation (Detector 3, ResNet18, FaceForensics++)
Not yet wired (future — Phase 1):
  - ai_generated            (Detector 2, ai_gen_detector.py)

To add Detector 2 later: have detector.py put its result into
detector_breakdown under the key "ai_generated" with the same shape
({"status","score","verdict","confidence","model_used","model_loaded"}),
using "ai_generated" as its fake-side verdict string. That's it — no
change needed here.
"""
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# Fake-side verdict labels across all detectors. Extend this set if a
# future detector introduces a new fake-side verdict string.
FAKE_VERDICTS = {"fake", "ai_generated"}
CONFIDENT_FAKE_THRESHOLD = 60.0

# Human-readable primary_reason per detector — extend when Detector 2 lands.
DETECTOR_LABELS = {
    "face_authenticity": "face_manipulation",
    "face_swap_manipulation": "face_swap_manipulation",
    "ai_generated": "ai_generated_image",
}

# Same calibrated 3-band thresholds used by every detector in this codebase.
UNCERTAIN_LOW = 35.0
UNCERTAIN_HIGH = 65.0


@dataclass
class FusionResult:
    score: float
    verdict: str                      # authentic | uncertain | fake
    confidence: float
    message: str = ""
    primary_reason: Optional[str] = None            # which detector drove a "fake" verdict, if any
    contributing_detectors: Dict[str, Any] = field(default_factory=dict)


def _is_usable(entry: Dict[str, Any]) -> bool:
    """A detector's result only counts toward fusion if it actually ran
    and is model-backed (not a NOT_LOADED / NOT_RUN / ERROR stub)."""
    if not entry:
        return False
    return entry.get("status") == "ANALYZED" and entry.get("model_loaded", False)


def fuse_face_detectors(detector_breakdown: Dict[str, Any]) -> FusionResult:
    """
    Combines however many detector results are present and usable in
    detector_breakdown into one FusionResult.

    Rule (PHASE1_ARCHITECTURE.md §5):
      1. If any usable detector is confidently on the fake side
         (confidence > 60), trust it immediately — detectors specialize
         in different fake-types, so an OR on the fake side avoids one
         detector's blind spot hiding another's catch.
      2. Otherwise, confidence-weighted average across all usable
         detectors' scores — a low-confidence detector shouldn't drag
         the combined score as hard as a confident one.
      3. If nothing is usable (e.g. only Detector 1's OpenCV-heuristic
         fallback ran, no model loaded anywhere), fall back to whatever
         single entry detector_breakdown does have, so the app still
         returns something instead of an empty result.
    """
    usable = {name: entry for name, entry in detector_breakdown.items() if _is_usable(entry)}

    # ── Step 1: OR on confident-fake ──────────────────────────────────
    for name, entry in usable.items():
        if entry.get("verdict") in FAKE_VERDICTS and entry.get("confidence", 0) > CONFIDENT_FAKE_THRESHOLD:
            return FusionResult(
                score=entry["score"], verdict="fake", confidence=entry["confidence"],
                message=f"{DETECTOR_LABELS.get(name, name).replace('_', ' ').capitalize()} detected with high confidence.",
                primary_reason=DETECTOR_LABELS.get(name, name),
                contributing_detectors=usable,
            )

    # ── Step 2: confidence-weighted average ───────────────────────────
    if usable:
        total_conf = sum(e.get("confidence", 0) for e in usable.values())
        if total_conf == 0:
            combined = sum(e["score"] for e in usable.values()) / len(usable)
        else:
            combined = sum(e["score"] * e.get("confidence", 0) for e in usable.values()) / total_conf
        verdict, confidence = _score_to_verdict(combined)
        message = ("Analysis complete." if verdict != "uncertain" else
                   "The available evidence is insufficient for a confident authenticity determination.")
        return FusionResult(
            score=round(combined, 2), verdict=verdict, confidence=confidence,
            message=message, primary_reason=None, contributing_detectors=usable,
        )

    # ── Step 3: nothing usable — fall back to any single entry present ─
    for name, entry in detector_breakdown.items():
        if entry and "score" in entry:
            return FusionResult(
                score=entry["score"], verdict=entry.get("verdict", "uncertain"),
                confidence=entry.get("confidence", 0.0),
                message=entry.get("message", ""),
                primary_reason=None, contributing_detectors={name: entry},
            )

    return FusionResult(score=50.0, verdict="uncertain", confidence=0.0,
                         message="No detector produced a usable result.",
                         primary_reason=None, contributing_detectors={})


def _score_to_verdict(score: float) -> Tuple[str, float]:
    """Identical calibrated 3-band logic used across all detectors — kept
    as a local copy (same design note as faceswap_detector.py) so this
    module has no import dependency on detector.py."""
    if score >= UNCERTAIN_HIGH:
        span = 100 - UNCERTAIN_HIGH
        conf = 60 + min((score - UNCERTAIN_HIGH) / span, 1.0) * 39
        return "authentic", round(min(conf, 99.9), 2)
    if score <= UNCERTAIN_LOW:
        span = UNCERTAIN_LOW
        conf = 60 + min((UNCERTAIN_LOW - score) / span, 1.0) * 39
        return "fake", round(min(conf, 99.9), 2)
    dist_from_edge = min(score - UNCERTAIN_LOW, UNCERTAIN_HIGH - score)
    band_half = (UNCERTAIN_HIGH - UNCERTAIN_LOW) / 2
    conf = 45 - (dist_from_edge / band_half) * 30
    return "uncertain", round(max(conf, 10.0), 2)