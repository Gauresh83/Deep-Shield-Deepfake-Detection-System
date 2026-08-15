"""
M3-ID Risk Fusion Engine — Part 6
===================================
Aggregates Face, Voice, and NLP scores into a single explainable verdict.

Key features:
  1. Weighted multi-modal scoring  (Face 35% | Voice 35% | NLP 30%)
  2. Confidence interval calculation
  3. Explainability (XAI) — per-module contribution bars
  4. Alert severity classification (CRITICAL | HIGH | MEDIUM | LOW | CLEAR)
  5. Evidence summary — human-readable finding bullets
  6. Anomaly boosting — if any single module is very low, overall score drops
  7. Consensus weighting — adjusts weights based on module availability
"""

import time
import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict


# ── Alert severity thresholds ──────────────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "CRITICAL": (0,   30),   # fusion score 0-30
    "HIGH":     (30,  45),   # 30-45
    "MEDIUM":   (45,  65),   # 45-65
    "LOW":      (65,  80),   # 65-80
    "CLEAR":    (80, 100),   # 80-100
}

# Base weights for each module
MODULE_WEIGHTS = {
    "face":  0.35,
    "voice": 0.35,
    "nlp":   0.30,
}


@dataclass
class ModuleContribution:
    module: str
    score: Optional[float]
    weight: float
    weighted_contribution: float
    status: str          # active | skipped | error
    confidence: float


@dataclass
class FusionResult:
    # Core output
    fusion_score: float              # 0-100
    verdict: str                     # authentic | suspicious | fake
    alert_severity: str              # CRITICAL | HIGH | MEDIUM | LOW | CLEAR
    confidence: float                # 0-100

    # Confidence interval (±)
    confidence_interval_low: float
    confidence_interval_high: float

    # XAI — per-module breakdown
    contributions: List[ModuleContribution] = field(default_factory=list)

    # Evidence bullets
    evidence_for_fake: List[str] = field(default_factory=list)
    evidence_for_authentic: List[str] = field(default_factory=list)
    anomaly_flags: List[str] = field(default_factory=list)

    # Module raw scores
    face_score: Optional[float] = None
    voice_score: Optional[float] = None
    nlp_score: Optional[float] = None

    # Metadata
    modules_active: int = 0
    processing_time_ms: int = 0
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class FusionEngine:
    """
    Risk Fusion Engine — combines all M3-ID module outputs into one verdict.
    """

    def fuse(
        self,
        face_score: Optional[float]  = None,
        voice_score: Optional[float] = None,
        nlp_score: Optional[float]   = None,
        face_details:  Optional[Dict] = None,
        voice_details: Optional[Dict] = None,
        nlp_details:   Optional[Dict] = None,
    ) -> FusionResult:
        start = time.time()

        active = {
            k: v for k, v in
            {"face": face_score, "voice": voice_score, "nlp": nlp_score}.items()
            if v is not None
        }

        if not active:
            return self._empty_result()

        # ── 1. Normalize weights for available modules ────────────────────────
        total_w = sum(MODULE_WEIGHTS[k] for k in active)
        norm_weights = {k: MODULE_WEIGHTS[k] / total_w for k in active}

        # ── 2. Weighted average ───────────────────────────────────────────────
        fusion_score = round(
            sum(v * norm_weights[k] for k, v in active.items()), 2)

        # ── 3. Anomaly boosting — if any module is extremely low, penalise ────
        min_score = min(active.values())
        if min_score < 25:
            # Pull fusion down toward the worst signal
            fusion_score = round(fusion_score * 0.75 + min_score * 0.25, 2)
        elif min_score < 40:
            fusion_score = round(fusion_score * 0.85 + min_score * 0.15, 2)

        fusion_score = max(0.0, min(100.0, fusion_score))

        # ── 4. Verdict ────────────────────────────────────────────────────────
        if fusion_score >= 75:
            verdict = "authentic"
        elif fusion_score >= 50:
            verdict = "suspicious"
        else:
            verdict = "fake"

        # ── 5. Alert severity ─────────────────────────────────────────────────
        severity = self._get_severity(fusion_score)

        # ── 6. Confidence interval ────────────────────────────────────────────
        # More modules = tighter interval; single module = wider uncertainty
        n = len(active)
        base_uncertainty = {1: 18, 2: 12, 3: 8}.get(n, 8)
        # Variance across module scores increases uncertainty
        scores_list = list(active.values())
        if len(scores_list) > 1:
            mean = sum(scores_list) / len(scores_list)
            variance = sum((s - mean) ** 2 for s in scores_list) / len(scores_list)
            spread_factor = math.sqrt(variance) / 10
        else:
            spread_factor = 2.0
        uncertainty = base_uncertainty + spread_factor
        ci_low  = round(max(0,   fusion_score - uncertainty), 2)
        ci_high = round(min(100, fusion_score + uncertainty), 2)

        # ── 7. Confidence (0-100, how far from nearest decision boundary) ─────
        boundaries = [50, 75]
        min_dist = min(abs(fusion_score - b) for b in boundaries)
        confidence = round(min(min_dist / 25 * 100, 99.9), 2)

        # ── 8. Per-module contributions (XAI) ─────────────────────────────────
        contributions = []
        all_modules = ["face", "voice", "nlp"]
        for mod in all_modules:
            score = active.get(mod)
            if score is not None:
                w = norm_weights[mod]
                contributions.append(ModuleContribution(
                    module=mod,
                    score=round(score, 2),
                    weight=round(w, 3),
                    weighted_contribution=round(score * w, 2),
                    status="active",
                    confidence=round(min(abs(score - 50) / 50 * 100, 99), 2),
                ))
            else:
                contributions.append(ModuleContribution(
                    module=mod,
                    score=None,
                    weight=0.0,
                    weighted_contribution=0.0,
                    status="skipped",
                    confidence=0.0,
                ))

        # ── 9. Evidence generation ─────────────────────────────────────────────
        evidence_fake = []
        evidence_auth = []
        anomaly_flags = []

        # Face evidence
        if face_score is not None:
            if face_score < 40:
                evidence_fake.append(f"Face score {face_score:.1f}% — strong deepfake indicators")
                if face_details:
                    if face_details.get("blending_artifact_score", 0) > 60:
                        evidence_fake.append("High blending artifact score at face boundaries")
                    if face_details.get("temporal_stability", 100) < 50:
                        evidence_fake.append("Low temporal stability across frames")
            elif face_score > 75:
                evidence_auth.append(f"Face score {face_score:.1f}% — high facial authenticity")
            else:
                anomaly_flags.append(f"Face score {face_score:.1f}% — borderline, manual review advised")

        # Voice evidence
        if voice_score is not None:
            if voice_score < 40:
                evidence_fake.append(f"Voice score {voice_score:.1f}% — likely voice clone/synthesis")
                if voice_details:
                    if voice_details.get("clone_probability", 0) > 60:
                        evidence_fake.append(f"Clone probability: {voice_details['clone_probability']:.1f}%")
                    if voice_details.get("pitch_naturalness", 100) < 40:
                        evidence_fake.append("Unnatural pitch patterns — low F0 variation")
            elif voice_score > 75:
                evidence_auth.append(f"Voice score {voice_score:.1f}% — authentic vocal characteristics")
            else:
                anomaly_flags.append(f"Voice score {voice_score:.1f}% — ambiguous vocal patterns")

        # NLP evidence
        if nlp_score is not None:
            if nlp_score < 40:
                evidence_fake.append(f"NLP score {nlp_score:.1f}% — AI-generated speech patterns detected")
                if nlp_details:
                    ai_markers = nlp_details.get("ai_marker_count", 0)
                    if ai_markers > 2:
                        evidence_fake.append(f"{ai_markers} AI-typical marker phrases found")
                    if nlp_details.get("filler_word_density", 1) < 0.005:
                        evidence_fake.append("Near-zero filler words — unnatural for human speech")
            elif nlp_score > 75:
                evidence_auth.append(f"NLP score {nlp_score:.1f}% — natural human linguistic patterns")
                if nlp_details:
                    if nlp_details.get("human_marker_count", 0) > 1:
                        evidence_auth.append("Human speech markers detected (fillers, natural phrasing)")
            else:
                anomaly_flags.append(f"NLP score {nlp_score:.1f}% — mixed linguistic signals")

        # Cross-modal anomaly flags
        if len(active) > 1:
            scores_list_2 = list(active.values())
            mean2 = sum(scores_list_2) / len(scores_list_2)
            for mod, s in active.items():
                if abs(s - mean2) > 30:
                    anomaly_flags.append(
                        f"Cross-modal inconsistency: {mod} score ({s:.1f}%) "
                        f"deviates {abs(s-mean2):.1f}% from average"
                    )

        # ── 10. Recommendation ────────────────────────────────────────────────
        if severity == "CRITICAL":
            recommendation = "BLOCK — High confidence deepfake. Do not proceed with verification."
        elif severity == "HIGH":
            recommendation = "ESCALATE — Strong indicators of manipulation. Require secondary verification."
        elif severity == "MEDIUM":
            recommendation = "REVIEW — Ambiguous signals. Human review recommended before proceeding."
        elif severity == "LOW":
            recommendation = "MONITOR — Mostly authentic. Log and monitor for further activity."
        else:
            recommendation = "PASS — All modalities verified as authentic. Proceed with confidence."

        ms = int((time.time() - start) * 1000)

        return FusionResult(
            fusion_score=fusion_score,
            verdict=verdict,
            alert_severity=severity,
            confidence=confidence,
            confidence_interval_low=ci_low,
            confidence_interval_high=ci_high,
            contributions=contributions,
            evidence_for_fake=evidence_fake,
            evidence_for_authentic=evidence_auth,
            anomaly_flags=anomaly_flags,
            face_score=face_score,
            voice_score=voice_score,
            nlp_score=nlp_score,
            modules_active=len(active),
            processing_time_ms=ms,
            recommendation=recommendation,
        )

    @staticmethod
    def _get_severity(score: float) -> str:
        for label, (lo, hi) in SEVERITY_THRESHOLDS.items():
            if lo <= score < hi:
                return label
        return "CLEAR"

    @staticmethod
    def _empty_result() -> FusionResult:
        return FusionResult(
            fusion_score=50.0, verdict="suspicious",
            alert_severity="MEDIUM", confidence=0.0,
            confidence_interval_low=30.0, confidence_interval_high=70.0,
            recommendation="No module data available — cannot compute fusion score.",
        )


# Singleton
fusion_engine = FusionEngine()
