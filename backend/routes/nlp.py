"""
NLP Linguistic Analysis API — Part 5
"""
from fastapi import APIRouter, Depends, Body
from typing import Optional
from ..core.security import get_current_active_user
from ..modules.nlp.detector import nlp_detector

router = APIRouter(prefix="/api/nlp", tags=["NLP Analysis"])


@router.get("/status", response_model=dict)
def nlp_status(current_user=Depends(get_current_active_user)):
    """Check NLP module status and capabilities."""
    return {
        "module": "Linguistic NLP Analysis",
        "version": "1.0.0 (Part 5)",
        "model_loaded": nlp_detector.model_loaded,
        "transformers_available": nlp_detector.transformers_available,
        "torch_available": nlp_detector.torch_available,
        "mode": (
            "BERT fine-tuned classifier" if nlp_detector.model_loaded
            else "Statistical NLP heuristics (no external deps needed)"
        ),
        "weights_note": (
            "WEIGHTS_PATH points directly at backend/modules/nlp/weights/ (flat, "
            "matching face/voice module convention) — put config.json, model "
            "weights, and tokenizer files directly in that folder, not a subfolder."
        ),
        "min_text_length": {
            "min_words": nlp_detector.MIN_WORD_COUNT,
            "min_sentences": nlp_detector.MIN_SENTENCE_COUNT,
            "reason": "Below this, TTR/burstiness/filler-density are statistically meaningless.",
        },
        "features_extracted": [
            "Perplexity proxy (bigram entropy)",
            "Type-Token Ratio (lexical diversity)",
            "Average sentence length",
            "Sentence length variance (burstiness)",
            "Filler word density",
            "Passive voice ratio",
            "Repetition score (n-gram overlap)",
            "Punctuation density",
            "AI marker phrase detection",
            "Human speech marker detection",
        ],
        "weights_path": nlp_detector.WEIGHTS_PATH,
        "instructions": {
            "install_transformers": "pip install transformers torch",
            "model_weights": "Place config.json/model weights/tokenizer files directly in backend/modules/nlp/weights/",
            "pretrain_dataset": "HC3 / GPT-wiki-intro / OpenAI Text Classifier datasets",
        }
    }


@router.post("/analyze", response_model=dict)
def analyze_text(
    text: str = Body(..., embed=True, min_length=10),
    current_user=Depends(get_current_active_user),
):
    """
    Analyze text/transcript for AI-generated or unnatural linguistic patterns.

    Detects:
    - Unusually uniform sentence length (AI trait)
    - Near-zero filler words (AI trait)
    - High vocabulary diversity (AI trait)
    - Low burstiness (AI trait)
    - AI-typical marker phrases
    - Human speech markers
    - Passive voice ratio
    - Bigram repetition
    """
    result = nlp_detector.analyze_text(text)
    return {
        "status": result.status,     # ANALYZED | INSUFFICIENT_TEXT
        "message": result.message,
        "analysis": result.to_dict(),
        "summary": {
            "score": result.score,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "ai_generated_probability": result.ai_generated_probability,
            "word_count": result.word_count,
            "anomalous_features": result.anomalous_features,
            "processing_time_ms": result.processing_time_ms,
        }
    }


@router.post("/analyze-batch", response_model=dict)
def analyze_batch(
    texts: list = Body(..., embed=True),
    current_user=Depends(get_current_active_user),
):
    """Analyze multiple text snippets (e.g. multiple speaker turns)."""
    if len(texts) > 20:
        from fastapi import HTTPException
        raise HTTPException(400, "Max 20 texts per batch.")
    results = []
    for t in texts:
        if isinstance(t, str) and len(t.strip()) >= 10:
            r = nlp_detector.analyze_text(t)
            results.append({
                "text_preview": t[:80] + "..." if len(t) > 80 else t,
                "status": r.status,          # ANALYZED | INSUFFICIENT_TEXT
                "message": r.message,
                "score": r.score,
                "verdict": r.verdict,
                "ai_probability": r.ai_generated_probability,
                "word_count": r.word_count,
            })

    # Only average/verdict over texts that actually got a real analysis —
    # an INSUFFICIENT_TEXT entry's score is a meaningless placeholder (0),
    # averaging it in would silently drag the whole batch score down.
    analyzed = [r for r in results if r["status"] == "ANALYZED"]
    avg_score = round(sum(r["score"] for r in analyzed) / len(analyzed), 2) if analyzed else None

    # Match detector.py's calibrated 3-band thresholds (35/65), not the old
    # uncalibrated 45/70 split, and the current verdict vocabulary.
    if avg_score is None:
        batch_verdict = "insufficient_data"
    elif avg_score >= nlp_detector.UNCERTAIN_HIGH:
        batch_verdict = "authentic"
    elif avg_score <= nlp_detector.UNCERTAIN_LOW:
        batch_verdict = "fake"
    else:
        batch_verdict = "uncertain"

    return {
        "batch_size": len(results),
        "analyzed_count": len(analyzed),
        "skipped_insufficient_text": len(results) - len(analyzed),
        "avg_score": avg_score,
        "batch_verdict": batch_verdict,
        "results": results,
    }