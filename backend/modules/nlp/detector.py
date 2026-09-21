"""
M3-ID Linguistic NLP Analysis Module — Part 5
===============================================
Detects AI-generated, scripted, or unnatural speech patterns in text.

Architecture (best available, auto-selected):

Tier 1 — BERT classifier (when transformers installed):
  • Fine-tuned DistilBERT binary classifier (HC3 + M4 + RAID)
  • Input: tokenized transcript (max 512 tokens)
  • Output: probability of AI/synthetic origin

Tier 2 — Statistical NLP (always available, no deps):
  • Perplexity proxy via bigram entropy
  • Type-Token Ratio (lexical diversity)
  • Average sentence length & variance
  • Punctuation density anomaly
  • Passive voice ratio
  • Filler word density (um, uh, you know…)
  • Repetition score (n-gram overlap)
  • Burstiness (sentence-length distribution)

Tier 3 — TF-IDF keyword anomaly:
  • Compares transcript against known human vs AI vocabulary patterns
  • Scores unusual word frequency distributions

Model weights path:
  backend/modules/nlp/weights/nlp_model.pt
"""

import os
import re
import math
import time
import json
import string
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import Counter


# ── Human speech reference patterns (from linguistic research) ────────────────
# These are empirically observed ranges for genuine human speech transcripts.
HUMAN_SPEECH_PATTERNS = {
    "avg_sentence_len":   (8,  22),   # words per sentence
    "type_token_ratio":   (0.4, 0.8), # unique/total words
    "filler_density":     (0.01, 0.06),# filler words / total words
    "punctuation_density":(0.05, 0.15),
    "passive_ratio":      (0.05, 0.20),
    "repetition_score":   (0.0, 0.25),
}

# AI-generated text tends toward: long uniform sentences, high vocab diversity,
# almost zero fillers, low burstiness, high passive voice, unusual collocations.

FILLER_WORDS = {
    "um","uh","er","ah","hmm","you know","i mean","like","basically",
    "actually","literally","honestly","right","okay","so","well","anyway"
}

AI_MARKER_PHRASES = [
    "certainly","absolutely","of course","it is important to note",
    "it is worth mentioning","in conclusion","furthermore","moreover",
    "in summary","delve","straightforward","utilize","leverage",
    "ensure","provide","facilitate","implement","comprehensive"
]

HUMAN_MARKER_PHRASES = [
    "um","uh","you know","i mean","kind of","sort of","like i said",
    "wait","hang on","actually","honestly","right so"
]


@dataclass
class NLPAnalysisResult:
    score: float                      # 0-100  (higher = more human/authentic)
    verdict: str                      # authentic | uncertain | fake
    confidence: float                 # 0-100

    # Statistical features
    perplexity_proxy: float           # higher = more predictable (AI-like)
    type_token_ratio: float           # lexical diversity 0-1
    avg_sentence_length: float        # words per sentence
    sentence_len_variance: float      # burstiness / naturalness
    filler_word_density: float        # 0-1 (human speech has some fillers)
    passive_voice_ratio: float        # 0-1
    repetition_score: float           # 0-1 (high = repetitive)
    punctuation_density: float        # 0-1
    ai_marker_count: int              # count of AI-typical phrases
    human_marker_count: int           # count of human speech markers
    ai_generated_probability: float   # 0-100

    # Text metadata
    word_count: int
    sentence_count: int
    unique_words: int
    avg_word_length: float
    text_length: int

    # Detailed findings
    top_ai_markers: List[str] = field(default_factory=list)
    top_human_markers: List[str] = field(default_factory=list)
    anomalous_features: List[str] = field(default_factory=list)

    # Model info
    model_used: str = "Statistical NLP"
    model_loaded: bool = False
    processing_time_ms: int = 0

    # ── Structured-gating fields (mirrors the face module's Phase 0 pattern) ──
    status: str = "ANALYZED"   # ANALYZED | INSUFFICIENT_TEXT
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NLPDetector:
    """
    Linguistic Analysis Engine.

    Tier 1: BERT fine-tuned classifier (needs transformers + torch + weights)
    Tier 2: Statistical + TF-IDF heuristics (always works, no pip installs needed)
    """

    # NOTE: unlike a typical HF save_pretrained() layout, this repo keeps every
    # module's weight files directly inside modules/<name>/weights/ (flat,
    # matching the face/voice modules' convention) rather than in a named
    # subfolder — so this points straight at weights/, not weights/nlp_model/.
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights")
    VOCAB_PATH   = os.path.join(os.path.dirname(__file__), "weights", "tfidf_vocab.json")

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tfidf_vocab: Dict[str, float] = {}
        self.model_loaded = False
        self.transformers_available = False
        self.torch_available = False
        self._try_load_libs()

    def _try_load_libs(self):
        try:
            import torch
            self.torch_available = True
        except ImportError:
            pass
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.transformers_available = True
            if self.torch_available and os.path.exists(self.WEIGHTS_PATH):
                self._load_model()
        except ImportError:
            pass
        # Load TF-IDF vocab if available
        if os.path.exists(self.VOCAB_PATH):
            try:
                with open(self.VOCAB_PATH) as f:
                    self.tfidf_vocab = json.load(f)
            except Exception:
                pass

    def _load_model(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.WEIGHTS_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.WEIGHTS_PATH)
            self.model.eval()
            self.model_loaded = True
            print(f"[NLPDetector] {self.model.config.model_type.upper()} weights loaded from {self.WEIGHTS_PATH}.")
        except Exception as e:
            print(f"[NLPDetector] Could not load BERT weights: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    # Minimum text length below which statistical features (TTR, sentence-length
    # variance, filler-density) are mathematically unreliable — e.g. TTR is
    # guaranteed to be ~1.0 for any short sentence with no repeated words,
    # which has nothing to do with AI-generation. This mirrors the face
    # module's Phase 0 "don't classify what you can't reliably classify" gate.
    MIN_WORD_COUNT = 25
    MIN_SENTENCE_COUNT = 2

    def analyze_text(self, text: str) -> NLPAnalysisResult:
        """Main entry — analyze text and return full NLP result."""
        start = time.time()
        text = text.strip()

        word_count = len(text.split())
        sentence_count = len(self._split_sentences(text))

        if word_count < self.MIN_WORD_COUNT or sentence_count < self.MIN_SENTENCE_COUNT:
            result = NLPAnalysisResult(
                status="INSUFFICIENT_TEXT", verdict="uncertain",
                message=(f"Text is too short ({word_count} words, {sentence_count} "
                          f"sentence{'s' if sentence_count != 1 else ''}) for a reliable "
                          f"linguistic analysis. Statistical features like vocabulary "
                          f"diversity and sentence-length variance need at least "
                          f"~{self.MIN_WORD_COUNT} words across {self.MIN_SENTENCE_COUNT}+ "
                          f"sentences to be meaningful — shorter text will look "
                          f"artificially \"suspicious\" no matter who wrote it."),
                score=0.0, confidence=0.0,
                perplexity_proxy=0.0, type_token_ratio=0.0, avg_sentence_length=0.0,
                sentence_len_variance=0.0, filler_word_density=0.0, passive_voice_ratio=0.0,
                repetition_score=0.0, punctuation_density=0.0, ai_marker_count=0,
                human_marker_count=0, ai_generated_probability=0.0,
                word_count=word_count, sentence_count=sentence_count, unique_words=0,
                avg_word_length=0.0, text_length=len(text),
                model_used="N/A — text too short to analyze", model_loaded=self.model_loaded,
                processing_time_ms=int((time.time() - start) * 1000),
            )
            return result

        if self.model_loaded:
            result = self._analyze_with_bert(text)
        else:
            result = self._analyze_statistical(text)

        result.processing_time_ms = int((time.time() - start) * 1000)
        return result

    # ── BERT inference ────────────────────────────────────────────────────────

    def _analyze_with_bert(self, text: str) -> NLPAnalysisResult:
        import torch
        # Tokenize (truncate to 256 tokens, matching training)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=256, padding=True)

        # DistilBERT's forward() does not accept token_type_ids (BERT does,
        # DistilBERT doesn't use segment embeddings at all) — but tokenizers
        # add it by default. Strip anything the model's forward() signature
        # doesn't actually accept, so this works for either architecture.
        import inspect
        accepted = set(inspect.signature(self.model.forward).parameters)
        inputs = {k: v for k, v in inputs.items() if k in accepted}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)[0]
            ai_prob = probs[1].item()  # class 1 = AI-generated

        auth_score = round((1 - ai_prob) * 100, 2)

        # Still compute statistical features for explainability
        stats = self._extract_features(text)
        verdict, conf = self._score_to_verdict(auth_score)

        r = self._build_result(auth_score, verdict, conf, stats, text)
        r.model_used = "BERT-base-uncased (fine-tuned)"
        r.model_loaded = True
        r.ai_generated_probability = round(ai_prob * 100, 2)
        return r

    # ── Statistical analysis ──────────────────────────────────────────────────

    def _analyze_statistical(self, text: str) -> NLPAnalysisResult:
        stats = self._extract_features(text)

        # Score each feature vs human speech reference
        scores = []

        # 1. Sentence length naturalness
        sl = stats["avg_sentence_length"]
        lo, hi = HUMAN_SPEECH_PATTERNS["avg_sentence_len"]
        sl_score = 100 if lo <= sl <= hi else max(0, 100 - abs(sl - (lo+hi)/2) * 4)
        scores.append(("sentence_length", sl_score, 0.20))

        # 2. Lexical diversity (TTR)
        ttr = stats["type_token_ratio"]
        lo, hi = HUMAN_SPEECH_PATTERNS["type_token_ratio"]
        ttr_score = 100 if lo <= ttr <= hi else max(0, 100 - abs(ttr - (lo+hi)/2) * 150)
        scores.append(("ttr", ttr_score, 0.15))

        # 3. Filler word density — human speech has fillers, AI doesn't
        fd = stats["filler_word_density"]
        lo, hi = HUMAN_SPEECH_PATTERNS["filler_density"]
        if fd >= lo:
            fd_score = min(100, 60 + fd * 400)  # fillers boost authenticity
        else:
            fd_score = max(20, fd * 6000)        # near-zero fillers = suspicious
        scores.append(("filler", fd_score, 0.20))

        # 4. Sentence length variance (burstiness)
        slv = stats["sentence_len_variance"]
        # Natural speech: high variance (questions, exclamations mixed)
        slv_score = min(100, slv * 3)
        scores.append(("burstiness", slv_score, 0.15))

        # 5. Perplexity proxy (low = predictable = AI-like)
        pp = stats["perplexity_proxy"]
        pp_score = min(100, pp * 8)
        scores.append(("perplexity", pp_score, 0.10))

        # 6. AI marker penalty
        ai_markers = stats["ai_marker_count"]
        ai_penalty = min(40, ai_markers * 8)
        ai_score = max(10, 100 - ai_penalty)
        scores.append(("ai_markers", ai_score, 0.10))

        # 7. Human markers bonus
        hm = stats["human_marker_count"]
        hm_score = min(100, 50 + hm * 12)
        scores.append(("human_markers", hm_score, 0.10))

        # Weighted average
        total_w = sum(w for _, _, w in scores)
        auth_score = round(
            sum(s * w / total_w for _, s, w in scores), 2)

        verdict, conf = self._score_to_verdict(auth_score)
        r = self._build_result(auth_score, verdict, conf, stats, text)
        r.model_used = "Statistical NLP (install transformers for BERT)"
        r.model_loaded = False
        r.ai_generated_probability = round(100 - auth_score, 2)
        return r

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract all linguistic features from raw text."""
        # Tokenize
        words_raw = text.split()
        words = [w.lower().strip(string.punctuation) for w in words_raw if w.strip(string.punctuation)]
        sentences = self._split_sentences(text)

        word_count = len(words)
        sentence_count = max(1, len(sentences))
        unique_words = len(set(words))
        ttr = unique_words / word_count if word_count > 0 else 0

        # Sentence lengths
        sent_lengths = [len(s.split()) for s in sentences]
        avg_sl = sum(sent_lengths) / len(sent_lengths) if sent_lengths else 0
        sl_variance = self._variance(sent_lengths)

        # Filler words
        filler_count = sum(1 for w in words if w in FILLER_WORDS)
        filler_density = filler_count / max(1, word_count)

        # Passive voice (simplified: "was/were/been + past participle")
        passive_count = len(re.findall(
            r"\b(was|were|been|is|are|has been|have been)\b\s+\w+ed\b",
            text.lower()))
        passive_ratio = passive_count / sentence_count

        # Punctuation density
        punct_count = sum(1 for c in text if c in string.punctuation)
        punct_density = punct_count / max(1, len(text))

        # Repetition score (bigram overlap)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
        repetition_score = repeated_bigrams / max(1, len(bigrams))

        # Perplexity proxy via bigram entropy
        perplexity_proxy = self._bigram_entropy(words)

        # AI / human marker detection
        text_lower = text.lower()
        ai_markers_found   = [m for m in AI_MARKER_PHRASES    if m in text_lower]
        human_markers_found= [m for m in HUMAN_MARKER_PHRASES if m in text_lower]

        # Average word length
        avg_word_len = (sum(len(w) for w in words) / max(1, word_count))

        # Anomalous features
        anomalies = []
        if avg_sl > 25:        anomalies.append(f"Very long sentences (avg {avg_sl:.1f} words)")
        if ttr > 0.85:         anomalies.append(f"Unusually high vocabulary diversity (TTR {ttr:.2f})")
        if filler_density < 0.005: anomalies.append("Almost no filler words — atypical of natural speech")
        if sl_variance < 3:    anomalies.append("Very uniform sentence lengths — low burstiness")
        if len(ai_markers_found) >= 3: anomalies.append(f"Multiple AI-typical phrases: {', '.join(ai_markers_found[:3])}")
        if passive_ratio > 0.3:anomalies.append(f"High passive voice ratio ({passive_ratio:.1%})")
        if repetition_score > 0.3: anomalies.append("High n-gram repetition detected")

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "unique_words": unique_words,
            "type_token_ratio": round(ttr, 4),
            "avg_sentence_length": round(avg_sl, 2),
            "sentence_len_variance": round(sl_variance, 2),
            "filler_word_density": round(filler_density, 4),
            "passive_voice_ratio": round(passive_ratio, 4),
            "punctuation_density": round(punct_density, 4),
            "repetition_score": round(repetition_score, 4),
            "perplexity_proxy": round(perplexity_proxy, 2),
            "ai_marker_count": len(ai_markers_found),
            "human_marker_count": len(human_markers_found),
            "avg_word_length": round(avg_word_len, 2),
            "text_length": len(text),
            "ai_markers_found": ai_markers_found[:5],
            "human_markers_found": human_markers_found[:5],
            "anomalies": anomalies,
        }

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences. Splits both on the normal '. ' (punctuation
        + whitespace) pattern AND on punctuation immediately followed by a
        capital letter with no space ('...technology.How are you...') — a
        very common typo that would otherwise make two real sentences count
        as one and incorrectly trip the short-text gate.
        """
        sentences = re.split(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    def _bigram_entropy(self, words: List[str]) -> float:
        """
        Compute bigram entropy as a perplexity proxy.
        Natural human speech has higher entropy (less predictable).
        AI text can be very predictable bigram-wise.
        """
        if len(words) < 2:
            return 5.0
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        total = len(bigrams)
        # Shannon entropy
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in bigram_freq.values() if c > 0
        )
        return round(entropy, 3)

    # ── Result builder ────────────────────────────────────────────────────────

    def _build_result(self, score, verdict, conf, stats, text) -> NLPAnalysisResult:
        return NLPAnalysisResult(
            score=score,
            verdict=verdict,
            confidence=conf,
            perplexity_proxy=stats["perplexity_proxy"],
            type_token_ratio=stats["type_token_ratio"],
            avg_sentence_length=stats["avg_sentence_length"],
            sentence_len_variance=stats["sentence_len_variance"],
            filler_word_density=stats["filler_word_density"],
            passive_voice_ratio=stats["passive_voice_ratio"],
            repetition_score=stats["repetition_score"],
            punctuation_density=stats["punctuation_density"],
            ai_marker_count=stats["ai_marker_count"],
            human_marker_count=stats["human_marker_count"],
            ai_generated_probability=0.0,  # set by caller
            word_count=stats["word_count"],
            sentence_count=stats["sentence_count"],
            unique_words=stats["unique_words"],
            avg_word_length=stats["avg_word_length"],
            text_length=stats["text_length"],
            top_ai_markers=stats.get("ai_markers_found", []),
            top_human_markers=stats.get("human_markers_found", []),
            anomalous_features=stats.get("anomalies", []),
        )

    UNCERTAIN_LOW = 35.0
    UNCERTAIN_HIGH = 65.0

    @classmethod
    def _score_to_verdict(cls, score: float) -> Tuple[str, float]:
        """Same calibrated 3-band approach as the face module's detector.py —
        a genuine 'uncertain' middle band instead of a forced binary split."""
        if score >= cls.UNCERTAIN_HIGH:
            span = 100 - cls.UNCERTAIN_HIGH
            conf = 60 + min((score - cls.UNCERTAIN_HIGH) / span, 1.0) * 39
            return "authentic", round(min(conf, 99.9), 2)
        if score <= cls.UNCERTAIN_LOW:
            span = cls.UNCERTAIN_LOW
            conf = 60 + min((cls.UNCERTAIN_LOW - score) / span, 1.0) * 39
            return "fake", round(min(conf, 99.9), 2)
        dist_from_edge = min(score - cls.UNCERTAIN_LOW, cls.UNCERTAIN_HIGH - score)
        band_half = (cls.UNCERTAIN_HIGH - cls.UNCERTAIN_LOW) / 2
        conf = 45 - (dist_from_edge / band_half) * 30
        return "uncertain", round(max(conf, 10.0), 2)


# Singleton
nlp_detector = NLPDetector()