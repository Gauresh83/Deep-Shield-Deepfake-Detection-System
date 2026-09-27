"""
AI-Generated Image Detector — Detector 2 (Phase 1)
=====================================================
Detects fully AI-generated / diffusion-model images (Midjourney, Stable
Diffusion, Flux, DALL-E, and older GAN generators) — as distinct from
Detector 1 (StyleGAN2-era synthetic-face detector) and Detector 3
(face-swap/manipulation detector on otherwise-real footage).

Unlike Detector 1/3, this one takes the FULL IMAGE as input, not just the
face crop — diffusion-generation artifacts (frequency-domain signatures,
global coherence issues) aren't confined to the face region, so cropping
to just the face would throw away most of the useful signal.

Architecture: ResNet18 (same choice as Detector 3, for consistency and to
keep training/inference compute modest — see PHASE1_ARCHITECTURE.md §4,
which suggested EfficientNet-B4 as a starting point; ResNet18 was used
instead to match the established, already-working pattern from Detector 3
and keep GPU-quota usage predictable).

Training data (practical, scoped-down — see conversation notes): this is
NOT the full official AI-GenBench benchmark (which requires reconstructing
a multi-source real-image pipeline from COCO+ImageNet+RAISE+LAION via the
official dataset_creation scripts — impractical on a Kaggle free-tier
budget). This uses:
  - FAKE: a subset of lrzpellegrini/AI-GenBench-fake_part (HuggingFace) —
    covers modern diffusion generators as well as older GANs
  - REAL: a subset of COCO val2017 (public, no-approval-needed)
This is an honest, explicitly-scoped substitute for the full benchmark,
not a claim of reproducing it — say so plainly if asked (e.g. in a viva).

Drop trained weights at:
    backend/modules/face/weights/ai_gen_model.pth
(a plain torchvision state_dict, same convention as faceswap_model.pth —
NOT a HuggingFace save_pretrained() folder.)
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass
class AIGenResult:
    status: str = "ANALYZED"        # ANALYZED | NOT_LOADED
    score: float = 50.0             # 0-100, higher = more likely a real camera photo
    verdict: str = "uncertain"      # authentic | uncertain | ai_generated
    confidence: float = 0.0
    model_used: str = "N/A"
    model_loaded: bool = False
    message: str = ""


class AIGenDetector:
    """
    Detector 2: AI-generated / diffusion-image classifier.
    Same fail-open convention as FaceSwapDetector — never crashes the app
    if weights/torch/torchvision are missing, just reports model_loaded=False.
    """

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "ai_gen_model.pth")
    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Same calibrated-band convention as detector.py / faceswap_detector.py
    UNCERTAIN_LOW = 35.0
    UNCERTAIN_HIGH = 65.0

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.torch_available = False
        self.torchvision_available = False
        self._transform = None
        self._try_load()

    def _try_load(self):
        try:
            import torch  # noqa: F401
            self.torch_available = True
        except ImportError:
            return
        try:
            import torchvision  # noqa: F401
            self.torchvision_available = True
        except ImportError:
            return

        if not os.path.exists(self.WEIGHTS_PATH):
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)  # 0=real, 1=ai_generated

            state = torch.load(self.WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            self.model = model
            self._transform = transforms.Compose([
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])
            self.model_loaded = True
            print(f"[AIGenDetector] ResNet18 weights loaded from {self.WEIGHTS_PATH}.")
        except Exception as e:
            print(f"[AIGenDetector] Could not load weights: {e}")
            self.model = None
            self.model_loaded = False

    def analyze(self, full_image_bgr, face_crop_bgr=None) -> AIGenResult:
        """
        full_image_bgr: the WHOLE uploaded image (OpenCV BGR array), not just
        the face crop — see module docstring for why. face_crop_bgr is
        accepted for interface-compatibility with the architecture doc but
        currently unused (kept as an auxiliary-signal hook for a future,
        two-input version of this model).
        """
        if not self.model_loaded:
            return AIGenResult(
                status="NOT_LOADED", verdict="uncertain", score=0.0, confidence=0.0,
                model_used="N/A — ai_gen_model.pth not found or torch/torchvision missing",
                model_loaded=False,
                message="AI-generated-image detector weights are not loaded; this signal is unavailable.",
            )

        import torch
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(full_image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            ai_gen_prob = probs[1].item()  # class 1 = ai_generated

        score = round((1 - ai_gen_prob) * 100, 2)  # higher score = more real
        verdict, confidence = self._score_to_verdict(score)

        return AIGenResult(
            status="ANALYZED", score=score, verdict=verdict, confidence=confidence,
            model_used="ResNet18 (AI-GenBench subset + COCO val2017)",
            model_loaded=True,
            message="" if verdict != "uncertain" else
                    "AI-generation signal is inconclusive for this image.",
        )

    @classmethod
    def _score_to_verdict(cls, score: float) -> Tuple[str, float]:
        """Identical calibrated 3-band logic to the other detectors — verdict
        value is 'ai_generated' instead of 'fake' to keep this detector's
        specialty explicit in the fused breakdown."""
        if score >= cls.UNCERTAIN_HIGH:
            span = 100 - cls.UNCERTAIN_HIGH
            conf = 60 + min((score - cls.UNCERTAIN_HIGH) / span, 1.0) * 39
            return "authentic", round(min(conf, 99.9), 2)
        if score <= cls.UNCERTAIN_LOW:
            span = cls.UNCERTAIN_LOW
            conf = 60 + min((cls.UNCERTAIN_LOW - score) / span, 1.0) * 39
            return "ai_generated", round(min(conf, 99.9), 2)
        dist_from_edge = min(score - cls.UNCERTAIN_LOW, cls.UNCERTAIN_HIGH - score)
        band_half = (cls.UNCERTAIN_HIGH - cls.UNCERTAIN_LOW) / 2
        conf = 45 - (dist_from_edge / band_half) * 30
        return "uncertain", round(max(conf, 10.0), 2)


# Singleton, same pattern as face_detector/faceswap_detector/voice_detector/nlp_detector
ai_gen_detector = AIGenDetector()