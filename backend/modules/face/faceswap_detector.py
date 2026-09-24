"""
Face-Swap / Manipulation Detector — Detector 3 (Phase 2)
===========================================================
Fine-tuned ResNet18 (ImageNet-pretrained backbone, layer4 + fc unfrozen)
trained on FaceForensics++ (c40, Deepfakes + FaceSwap, video-ID-level
leakage-safe split). Binary output: real (camera-captured face) vs fake
(face-swapped/manipulated face) — this model does NOT distinguish which
manipulation method was used (Deepfakes vs FaceSwap), only real-vs-fake.

This is a SEPARATE model/file from FaceDetector (detector.py, "Detector 1")
by design — see PHASE1_ARCHITECTURE.md §9. Detector 1 specializes in
fully-synthetic/GAN faces (trained on Real-and-Fake-Faces); this one
specializes in face-swap manipulation on otherwise-real videos/photos
(the "CEO video-call fraud" scenario). They are meant to run side by side
and be combined by fusion.py, not to replace one another.

Drop trained weights at:
    backend/modules/face/weights/faceswap_model.pth
(a plain state_dict saved with torch.save(model.state_dict(), ...) —
NOT a HuggingFace-style folder, this is a torchvision model.)
"""
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class FaceSwapResult:
    status: str = "ANALYZED"        # ANALYZED | NOT_LOADED
    score: float = 50.0             # 0-100, higher = more likely real/untampered
    verdict: str = "uncertain"      # authentic | uncertain | fake
    confidence: float = 0.0
    model_used: str = "N/A"
    model_loaded: bool = False
    message: str = ""


class FaceSwapDetector:
    """
    Detector 3: face-swap/manipulation classifier.
    Mirrors FaceDetector's loading conventions (see modules/face/detector.py)
    so it fails the same way (fails open with model_loaded=False, never
    crashes the app) when weights/torch/torchvision aren't available.
    """

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "faceswap_model.pth")
    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Same calibrated-band convention as detector.py — see that file's
    # _score_to_verdict for the full rationale.
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

            # Must exactly match the training script's architecture:
            # resnet18 backbone with fc replaced by a 2-class linear head.
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)

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
            print(f"[FaceSwapDetector] ResNet18 weights loaded from {self.WEIGHTS_PATH}.")
        except Exception as e:
            print(f"[FaceSwapDetector] Could not load weights: {e}")
            self.model = None
            self.model_loaded = False

    def analyze_face_crop(self, face_crop_bgr) -> FaceSwapResult:
        """
        face_crop_bgr: an OpenCV BGR numpy array — the ALREADY-DETECTED,
        ALREADY-QUALITY-GATED face crop from FaceDetector's pipeline. This
        method does not do its own face detection or quality gate — it
        reuses whatever FaceDetector (Phase 0) already established, so a
        no-face or low-quality image never reaches this model either.
        """
        if not self.model_loaded:
            return FaceSwapResult(
                status="NOT_LOADED", verdict="uncertain", score=0.0, confidence=0.0,
                model_used="N/A — faceswap_model.pth not found or torch/torchvision missing",
                model_loaded=False,
                message="Face-swap detector weights are not loaded; this signal is unavailable.",
            )

        import torch
        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self._transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            fake_prob = probs[1].item()  # class 1 = fake, per label_map in training

        score = round((1 - fake_prob) * 100, 2)  # higher score = more real
        verdict, confidence = self._score_to_verdict(score)

        return FaceSwapResult(
            status="ANALYZED", score=score, verdict=verdict, confidence=confidence,
            model_used="ResNet18 (FaceForensics++ c40, Deepfakes+FaceSwap)",
            model_loaded=True,
            message="" if verdict != "uncertain" else
                    "Face-swap signal is inconclusive for this face.",
        )

    @classmethod
    def _score_to_verdict(cls, score: float) -> Tuple[str, float]:
        """Identical calibrated 3-band logic to FaceDetector._score_to_verdict —
        kept as a literal copy rather than a shared import so this module has
        no dependency on detector.py and can be dropped in/out independently."""
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


# Singleton, same pattern as face_detector/voice_detector/nlp_detector
faceswap_detector = FaceSwapDetector()