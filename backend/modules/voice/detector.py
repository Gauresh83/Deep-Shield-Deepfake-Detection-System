"""
M3-ID Voice Clone Detection Module — Part 3
=============================================
Architecture:
  1. Audio loading & pre-processing  (librosa)
  2. MFCC feature extraction         (13 coefficients + delta + delta-delta)
  3. Spectrogram & chroma features
  4. Classification model            (CNN-LSTM when models present, 
                                      rule-based heuristics as fallback)
  5. Result packaging with full details

Real model weights (ASVspoof 2019 trained) can be dropped into:
  backend/modules/voice/weights/voice_model.pt

When weights are absent the module runs an advanced signal-analysis
heuristic that examines real audio properties — NOT random numbers.
"""

import os
import time
import json
import math
import struct
import wave
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class VoiceAnalysisResult:
    score: float                     # 0-100  (higher = more authentic)
    verdict: str                     # authentic | suspicious | fake
    confidence: float                # 0-100
    processing_time_ms: int

    # Feature details
    mfcc_anomaly_score: float        # 0-100  (higher = more anomalous)
    spectral_consistency: float      # 0-100
    pitch_naturalness: float         # 0-100
    temporal_coherence: float        # 0-100
    clone_probability: float         # 0-100

    # Signal metadata
    duration_seconds: float
    sample_rate: int
    num_channels: int
    bit_depth: int
    frame_count: int

    # Model info
    model_used: str
    features_extracted: int
    model_loaded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoiceDetector:
    """
    Voice Clone Detection Engine.

    When PyTorch + librosa are installed and model weights exist:
        → Runs full CNN-LSTM pipeline on MFCC features (ASVspoof 2019 trained)

    Otherwise:
        → Runs signal-level heuristic analysis on raw PCM / WAV data.
          This still extracts real audio statistics — not mock random numbers.
    """

    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "voice_model.pt")

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.librosa_available = False
        self.torch_available = False
        self._try_load_libraries()

    def _try_load_libraries(self):
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            pass

        try:
            import torch
            self.torch_available = True
            if os.path.exists(self.WEIGHTS_PATH):
                self._load_model()
        except ImportError:
            pass

    def _load_model(self):
        """Load trained CNN-LSTM weights if available."""
        try:
            import torch
            # Model architecture (matches ASVspoof training setup)
            self.model = self._build_model()
            state = torch.load(self.WEIGHTS_PATH, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_loaded = True
            print("[VoiceDetector] Loaded trained weights from", self.WEIGHTS_PATH)
        except Exception as e:
            print(f"[VoiceDetector] Could not load weights: {e}")
            self.model_loaded = False

    def _build_model(self):
        """
        CNN-LSTM architecture for voice spoof detection.
        Input:  (batch, 1, 13, T)  — MFCC spectrogram
        Output: (batch, 1)          — spoof probability
        """
        import torch.nn as nn

        class VoiceSpoofCNNLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d((1, 2)),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d((1, 2)),
                    nn.Dropout2d(0.25),
                )
                self.lstm = nn.LSTM(
                    input_size=64 * 13,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.3,
                    bidirectional=True,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                # x: (B, 1, 13, T)
                x = self.cnn(x)              # (B, 64, 13, T//4)
                B, C, H, W = x.shape
                x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B, T//4, 64*13)
                out, _ = self.lstm(x)
                x = out[:, -1, :]            # last time step
                return self.classifier(x)

        return VoiceSpoofCNNLSTM()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_file(self, file_path: str) -> VoiceAnalysisResult:
        """Main entry point — analyze an audio file and return results."""
        start = time.time()

        if self.librosa_available and self.model_loaded:
            result = self._analyze_with_model(file_path)
        elif self.librosa_available:
            result = self._analyze_with_librosa(file_path)
        else:
            result = self._analyze_with_wave(file_path)

        result.processing_time_ms = int((time.time() - start) * 1000)
        return result

    def _analyze_with_model(self, file_path: str) -> VoiceAnalysisResult:
        """Full CNN-LSTM inference pipeline."""
        import torch
        import librosa
        import numpy as np

        y, sr = librosa.load(file_path, sr=16000, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (39, T)

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)

        # Reshape for model: (1, 1, 13, T)  — use only base MFCCs
        x = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32)

        with torch.no_grad():
            spoof_prob = self.model(x).item()  # 0=real, 1=spoof

        authenticity_score = round((1 - spoof_prob) * 100, 2)

        # Additional signal features for explainability
        spectral = self._spectral_consistency(y, sr)
        pitch = self._pitch_naturalness(y, sr)
        temporal = self._temporal_coherence(y, sr)

        verdict, confidence = self._score_to_verdict(authenticity_score)

        return VoiceAnalysisResult(
            score=authenticity_score,
            verdict=verdict,
            confidence=confidence,
            processing_time_ms=0,
            mfcc_anomaly_score=round(spoof_prob * 100, 2),
            spectral_consistency=spectral,
            pitch_naturalness=pitch,
            temporal_coherence=temporal,
            clone_probability=round(spoof_prob * 100, 2),
            duration_seconds=round(len(y) / sr, 2),
            sample_rate=sr,
            num_channels=1,
            bit_depth=16,
            frame_count=len(y),
            model_used="CNN-LSTM (ASVspoof 2019 trained)",
            features_extracted=39 * mfcc.shape[1],
            model_loaded=True,
        )

    def _analyze_with_librosa(self, file_path: str) -> VoiceAnalysisResult:
        """Librosa-based heuristic analysis (no trained weights)."""
        import librosa
        import numpy as np

        y, sr = librosa.load(file_path, sr=16000, mono=True)
        duration = len(y) / sr

        # ── MFCC features ──────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Anomaly indicators for cloned/synthesized voice:
        # 1. MFCC variance — TTS voices have lower variance
        mfcc_var = float(np.var(mfcc))
        mfcc_var_norm = min(mfcc_var / 200.0, 1.0)  # normalise to 0-1

        # 2. MFCC delta smoothness — clones are over-smooth
        delta_roughness = float(np.std(mfcc_delta))
        delta_norm = min(delta_roughness / 8.0, 1.0)

        # 3. Spectral rolloff consistency
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_std = float(np.std(rolloff))
        rolloff_norm = min(rolloff_std / 2000.0, 1.0)

        # 4. Zero crossing rate (high = more natural consonants)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = float(np.mean(zcr))
        zcr_norm = min(zcr_mean / 0.15, 1.0)

        # 5. Spectral centroid stability
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-8))
        centroid_norm = min(centroid_cv / 0.3, 1.0)

        # 6. RMS energy pattern (TTS = uniform, real = variable)
        rms = librosa.feature.rms(y=y)
        rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-8))
        rms_norm = min(rms_cv / 0.8, 1.0)

        # Combine heuristics — weighted authenticity signal
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        signals = [mfcc_var_norm, delta_norm, rolloff_norm, zcr_norm, centroid_norm, rms_norm]
        raw_score = sum(w * s for w, s in zip(weights, signals))
        score = round(raw_score * 100, 2)

        spectral = self._spectral_consistency(y, sr)
        pitch = self._pitch_naturalness(y, sr)
        temporal = self._temporal_coherence(y, sr)
        verdict, confidence = self._score_to_verdict(score)

        return VoiceAnalysisResult(
            score=score,
            verdict=verdict,
            confidence=confidence,
            processing_time_ms=0,
            mfcc_anomaly_score=round(100 - score, 2),
            spectral_consistency=spectral,
            pitch_naturalness=pitch,
            temporal_coherence=temporal,
            clone_probability=round(100 - score, 2),
            duration_seconds=round(duration, 2),
            sample_rate=sr,
            num_channels=1,
            bit_depth=16,
            frame_count=len(y),
            model_used="Librosa Heuristic (install model weights for CNN-LSTM)",
            features_extracted=6,
            model_loaded=False,
        )

    def _analyze_with_wave(self, file_path: str) -> VoiceAnalysisResult:
        """Fallback: raw WAV analysis without librosa."""
        try:
            return self._raw_wav_analysis(file_path)
        except Exception:
            # If not a WAV file or parse fails, return safe defaults
            return VoiceAnalysisResult(
                score=50.0, verdict="suspicious", confidence=20.0,
                processing_time_ms=0, mfcc_anomaly_score=50.0,
                spectral_consistency=50.0, pitch_naturalness=50.0,
                temporal_coherence=50.0, clone_probability=50.0,
                duration_seconds=0, sample_rate=0, num_channels=0,
                bit_depth=0, frame_count=0,
                model_used="Fallback (install librosa for full analysis)",
                features_extracted=0, model_loaded=False,
            )

    def _raw_wav_analysis(self, file_path: str) -> VoiceAnalysisResult:
        """Analyze raw PCM from WAV file — no external libraries needed."""
        with wave.open(file_path, 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        duration = n_frames / sr

        # Decode PCM samples
        fmt = {1: 'b', 2: 'h', 4: 'i'}.get(sampwidth, 'h')
        samples = list(struct.unpack(f"{n_frames * channels}{fmt}", raw_bytes))
        if channels > 1:
            samples = samples[::channels]  # mono downmix

        n = len(samples)
        if n == 0:
            raise ValueError("Empty audio file")

        # Basic statistics
        mean_amp = sum(abs(s) for s in samples) / n
        max_amp = max(abs(s) for s in samples)
        max_possible = (2 ** (sampwidth * 8 - 1)) - 1

        # Amplitude variation (real speech is highly variable)
        mid = n // 2
        first_half_rms = math.sqrt(sum(s*s for s in samples[:mid]) / mid)
        second_half_rms = math.sqrt(sum(s*s for s in samples[mid:]) / (n - mid))
        energy_variation = abs(first_half_rms - second_half_rms) / (max(first_half_rms, second_half_rms) + 1)

        # Zero crossing rate
        zcr = sum(1 for i in range(1, n) if (samples[i] >= 0) != (samples[i-1] >= 0)) / n

        # Dynamic range usage (TTS voices often have narrow dynamic range)
        dynamic_ratio = mean_amp / (max_possible + 1)

        # Rough silence ratio
        threshold = max_possible * 0.01
        silence_ratio = sum(1 for s in samples if abs(s) < threshold) / n

        # Score heuristics
        # Cloned/TTS audio tends to: low dynamic range variation, uniform energy, high silence uniformity
        energy_var_score = min(energy_variation * 2, 1.0)
        zcr_score = min(zcr * 20, 1.0)
        dynamic_score = min(dynamic_ratio * 5, 1.0)
        silence_score = 1.0 - silence_ratio

        raw = (energy_var_score * 0.3 + zcr_score * 0.3 + dynamic_score * 0.2 + silence_score * 0.2)
        score = round(raw * 100, 2)
        verdict, confidence = self._score_to_verdict(score)

        return VoiceAnalysisResult(
            score=score,
            verdict=verdict,
            confidence=confidence,
            processing_time_ms=0,
            mfcc_anomaly_score=round(100 - score, 2),
            spectral_consistency=round(energy_var_score * 100, 2),
            pitch_naturalness=round(zcr_score * 100, 2),
            temporal_coherence=round(dynamic_score * 100, 2),
            clone_probability=round(100 - score, 2),
            duration_seconds=round(duration, 2),
            sample_rate=sr,
            num_channels=channels,
            bit_depth=sampwidth * 8,
            frame_count=n_frames,
            model_used="Raw WAV Heuristic (install librosa for full MFCC analysis)",
            features_extracted=4,
            model_loaded=False,
        )

    # ── Helper signal analysis functions ──────────────────────────────────────

    def _spectral_consistency(self, y, sr) -> float:
        """Measure spectral consistency — clones show unnatural uniformity."""
        try:
            import librosa, numpy as np
            spec = librosa.stft(y)
            mag = abs(spec)
            frame_energies = mag.sum(axis=0)
            cv = float(np.std(frame_energies) / (np.mean(frame_energies) + 1e-8))
            return round(min(cv / 0.5, 1.0) * 100, 2)
        except Exception:
            return 50.0

    def _pitch_naturalness(self, y, sr) -> float:
        """Measure pitch naturalness using fundamental frequency variation."""
        try:
            import librosa, numpy as np
            f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=500, sr=sr)
            voiced_f0 = f0[voiced]
            if len(voiced_f0) < 2:
                return 50.0
            f0_std = float(np.std(voiced_f0))
            # Natural speech: 15-60 Hz std; TTS: very low std
            naturalness = min(f0_std / 30.0, 1.0)
            return round(naturalness * 100, 2)
        except Exception:
            return 50.0

    def _temporal_coherence(self, y, sr) -> float:
        """Check temporal coherence — clones often have unnatural transitions."""
        try:
            import librosa, numpy as np
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) < 2:
                return 50.0
            intervals = np.diff(onset_frames)
            cv = float(np.std(intervals) / (np.mean(intervals) + 1e-8))
            return round(min(cv / 0.8, 1.0) * 100, 2)
        except Exception:
            return 50.0

    @staticmethod
    def _score_to_verdict(score: float):
        if score >= 70:
            verdict = "authentic"
            confidence = min((score - 70) / 30 * 100, 99.9)
        elif score >= 45:
            verdict = "suspicious"
            confidence = 50.0
        else:
            verdict = "fake"
            confidence = min((45 - score) / 45 * 100, 99.9)
        return verdict, round(confidence, 2)


# Singleton instance
voice_detector = VoiceDetector()
