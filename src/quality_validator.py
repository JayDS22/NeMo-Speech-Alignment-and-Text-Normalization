"""
Quality Validation for ASR Dataset Preparation.

Computes WER, CER, confidence metrics, and generates
automated quality reports for aligned speech data.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

try:
    from jiwer import wer as compute_wer, cer as compute_cer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


@dataclass
class QualityMetrics:
    """Quality metrics for a single alignment."""
    wer: float = 0.0
    cer: float = 0.0
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    silence_ratio: float = 0.0
    num_words: int = 0
    num_low_confidence_words: int = 0
    duration: float = 0.0
    is_valid: bool = True
    flags: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Aggregate quality report for a dataset."""
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    avg_wer: float = 0.0
    avg_cer: float = 0.0
    avg_confidence: float = 0.0
    total_duration_hrs: float = 0.0
    valid_duration_hrs: float = 0.0
    flag_distribution: Dict[str, int] = field(default_factory=dict)
    per_file_metrics: List[Dict] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class QualityValidator:
    """
    Automated quality validation for aligned ASR datasets.

    Features:
        - WER/CER computation
        - Per-word confidence thresholding
        - Silence ratio checks
        - Automated flagging of problematic alignments
        - Batch QA report generation
    """

    def __init__(
        self,
        wer_threshold: float = 0.30,
        cer_threshold: float = 0.15,
        min_confidence: float = 0.60,
        min_words_per_segment: int = 2,
        max_silence_ratio: float = 0.80,
    ):
        self.wer_threshold = wer_threshold
        self.cer_threshold = cer_threshold
        self.min_confidence = min_confidence
        self.min_words_per_segment = min_words_per_segment
        self.max_silence_ratio = max_silence_ratio

    def validate_alignment(
        self,
        reference: str,
        hypothesis: str,
        word_confidences: List[float],
        duration: float,
        aligned_duration: float,
        silence_ratio: float = 0.0,
    ) -> QualityMetrics:
        """
        Validate a single alignment result.

        Args:
            reference: Original transcript.
            hypothesis: Aligned/normalized transcript.
            word_confidences: Per-word confidence scores.
            duration: Total audio duration.
            aligned_duration: Duration of aligned speech.
            silence_ratio: Ratio of silence to total duration.

        Returns:
            QualityMetrics with validation results.
        """
        flags = []

        # Compute WER/CER
        wer_score = self._compute_wer(reference, hypothesis)
        cer_score = self._compute_cer(reference, hypothesis)

        # Confidence stats
        if word_confidences:
            avg_conf = float(np.mean(word_confidences))
            min_conf = float(np.min(word_confidences))
            max_conf = float(np.max(word_confidences))
            n_low_conf = sum(1 for c in word_confidences if c < self.min_confidence)
        else:
            avg_conf = 0.0
            min_conf = 0.0
            max_conf = 0.0
            n_low_conf = 0

        num_words = len(reference.split())

        # Flag checks
        if wer_score > self.wer_threshold:
            flags.append(f"HIGH_WER:{wer_score:.2f}")

        if cer_score > self.cer_threshold:
            flags.append(f"HIGH_CER:{cer_score:.2f}")

        if avg_conf < self.min_confidence:
            flags.append(f"LOW_CONFIDENCE:{avg_conf:.2f}")

        if num_words < self.min_words_per_segment:
            flags.append(f"TOO_FEW_WORDS:{num_words}")

        if silence_ratio > self.max_silence_ratio:
            flags.append(f"HIGH_SILENCE:{silence_ratio:.2f}")

        if n_low_conf > num_words * 0.5:
            flags.append(f"MANY_LOW_CONF_WORDS:{n_low_conf}/{num_words}")

        if duration < 0.3:
            flags.append("TOO_SHORT")

        if duration > 30.0:
            flags.append("TOO_LONG")

        is_valid = len(flags) == 0

        return QualityMetrics(
            wer=wer_score,
            cer=cer_score,
            avg_confidence=avg_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            silence_ratio=silence_ratio,
            num_words=num_words,
            num_low_confidence_words=n_low_conf,
            duration=duration,
            is_valid=is_valid,
            flags=flags,
        )

    def generate_report(
        self,
        metrics_list: List[QualityMetrics],
        filepaths: Optional[List[str]] = None,
    ) -> QualityReport:
        """
        Generate aggregate quality report from a list of metrics.

        Args:
            metrics_list: Per-file quality metrics.
            filepaths: Optional file paths for per-file details.

        Returns:
            QualityReport with aggregate and per-file statistics.
        """
        if not metrics_list:
            return QualityReport()

        if filepaths is None:
            filepaths = [f"file_{i}" for i in range(len(metrics_list))]

        total = len(metrics_list)
        valid = sum(1 for m in metrics_list if m.is_valid)
        invalid = total - valid

        wers = [m.wer for m in metrics_list]
        cers = [m.cer for m in metrics_list]
        confs = [m.avg_confidence for m in metrics_list]
        durations = [m.duration for m in metrics_list]

        # Flag distribution
        flag_dist = {}
        for m in metrics_list:
            for flag in m.flags:
                flag_type = flag.split(":")[0]
                flag_dist[flag_type] = flag_dist.get(flag_type, 0) + 1

        # Per-file details
        per_file = []
        for filepath, m in zip(filepaths, metrics_list):
            per_file.append({
                "filepath": filepath,
                "wer": round(m.wer, 4),
                "cer": round(m.cer, 4),
                "confidence": round(m.avg_confidence, 4),
                "duration": round(m.duration, 2),
                "is_valid": m.is_valid,
                "flags": m.flags,
            })

        return QualityReport(
            total_files=total,
            valid_files=valid,
            invalid_files=invalid,
            avg_wer=round(float(np.mean(wers)), 4),
            avg_cer=round(float(np.mean(cers)), 4),
            avg_confidence=round(float(np.mean(confs)), 4),
            total_duration_hrs=round(sum(durations) / 3600, 4),
            valid_duration_hrs=round(
                sum(d for d, m in zip(durations, metrics_list) if m.is_valid) / 3600, 4
            ),
            flag_distribution=flag_dist,
            per_file_metrics=per_file,
        )

    def save_report(self, report: QualityReport, filepath: str):
        """Save quality report to JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        report_dict = {
            "summary": {
                "total_files": report.total_files,
                "valid_files": report.valid_files,
                "invalid_files": report.invalid_files,
                "acceptance_rate": round(
                    report.valid_files / max(report.total_files, 1) * 100, 1
                ),
                "avg_wer": report.avg_wer,
                "avg_cer": report.avg_cer,
                "avg_confidence": report.avg_confidence,
                "total_duration_hrs": report.total_duration_hrs,
                "valid_duration_hrs": report.valid_duration_hrs,
                "timestamp": report.timestamp,
            },
            "flag_distribution": report.flag_distribution,
            "per_file_metrics": report.per_file_metrics,
        }

        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)

    # ── WER / CER ─────────────────────────────────────────────────

    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate."""
        if HAS_JIWER:
            try:
                return float(compute_wer(reference, hypothesis))
            except Exception:
                pass
        return self._manual_wer(reference, hypothesis)

    def _compute_cer(self, reference: str, hypothesis: str) -> float:
        """Compute Character Error Rate."""
        if HAS_JIWER:
            try:
                return float(compute_cer(reference, hypothesis))
            except Exception:
                pass
        return self._manual_cer(reference, hypothesis)

    @staticmethod
    def _manual_wer(reference: str, hypothesis: str) -> float:
        """Manual WER computation using edit distance."""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1):
            d[i, 0] = i
        for j in range(len(hyp_words) + 1):
            d[0, j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,      # deletion
                        d[i, j - 1] + 1,      # insertion
                        d[i - 1, j - 1] + 1,  # substitution
                    )

        return float(d[len(ref_words), len(hyp_words)] / len(ref_words))

    @staticmethod
    def _manual_cer(reference: str, hypothesis: str) -> float:
        """Manual CER computation using edit distance."""
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())

        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0

        if not hyp_chars:
            return 1.0

        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        for i in range(len(ref_chars) + 1):
            d[i, 0] = i
        for j in range(len(hyp_chars) + 1):
            d[0, j] = j

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(
                        d[i - 1, j] + 1,
                        d[i, j - 1] + 1,
                        d[i - 1, j - 1] + 1,
                    )

        return float(d[len(ref_chars), len(hyp_chars)] / len(ref_chars))
