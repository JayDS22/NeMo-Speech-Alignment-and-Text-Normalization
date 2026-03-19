"""Tests for QualityValidator."""

import json
import pytest
from src.quality_validator import QualityValidator, QualityMetrics, QualityReport


@pytest.fixture
def validator():
    return QualityValidator(
        wer_threshold=0.30,
        cer_threshold=0.15,
        min_confidence=0.60,
    )


class TestWER:
    def test_identical_strings(self, validator):
        wer = validator._compute_wer("hello world", "hello world")
        assert wer == 0.0

    def test_completely_different(self, validator):
        wer = validator._compute_wer("hello world", "foo bar")
        assert wer == 1.0

    def test_one_substitution(self, validator):
        wer = validator._compute_wer("the cat sat", "the dog sat")
        assert abs(wer - 1/3) < 0.01

    def test_empty_reference(self, validator):
        wer = validator._compute_wer("", "hello")
        assert wer == 1.0

    def test_both_empty(self, validator):
        wer = validator._compute_wer("", "")
        assert wer == 0.0


class TestCER:
    def test_identical(self, validator):
        cer = validator._compute_cer("hello", "hello")
        assert cer == 0.0

    def test_one_char_diff(self, validator):
        cer = validator._compute_cer("hello", "hallo")
        assert cer == pytest.approx(0.2, abs=0.01)

    def test_empty_reference(self, validator):
        cer = validator._manual_cer("", "abc")
        assert cer == 1.0


class TestValidation:
    def test_valid_alignment(self, validator):
        metrics = validator.validate_alignment(
            reference="hello world how are you",
            hypothesis="hello world how are you",
            word_confidences=[0.9, 0.85, 0.88, 0.92, 0.87],
            duration=3.0,
            aligned_duration=2.5,
            silence_ratio=0.17,
        )
        assert metrics.is_valid
        assert metrics.wer == 0.0
        assert len(metrics.flags) == 0

    def test_high_wer_flagged(self, validator):
        metrics = validator.validate_alignment(
            reference="the quick brown fox",
            hypothesis="a slow red dog",
            word_confidences=[0.7, 0.6, 0.5, 0.4],
            duration=2.0,
            aligned_duration=1.5,
        )
        assert not metrics.is_valid
        assert any("HIGH_WER" in f for f in metrics.flags)

    def test_low_confidence_flagged(self, validator):
        metrics = validator.validate_alignment(
            reference="hello world",
            hypothesis="hello world",
            word_confidences=[0.3, 0.2],
            duration=1.5,
            aligned_duration=1.0,
        )
        assert not metrics.is_valid
        assert any("LOW_CONFIDENCE" in f for f in metrics.flags)

    def test_too_short_flagged(self, validator):
        metrics = validator.validate_alignment(
            reference="hi",
            hypothesis="hi",
            word_confidences=[0.9],
            duration=0.1,
            aligned_duration=0.08,
        )
        assert any("TOO_SHORT" in f for f in metrics.flags)

    def test_high_silence_flagged(self, validator):
        metrics = validator.validate_alignment(
            reference="hello world test",
            hypothesis="hello world test",
            word_confidences=[0.8, 0.8, 0.8],
            duration=10.0,
            aligned_duration=1.0,
            silence_ratio=0.9,
        )
        assert any("HIGH_SILENCE" in f for f in metrics.flags)


class TestReport:
    def test_generate_report(self, validator):
        metrics_list = [
            QualityMetrics(wer=0.0, cer=0.0, avg_confidence=0.9, duration=3.0, is_valid=True),
            QualityMetrics(wer=0.5, cer=0.3, avg_confidence=0.4, duration=2.0, is_valid=False,
                           flags=["HIGH_WER:0.50"]),
            QualityMetrics(wer=0.1, cer=0.05, avg_confidence=0.85, duration=4.0, is_valid=True),
        ]
        report = validator.generate_report(metrics_list)
        assert report.total_files == 3
        assert report.valid_files == 2
        assert report.invalid_files == 1
        assert report.avg_wer == pytest.approx(0.2, abs=0.01)

    def test_empty_report(self, validator):
        report = validator.generate_report([])
        assert report.total_files == 0

    def test_save_report(self, validator, tmp_path):
        metrics_list = [
            QualityMetrics(wer=0.1, cer=0.05, avg_confidence=0.85, duration=3.0, is_valid=True),
        ]
        report = validator.generate_report(metrics_list, ["test.wav"])
        filepath = str(tmp_path / "report.json")
        validator.save_report(report, filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert "summary" in data
        assert data["summary"]["total_files"] == 1
