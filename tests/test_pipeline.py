"""Integration tests for the full pipeline."""

import pytest
import numpy as np

from src.pipeline import SpeechAlignmentPipeline
from src.audio_preprocessor import AudioPreprocessor


@pytest.fixture
def pipeline():
    return SpeechAlignmentPipeline()


@pytest.fixture
def sample_audio():
    preprocessor = AudioPreprocessor()
    audio, sr = preprocessor.generate_synthetic_audio(duration_sec=3.0)
    return audio, sr


class TestPipelineIntegration:
    def test_process_from_audio(self, pipeline, sample_audio):
        audio, sr = sample_audio
        transcript = "Dr. Smith spent $42 on 5kg of apples at 3:30 PM on 01/15/2024"

        alignment, norm_result, metrics = pipeline.process_from_audio(
            audio, sr, transcript
        )

        # Check normalization happened
        assert "doctor" in norm_result.normalized
        assert "dollar" in norm_result.normalized
        assert norm_result.num_changes > 0

        # Check alignment produced words
        assert len(alignment.words) > 0
        assert alignment.total_duration > 0

        # Check quality metrics
        assert isinstance(metrics.wer, float)
        assert isinstance(metrics.cer, float)
        assert metrics.duration > 0

    def test_manifest_generation(self, pipeline, sample_audio):
        audio, sr = sample_audio
        transcript = "hello world test sentence"

        alignment, _, _ = pipeline.process_from_audio(audio, sr, transcript)
        manifest = alignment.to_manifest_entry()

        assert "audio_filepath" in manifest
        assert "text" in manifest
        assert "duration" in manifest
        assert "words" in manifest
        assert len(manifest["words"]) > 0
        assert all("start" in w and "end" in w for w in manifest["words"])

    def test_empty_transcript(self, pipeline, sample_audio):
        audio, sr = sample_audio
        alignment, norm_result, metrics = pipeline.process_from_audio(
            audio, sr, ""
        )
        assert len(alignment.words) == 0

    def test_long_transcript(self, pipeline, sample_audio):
        audio, sr = sample_audio
        transcript = " ".join(["word"] * 50)
        alignment, norm_result, metrics = pipeline.process_from_audio(
            audio, sr, transcript
        )
        assert len(alignment.words) == 50

    def test_normalization_types(self, pipeline, sample_audio):
        audio, sr = sample_audio
        test_cases = [
            ("I have 42 items", "forty"),
            ("Meeting at 3:30 PM", "three thirty"),
            ("Weighs 5kg", "five kilogram"),
        ]
        for transcript, expected_substring in test_cases:
            _, norm_result, _ = pipeline.process_from_audio(
                audio, sr, transcript
            )
            assert expected_substring in norm_result.normalized, \
                f"Expected '{expected_substring}' in '{norm_result.normalized}'"
