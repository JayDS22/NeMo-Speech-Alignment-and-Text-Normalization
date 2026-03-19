"""Tests for AudioPreprocessor."""

import os
import pytest
import numpy as np

from src.audio_preprocessor import AudioPreprocessor, AudioSegment, AudioInfo


@pytest.fixture
def preprocessor():
    return AudioPreprocessor(target_sample_rate=16000)


class TestSyntheticAudio:
    def test_generate_synthetic(self, preprocessor):
        audio, sr = preprocessor.generate_synthetic_audio(duration_sec=2.0)
        assert sr == 16000
        assert len(audio) == 32000
        assert audio.dtype == np.float32

    def test_audio_range(self, preprocessor):
        audio, sr = preprocessor.generate_synthetic_audio()
        assert np.max(np.abs(audio)) <= 1.1  # Allow small overshoot from noise


class TestResampling:
    def test_same_rate(self, preprocessor):
        audio = np.random.randn(16000).astype(np.float32)
        resampled = preprocessor.resample(audio, 16000)
        assert len(resampled) == len(audio)

    def test_downsample(self, preprocessor):
        audio = np.random.randn(44100).astype(np.float32)
        resampled = preprocessor.resample(audio, 44100)
        expected_len = int(44100 * 16000 / 44100)
        assert abs(len(resampled) - expected_len) <= 1

    def test_upsample(self, preprocessor):
        audio = np.random.randn(8000).astype(np.float32)
        resampled = preprocessor.resample(audio, 8000)
        expected_len = int(8000 * 16000 / 8000)
        assert abs(len(resampled) - expected_len) <= 1


class TestVAD:
    def test_detect_speech(self, preprocessor):
        sr = 16000
        # Create audio with speech-like energy in the middle
        audio = np.zeros(sr * 3, dtype=np.float32)  # 3 seconds
        audio[sr:2*sr] = np.random.randn(sr).astype(np.float32) * 0.5  # Speech in middle

        regions = preprocessor.detect_voice_activity(audio, sr, energy_threshold=0.01)
        assert len(regions) >= 1
        # Speech region should be roughly in the middle
        assert regions[0][0] >= 0.5
        assert regions[0][1] <= 2.5

    def test_silence_only(self, preprocessor):
        sr = 16000
        audio = np.zeros(sr * 2, dtype=np.float32)
        regions = preprocessor.detect_voice_activity(audio, sr)
        assert len(regions) == 0

    def test_continuous_speech(self, preprocessor):
        sr = 16000
        # Use sine wave for reliable energy detection
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.05 * np.random.randn(sr * 2)).astype(np.float32)
        regions = preprocessor.detect_voice_activity(audio, sr, energy_threshold=0.01)
        assert len(regions) >= 1


class TestSegmentation:
    def test_segment_by_silence(self, preprocessor):
        sr = 16000
        # Create audio with 2 speech segments separated by silence
        audio = np.zeros(sr * 5, dtype=np.float32)
        audio[int(sr*0.5):int(sr*1.5)] = np.random.randn(sr).astype(np.float32) * 0.3
        audio[int(sr*3.0):int(sr*4.0)] = np.random.randn(sr).astype(np.float32) * 0.3

        segments = preprocessor.segment_by_silence(audio, sr)
        assert len(segments) >= 1
        for seg in segments:
            assert isinstance(seg, AudioSegment)
            assert seg.duration > 0


class TestMelSpectrogram:
    def test_extract_mel(self, preprocessor):
        audio, sr = preprocessor.generate_synthetic_audio(duration_sec=1.0)
        mel = preprocessor.extract_mel_spectrogram(audio, sr, n_mels=80)
        assert mel.shape[0] == 80 or mel.ndim == 2
        assert mel.shape[1] > 0


class TestSaveLoad:
    def test_save_and_load_wav(self, preprocessor, tmp_path):
        audio, sr = preprocessor.generate_synthetic_audio(duration_sec=1.0)
        filepath = str(tmp_path / "test.wav")
        preprocessor.save_wav(filepath, audio, sr)

        assert os.path.exists(filepath)
        loaded_audio, loaded_sr = preprocessor.load_audio(filepath)
        assert loaded_sr == sr
        assert abs(len(loaded_audio) - len(audio)) <= 1


class TestNormalization:
    def test_normalize_audio(self, preprocessor):
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        normalized = preprocessor._normalize_audio(audio)
        assert np.max(np.abs(normalized)) <= 1.0

    def test_silent_audio_normalization(self, preprocessor):
        audio = np.zeros(16000, dtype=np.float32)
        normalized = preprocessor._normalize_audio(audio)
        assert np.all(normalized == 0)
