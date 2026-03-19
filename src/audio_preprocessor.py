"""
Audio Preprocessor for speech data preparation.

Handles loading, resampling, VAD, segmentation, and feature extraction
for ASR dataset preparation pipelines.
"""

import os
import math
import struct
import wave
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


@dataclass
class AudioSegment:
    """A segment of audio with metadata."""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    duration: float = 0.0
    rms_energy: float = 0.0
    is_speech: bool = True

    def __post_init__(self):
        self.duration = self.end_time - self.start_time
        if len(self.audio) > 0:
            self.rms_energy = float(np.sqrt(np.mean(self.audio ** 2)))


@dataclass
class AudioInfo:
    """Metadata about an audio file."""
    filepath: str
    sample_rate: int
    duration: float
    num_samples: int
    num_channels: int
    bit_depth: int = 16
    rms_energy: float = 0.0
    peak_amplitude: float = 0.0


class AudioPreprocessor:
    """
    Audio preprocessing pipeline for ASR data preparation.

    Features:
        - Audio loading with format detection
        - Resampling to target sample rate
        - Mono conversion
        - Voice Activity Detection (energy-based)
        - Segmentation by silence
        - Mel-spectrogram extraction
        - Audio quality metrics
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        mono: bool = True,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 0.5,
    ):
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return (audio_array, sample_rate).

        Supports WAV natively; uses librosa/soundfile for other formats.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        # Try native WAV reading first
        if ext == ".wav":
            try:
                return self._load_wav_native(filepath)
            except Exception:
                pass

        # Fall back to librosa
        if HAS_LIBROSA:
            audio, sr = librosa.load(filepath, sr=None, mono=self.mono)
            return audio, sr

        # Fall back to soundfile
        if HAS_SOUNDFILE:
            audio, sr = sf.read(filepath, dtype='float32')
            if self.mono and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr

        raise ImportError(
            "No audio library available. Install librosa or soundfile: "
            "pip install librosa soundfile"
        )

    def _load_wav_native(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load WAV file using Python's wave module."""
        with wave.open(filepath, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            raw_data = wf.readframes(n_frames)

            if sampwidth == 2:
                fmt = f"<{n_frames * n_channels}h"
                data = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
            elif sampwidth == 4:
                fmt = f"<{n_frames * n_channels}i"
                data = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported bit depth: {sampwidth * 8}")

            if n_channels > 1 and self.mono:
                data = data.reshape(-1, n_channels).mean(axis=1)

        return data, sr

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == self.target_sample_rate:
            return audio

        if HAS_LIBROSA:
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sample_rate)

        # Simple linear interpolation fallback
        ratio = self.target_sample_rate / orig_sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def get_audio_info(self, filepath: str) -> AudioInfo:
        """Get metadata about an audio file."""
        audio, sr = self.load_audio(filepath)
        return AudioInfo(
            filepath=filepath,
            sample_rate=sr,
            duration=len(audio) / sr,
            num_samples=len(audio),
            num_channels=1 if audio.ndim == 1 else audio.shape[1],
            rms_energy=float(np.sqrt(np.mean(audio ** 2))),
            peak_amplitude=float(np.max(np.abs(audio))),
        )

    def preprocess(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Full preprocessing: load, resample, normalize.

        Returns:
            Tuple of (preprocessed_audio, sample_rate).
        """
        audio, sr = self.load_audio(filepath)
        audio = self.resample(audio, sr)
        audio = self._normalize_audio(audio)
        return audio, self.target_sample_rate

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95  # Leave some headroom
        return audio

    # ── Voice Activity Detection ──────────────────────────────────

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        min_speech_duration_ms: int = 250,
    ) -> List[Tuple[float, float]]:
        """
        Energy-based Voice Activity Detection.

        Args:
            audio: Audio signal array.
            sample_rate: Sample rate in Hz.
            frame_duration_ms: Frame size in milliseconds.
            energy_threshold: RMS energy threshold for speech detection.
            min_speech_duration_ms: Minimum speech segment duration.

        Returns:
            List of (start_time, end_time) tuples for speech regions.
        """
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        n_frames = len(audio) // frame_size

        # Compute frame-level energy
        energies = []
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame ** 2))
            energies.append(rms)

        energies = np.array(energies)

        # Adaptive threshold: use the gap between quiet and loud frames
        if len(energies) > 0:
            p10 = np.percentile(energies, 10)
            p90 = np.percentile(energies, 90)
            dynamic_range = p90 - p10
            if dynamic_range > 0.01:
                # Good dynamic range: set threshold between quiet and loud
                threshold = max(energy_threshold, p10 + dynamic_range * 0.2)
            else:
                # Uniform energy (continuous speech): use fixed threshold
                threshold = energy_threshold
        else:
            threshold = energy_threshold

        # Find speech regions
        is_speech = energies > threshold
        regions = []
        start = None

        for i, speech in enumerate(is_speech):
            if speech and start is None:
                start = i
            elif not speech and start is not None:
                end = i
                start_time = start * frame_duration_ms / 1000
                end_time = end * frame_duration_ms / 1000
                duration_ms = (end - start) * frame_duration_ms
                if duration_ms >= min_speech_duration_ms:
                    regions.append((start_time, end_time))
                start = None

        # Handle final region
        if start is not None:
            end_time = n_frames * frame_duration_ms / 1000
            duration_ms = (n_frames - start) * frame_duration_ms
            if duration_ms >= min_speech_duration_ms:
                regions.append((start * frame_duration_ms / 1000, end_time))

        return regions

    # ── Segmentation ──────────────────────────────────────────────

    def segment_by_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        min_silence_duration_ms: int = 500,
        silence_threshold: float = 0.01,
    ) -> List[AudioSegment]:
        """
        Segment audio by detecting silence gaps.

        Returns:
            List of AudioSegment objects.
        """
        speech_regions = self.detect_voice_activity(
            audio, sample_rate,
            energy_threshold=silence_threshold,
            min_speech_duration_ms=200,
        )

        segments = []
        for start_time, end_time in speech_regions:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            duration = end_time - start_time
            if self.min_duration_sec <= duration <= self.max_duration_sec:
                segments.append(AudioSegment(
                    audio=segment_audio,
                    sample_rate=sample_rate,
                    start_time=start_time,
                    end_time=end_time,
                ))

        return segments

    # ── Feature Extraction ────────────────────────────────────────

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        sample_rate: int,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
    ) -> np.ndarray:
        """
        Extract log mel-spectrogram features.

        Returns:
            Mel-spectrogram array of shape (n_mels, time_steps).
        """
        if HAS_LIBROSA:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sample_rate,
                n_mels=n_mels, n_fft=n_fft,
                hop_length=hop_length,
                fmin=fmin, fmax=fmax,
            )
            return librosa.power_to_db(mel_spec, ref=np.max)

        # Fallback: simple STFT-based approximation
        return self._simple_spectrogram(audio, n_fft, hop_length)

    def _simple_spectrogram(
        self, audio: np.ndarray, n_fft: int, hop_length: int
    ) -> np.ndarray:
        """Simple spectrogram without librosa."""
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        spec = np.zeros((n_fft // 2 + 1, max(n_frames, 1)))

        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed = frame * window
            fft_result = np.fft.rfft(windowed)
            spec[:, i] = np.abs(fft_result) ** 2

        # Convert to dB
        spec = np.maximum(spec, 1e-10)
        return 10 * np.log10(spec)

    # ── Utilities ─────────────────────────────────────────────────

    def generate_synthetic_audio(
        self,
        duration_sec: float = 3.0,
        frequency: float = 440.0,
        noise_level: float = 0.01,
    ) -> Tuple[np.ndarray, int]:
        """Generate synthetic audio for testing/demo purposes."""
        sr = self.target_sample_rate
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)

        # Mix of frequencies to simulate speech-like signal
        audio = (
            0.5 * np.sin(2 * np.pi * frequency * t) +
            0.3 * np.sin(2 * np.pi * (frequency * 1.5) * t) +
            0.2 * np.sin(2 * np.pi * (frequency * 2) * t)
        )

        # Add envelope for natural fading
        envelope = np.ones_like(t)
        fade_len = int(0.1 * sr)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        audio *= envelope

        # Add noise
        if noise_level > 0:
            audio += noise_level * np.random.randn(len(audio))

        audio = audio.astype(np.float32)
        return audio, sr

    def save_wav(self, filepath: str, audio: np.ndarray, sample_rate: int):
        """Save audio to WAV file."""
        if HAS_SOUNDFILE:
            sf.write(filepath, audio, sample_rate)
        else:
            # Native WAV writing
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
