"""
CTC-Based Forced Alignment Engine.

Provides word-level and segment-level timestamp generation
by aligning transcripts to audio using CTC-based acoustic models.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AlignedWord:
    """A single word with alignment information."""
    word: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    duration: float = 0.0

    def __post_init__(self):
        self.duration = self.end_time - self.start_time


@dataclass
class AlignedSegment:
    """A segment of aligned words."""
    text: str
    words: List[AlignedWord]
    start_time: float
    end_time: float
    duration: float = 0.0
    avg_confidence: float = 0.0

    def __post_init__(self):
        self.duration = self.end_time - self.start_time
        if self.words:
            self.avg_confidence = sum(w.confidence for w in self.words) / len(self.words)


@dataclass
class AlignmentResult:
    """Full alignment result for an audio-text pair."""
    audio_filepath: str
    transcript: str
    segments: List[AlignedSegment]
    words: List[AlignedWord]
    total_duration: float
    aligned_duration: float
    silence_ratio: float = 0.0
    avg_confidence: float = 0.0

    def __post_init__(self):
        if self.total_duration > 0:
            self.silence_ratio = 1.0 - (self.aligned_duration / self.total_duration)
        if self.words:
            self.avg_confidence = sum(w.confidence for w in self.words) / len(self.words)

    def to_manifest_entry(self) -> Dict:
        """Convert to NeMo-style manifest entry."""
        return {
            "audio_filepath": self.audio_filepath,
            "text": self.transcript,
            "duration": self.total_duration,
            "words": [
                {
                    "word": w.word,
                    "start": round(w.start_time, 4),
                    "end": round(w.end_time, 4),
                    "confidence": round(w.confidence, 4),
                }
                for w in self.words
            ],
        }


class ForcedAligner:
    """
    CTC-based Forced Alignment for audio-text alignment.

    This implementation provides:
        - Word-level timestamp generation
        - Confidence scoring per word
        - Segment-level grouping
        - Support for pre-computed CTC log-probabilities

    For production use with NVIDIA NeMo:
        from nemo.collections.asr.models import EncDecCTCModel
        model = EncDecCTCModel.from_pretrained("stt_en_conformer_ctc_large")

    This class provides a compatible interface that works both with
    NeMo models and with a built-in simulation for demo/testing.
    """

    # Standard English character vocabulary for CTC
    VOCAB = list(" abcdefghijklmnopqrstuvwxyz'") + ["<blank>"]

    def __init__(
        self,
        model_name: str = "stt_en_conformer_ctc_large",
        use_gpu: bool = False,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._model = None
        self._char_to_idx = {c: i for i, c in enumerate(self.VOCAB)}
        self._blank_idx = len(self.VOCAB) - 1

    def load_model(self):
        """
        Load the acoustic model.

        Attempts to load NeMo model; falls back to simulation mode.
        """
        try:
            from nemo.collections.asr.models import EncDecCTCModel
            self._model = EncDecCTCModel.from_pretrained(self.model_name)
            if not self.use_gpu:
                self._model = self._model.cpu()
            self._model.eval()
            print(f"Loaded NeMo model: {self.model_name}")
        except (ImportError, Exception) as e:
            print(f"NeMo not available ({e}). Using simulation mode for demo.")
            self._model = None

    def align(
        self,
        audio: np.ndarray,
        transcript: str,
        sample_rate: int = 16000,
        audio_filepath: str = "",
    ) -> AlignmentResult:
        """
        Align transcript to audio, producing word-level timestamps.

        Args:
            audio: Audio signal array.
            transcript: Text transcript to align.
            sample_rate: Audio sample rate.
            audio_filepath: Path for manifest generation.

        Returns:
            AlignmentResult with word and segment-level alignment.
        """
        total_duration = len(audio) / sample_rate
        words = transcript.strip().split()

        if not words:
            return AlignmentResult(
                audio_filepath=audio_filepath,
                transcript=transcript,
                segments=[],
                words=[],
                total_duration=total_duration,
                aligned_duration=0.0,
            )

        if self._model is not None:
            aligned_words = self._align_with_model(audio, words, sample_rate)
        else:
            aligned_words = self._align_simulation(audio, words, sample_rate)

        # Group into segments
        segments = self._group_into_segments(aligned_words, max_gap_sec=0.5)

        aligned_duration = sum(w.duration for w in aligned_words)

        return AlignmentResult(
            audio_filepath=audio_filepath,
            transcript=transcript,
            segments=segments,
            words=aligned_words,
            total_duration=total_duration,
            aligned_duration=aligned_duration,
        )

    def _align_with_model(
        self,
        audio: np.ndarray,
        words: List[str],
        sample_rate: int,
    ) -> List[AlignedWord]:
        """Align using actual NeMo CTC model via Viterbi forced alignment."""
        try:
            import torch

            # Get CTC log-probabilities
            audio_tensor = torch.tensor(audio).unsqueeze(0).float()
            audio_len = torch.tensor([len(audio)])

            with torch.no_grad():
                log_probs, encoded_len, _ = self._model.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len,
                )

            log_probs = log_probs.squeeze(0).cpu().numpy()
            return self._ctc_forced_align(log_probs, words, len(audio) / sample_rate)

        except Exception as e:
            print(f"Model alignment failed: {e}. Falling back to simulation.")
            return self._align_simulation(audio, words, sample_rate)

    def _ctc_forced_align(
        self,
        log_probs: np.ndarray,
        words: List[str],
        total_duration: float,
    ) -> List[AlignedWord]:
        """
        Perform CTC forced alignment using dynamic programming.

        Args:
            log_probs: CTC log-probabilities, shape (T, V).
            words: List of words to align.
            total_duration: Total audio duration in seconds.

        Returns:
            List of AlignedWord with timestamps.
        """
        T = log_probs.shape[0]
        transcript = " ".join(words).lower()

        # Build character-level target sequence with blanks
        target = []
        target.append(self._blank_idx)
        for char in transcript:
            idx = self._char_to_idx.get(char, self._char_to_idx.get(' ', 0))
            target.append(idx)
            target.append(self._blank_idx)

        S = len(target)

        # Viterbi forced alignment
        dp = np.full((T, S), -np.inf)
        backptr = np.zeros((T, S), dtype=np.int32)

        # Initialize
        dp[0, 0] = log_probs[0, target[0]]
        if S > 1:
            dp[0, 1] = log_probs[0, target[1]]

        # Forward pass
        for t in range(1, T):
            for s in range(S):
                # Stay in same state
                score_stay = dp[t - 1, s]
                best_score = score_stay
                best_ptr = s

                # Move from previous state
                if s > 0:
                    score_prev = dp[t - 1, s - 1]
                    if score_prev > best_score:
                        best_score = score_prev
                        best_ptr = s - 1

                # Skip blank (if current is not blank and prev-prev is different)
                if s > 1 and target[s] != self._blank_idx and target[s] != target[s - 2]:
                    score_skip = dp[t - 1, s - 2]
                    if score_skip > best_score:
                        best_score = score_skip
                        best_ptr = s - 2

                dp[t, s] = best_score + log_probs[t, target[s]]
                backptr[t, s] = best_ptr

        # Backtrace
        path = np.zeros(T, dtype=np.int32)
        path[T - 1] = S - 1 if dp[T - 1, S - 1] > dp[T - 1, S - 2] else S - 2

        for t in range(T - 2, -1, -1):
            path[t] = backptr[t + 1, path[t + 1]]

        # Extract word boundaries from path
        time_per_frame = total_duration / T
        aligned_words = []
        char_idx = 0
        word_idx = 0

        for word in words:
            word_chars = word.lower()
            # Find start frame for this word
            start_frame = None
            end_frame = None

            target_start = 1 + char_idx * 2  # Skip blanks
            target_end = target_start + len(word_chars) * 2

            for t in range(T):
                if target_start <= path[t] < target_end:
                    if start_frame is None:
                        start_frame = t
                    end_frame = t

            if start_frame is None:
                # Fallback: distribute evenly
                total_words = len(words)
                start_frame = int(T * word_idx / total_words)
                end_frame = int(T * (word_idx + 1) / total_words)

            start_time = start_frame * time_per_frame
            end_time = (end_frame + 1) * time_per_frame

            # Confidence from average log-prob
            frame_scores = []
            for t in range(start_frame, min(end_frame + 1, T)):
                frame_scores.append(log_probs[t, target[path[t]]])
            confidence = float(np.exp(np.mean(frame_scores))) if frame_scores else 0.5

            aligned_words.append(AlignedWord(
                word=word,
                start_time=round(start_time, 4),
                end_time=round(end_time, 4),
                confidence=min(max(confidence, 0.0), 1.0),
            ))

            char_idx += len(word_chars) + 1  # +1 for space
            word_idx += 1

        return aligned_words

    def _align_simulation(
        self,
        audio: np.ndarray,
        words: List[str],
        sample_rate: int,
    ) -> List[AlignedWord]:
        """
        Simulate forced alignment using energy-based heuristics.

        This provides realistic-looking alignment for demo purposes
        when NeMo models are not available.
        """
        total_duration = len(audio) / sample_rate
        n_words = len(words)

        if n_words == 0:
            return []

        # Compute frame-level energy for timing
        frame_size = int(sample_rate * 0.025)  # 25ms frames
        hop_size = int(sample_rate * 0.010)     # 10ms hop
        n_frames = max(1, (len(audio) - frame_size) // hop_size + 1)

        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop_size
            end = min(start + frame_size, len(audio))
            frame = audio[start:end]
            energies[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0

        # Distribute words proportionally to their length (character count)
        total_chars = sum(len(w) for w in words) + (n_words - 1)  # spaces
        time_per_char = total_duration * 0.85 / max(total_chars, 1)  # 85% speaking time
        silence_padding = total_duration * 0.15 / max(n_words + 1, 1)

        aligned_words = []
        current_time = silence_padding / 2

        for i, word in enumerate(words):
            word_duration = len(word) * time_per_char
            # Add some natural variation
            variation = np.random.uniform(0.85, 1.15)
            word_duration *= variation

            start_time = current_time
            end_time = current_time + word_duration

            # Compute confidence based on energy in this region
            start_frame = int(start_time / (hop_size / sample_rate))
            end_frame = int(end_time / (hop_size / sample_rate))
            start_frame = max(0, min(start_frame, n_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, n_frames))

            region_energy = energies[start_frame:end_frame]
            if len(region_energy) > 0 and np.max(energies) > 0:
                confidence = float(np.mean(region_energy) / (np.max(energies) + 1e-10))
                confidence = min(max(confidence * 1.5, 0.3), 0.98)
            else:
                confidence = 0.7 + np.random.uniform(-0.1, 0.15)

            aligned_words.append(AlignedWord(
                word=word,
                start_time=round(start_time, 4),
                end_time=round(end_time, 4),
                confidence=round(confidence, 4),
            ))

            current_time = end_time + silence_padding * np.random.uniform(0.5, 1.5)

        return aligned_words

    def _group_into_segments(
        self,
        words: List[AlignedWord],
        max_gap_sec: float = 0.5,
    ) -> List[AlignedSegment]:
        """Group aligned words into segments based on silence gaps."""
        if not words:
            return []

        segments = []
        current_words = [words[0]]

        for i in range(1, len(words)):
            gap = words[i].start_time - words[i - 1].end_time
            if gap > max_gap_sec:
                # Start new segment
                seg_text = " ".join(w.word for w in current_words)
                segments.append(AlignedSegment(
                    text=seg_text,
                    words=current_words,
                    start_time=current_words[0].start_time,
                    end_time=current_words[-1].end_time,
                ))
                current_words = [words[i]]
            else:
                current_words.append(words[i])

        # Final segment
        if current_words:
            seg_text = " ".join(w.word for w in current_words)
            segments.append(AlignedSegment(
                text=seg_text,
                words=current_words,
                start_time=current_words[0].start_time,
                end_time=current_words[-1].end_time,
            ))

        return segments

    def align_batch(
        self,
        audio_list: List[np.ndarray],
        transcripts: List[str],
        sample_rate: int = 16000,
        filepaths: Optional[List[str]] = None,
    ) -> List[AlignmentResult]:
        """Align a batch of audio-transcript pairs."""
        if filepaths is None:
            filepaths = [""] * len(audio_list)

        results = []
        for audio, transcript, filepath in zip(audio_list, transcripts, filepaths):
            result = self.align(audio, transcript, sample_rate, filepath)
            results.append(result)

        return results
