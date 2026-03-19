"""
End-to-end pipeline orchestrator for NeMo Speech Alignment & Text Normalization.

Usage:
    python -m src.pipeline --audio data/sample_audio/ --text data/sample_transcripts/ --output output/
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.text_normalizer import TextNormalizer, NormalizationResult
from src.audio_preprocessor import AudioPreprocessor, AudioSegment
from src.forced_aligner import ForcedAligner, AlignmentResult
from src.quality_validator import QualityValidator, QualityMetrics, QualityReport
from src.utils import load_config, save_manifest, find_audio_files, find_transcript_file


class SpeechAlignmentPipeline:
    """
    Production pipeline for speech data normalization and alignment.

    Orchestrates:
        1. Audio preprocessing (load, resample, VAD)
        2. Text normalization (numbers, abbreviations, dates, etc.)
        3. Forced alignment (word-level timestamps)
        4. Quality validation (WER, CER, confidence)
        5. Manifest generation (NeMo/Kaldi format)
    """

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        audio_cfg = config.get("audio", {})
        tn_cfg = config.get("text_normalization", {})
        align_cfg = config.get("forced_alignment", {})
        qa_cfg = config.get("quality_validation", {})

        self.audio_preprocessor = AudioPreprocessor(
            target_sample_rate=audio_cfg.get("sample_rate", 16000),
            mono=audio_cfg.get("mono", True),
            max_duration_sec=audio_cfg.get("max_duration_sec", 30.0),
            min_duration_sec=audio_cfg.get("min_duration_sec", 0.5),
        )

        self.text_normalizer = TextNormalizer(
            expand_numbers=tn_cfg.get("expand_numbers", True),
            expand_abbreviations=tn_cfg.get("expand_abbreviations", True),
            expand_dates=tn_cfg.get("expand_dates", True),
            expand_currency=tn_cfg.get("expand_currency", True),
            expand_time=tn_cfg.get("expand_time", True),
            expand_measures=tn_cfg.get("expand_measures", True),
            lowercase=tn_cfg.get("lowercase", True),
            language=tn_cfg.get("language", "en"),
        )

        self.aligner = ForcedAligner(
            model_name=align_cfg.get("model_name", "stt_en_conformer_ctc_large"),
            use_gpu=align_cfg.get("use_gpu", False),
            batch_size=align_cfg.get("batch_size", 8),
        )

        self.validator = QualityValidator(
            wer_threshold=qa_cfg.get("wer_threshold", 0.30),
            cer_threshold=qa_cfg.get("cer_threshold", 0.15),
            min_confidence=qa_cfg.get("min_confidence", 0.60),
            min_words_per_segment=qa_cfg.get("min_words_per_segment", 2),
            max_silence_ratio=qa_cfg.get("max_silence_ratio", 0.80),
        )

        self.output_config = config.get("output", {})

    def process_single(
        self,
        audio_path: str,
        transcript: str,
    ) -> Tuple[AlignmentResult, NormalizationResult, QualityMetrics]:
        """
        Process a single audio-transcript pair.

        Returns:
            Tuple of (AlignmentResult, NormalizationResult, QualityMetrics).
        """
        # 1. Preprocess audio
        audio, sr = self.audio_preprocessor.preprocess(audio_path)

        # 2. Normalize text
        norm_result = self.text_normalizer.normalize(transcript)

        # 3. Forced alignment
        alignment = self.aligner.align(
            audio=audio,
            transcript=norm_result.normalized,
            sample_rate=sr,
            audio_filepath=audio_path,
        )

        # 4. Quality validation
        word_confidences = [w.confidence for w in alignment.words]
        metrics = self.validator.validate_alignment(
            reference=transcript.lower(),
            hypothesis=norm_result.normalized,
            word_confidences=word_confidences,
            duration=alignment.total_duration,
            aligned_duration=alignment.aligned_duration,
            silence_ratio=alignment.silence_ratio,
        )

        return alignment, norm_result, metrics

    def process_from_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcript: str,
        audio_filepath: str = "synthetic",
    ) -> Tuple[AlignmentResult, NormalizationResult, QualityMetrics]:
        """
        Process from a pre-loaded audio array.
        """
        # Resample if needed
        if sample_rate != self.audio_preprocessor.target_sample_rate:
            audio = self.audio_preprocessor.resample(audio, sample_rate)
            sample_rate = self.audio_preprocessor.target_sample_rate

        # Normalize text
        norm_result = self.text_normalizer.normalize(transcript)

        # Align
        alignment = self.aligner.align(
            audio=audio,
            transcript=norm_result.normalized,
            sample_rate=sample_rate,
            audio_filepath=audio_filepath,
        )

        # Validate
        word_confidences = [w.confidence for w in alignment.words]
        metrics = self.validator.validate_alignment(
            reference=transcript.lower(),
            hypothesis=norm_result.normalized,
            word_confidences=word_confidences,
            duration=alignment.total_duration,
            aligned_duration=alignment.aligned_duration,
            silence_ratio=alignment.silence_ratio,
        )

        return alignment, norm_result, metrics

    def process_batch(
        self,
        audio_dir: str,
        output_dir: str,
        transcript_dir: Optional[str] = None,
    ) -> QualityReport:
        """
        Process a batch of audio files with their transcripts.

        Args:
            audio_dir: Directory containing audio files.
            output_dir: Directory for output manifests and reports.
            transcript_dir: Optional separate directory for transcripts.

        Returns:
            QualityReport for the batch.
        """
        os.makedirs(output_dir, exist_ok=True)

        audio_files = find_audio_files(audio_dir)
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return QualityReport()

        print(f"Found {len(audio_files)} audio files")

        manifest_entries = []
        metrics_list = []
        filepaths = []

        for i, audio_path in enumerate(audio_files):
            print(f"  [{i+1}/{len(audio_files)}] Processing {os.path.basename(audio_path)}...")

            # Find transcript
            txt_path = find_transcript_file(audio_path)
            if transcript_dir:
                base = os.path.splitext(os.path.basename(audio_path))[0]
                alt_path = os.path.join(transcript_dir, base + ".txt")
                if os.path.exists(alt_path):
                    txt_path = alt_path

            if txt_path is None:
                print(f"    WARNING: No transcript found, skipping.")
                continue

            with open(txt_path, 'r') as f:
                transcript = f.read().strip()

            try:
                alignment, norm_result, metrics = self.process_single(audio_path, transcript)
                manifest_entries.append(alignment.to_manifest_entry())
                metrics_list.append(metrics)
                filepaths.append(audio_path)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue

        # Save manifest
        manifest_fmt = self.output_config.get("manifest_format", "nemo")
        manifest_path = os.path.join(output_dir, f"manifest.{'jsonl' if manifest_fmt == 'nemo' else 'txt'}")
        save_manifest(manifest_entries, manifest_path, format=manifest_fmt)
        print(f"\nManifest saved: {manifest_path}")

        # Generate and save report
        report = self.validator.generate_report(metrics_list, filepaths)
        report_path = os.path.join(output_dir, "quality_report.json")
        self.validator.save_report(report, report_path)
        print(f"Quality report saved: {report_path}")

        # Print summary
        print(f"\n{'='*50}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Total files:      {report.total_files}")
        print(f"Valid files:       {report.valid_files}")
        print(f"Invalid files:     {report.invalid_files}")
        print(f"Acceptance rate:   {report.valid_files/max(report.total_files,1)*100:.1f}%")
        print(f"Average WER:       {report.avg_wer:.4f}")
        print(f"Average CER:       {report.avg_cer:.4f}")
        print(f"Average Confidence:{report.avg_confidence:.4f}")
        print(f"Total duration:    {report.total_duration_hrs:.2f} hrs")
        print(f"Valid duration:    {report.valid_duration_hrs:.2f} hrs")
        if report.flag_distribution:
            print(f"\nFlag distribution:")
            for flag, count in sorted(report.flag_distribution.items()):
                print(f"  {flag}: {count}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="NeMo Speech Alignment & Text Normalization Pipeline"
    )
    parser.add_argument("--audio", required=True, help="Audio directory or file")
    parser.add_argument("--text", help="Transcript directory (optional)")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Config file")

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded config: {args.config}")

    pipeline = SpeechAlignmentPipeline(config)

    start_time = time.time()
    report = pipeline.process_batch(args.audio, args.output, args.text)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
