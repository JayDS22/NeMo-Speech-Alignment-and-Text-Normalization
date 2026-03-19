"""
Shared utilities for the NeMo Speech Alignment pipeline.
"""

import json
import os
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_manifest(entries: List[Dict], filepath: str, format: str = "nemo"):
    """
    Save alignment results as a manifest file.

    Args:
        entries: List of manifest entries (dicts).
        filepath: Output file path.
        format: "nemo" (JSONL) or "kaldi" (text).
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if format == "nemo":
        with open(filepath, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
    elif format == "kaldi":
        with open(filepath, 'w') as f:
            for entry in entries:
                utt_id = os.path.splitext(os.path.basename(entry.get("audio_filepath", "")))[0]
                text = entry.get("text", "")
                f.write(f"{utt_id} {text}\n")
    else:
        raise ValueError(f"Unknown manifest format: {format}")


def load_manifest(filepath: str) -> List[Dict]:
    """Load a NeMo-style JSONL manifest."""
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS.ms string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def find_audio_files(
    directory: str,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """Recursively find audio files in a directory."""
    if extensions is None:
        extensions = [".wav", ".flac", ".mp3", ".ogg"]

    audio_files = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if any(f.lower().endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, f))

    return audio_files


def find_transcript_file(audio_path: str) -> Optional[str]:
    """Find the transcript file corresponding to an audio file."""
    base = os.path.splitext(audio_path)[0]
    for ext in [".txt", ".lab", ".trans"]:
        txt_path = base + ext
        if os.path.exists(txt_path):
            return txt_path
    return None
