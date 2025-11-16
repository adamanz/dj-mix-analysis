"""
Play a DJ mix locally and show the precomputed transitions / track IDs in real time.

Prerequisite: run analyze_mix.py to generate mix.tracklist.json

Usage:
    python demo_play_mix.py mixes/cercle_set_01.wav
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import sounddevice as sd
import torchaudio

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import librosa
except ImportError:
    librosa = None


def load_tracklist(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "segments" not in data:
        raise ValueError(f"Tracklist at {path} missing 'segments' key")
    return data


def load_audio(path: Path) -> tuple[np.ndarray, int, float]:
    # Try torchaudio first, fallback to soundfile/librosa if it fails
    try:
        waveform, sr = torchaudio.load(path)
    except (ImportError, RuntimeError, Exception):
        # Fallback to soundfile or librosa
        if sf is not None:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim == 1:
                waveform = torch.tensor(data).unsqueeze(0)
            else:
                waveform = torch.tensor(data.mean(axis=1)).unsqueeze(0)
        elif librosa is not None:
            data, sr = librosa.load(str(path), sr=None, mono=False)
            if isinstance(data, np.ndarray) and data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.mean(axis=0)).unsqueeze(0)
        else:
            raise RuntimeError("Need torchaudio, soundfile, or librosa to load audio")
    
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    data = waveform.squeeze(0).numpy()
    duration = data.shape[0] / sr
    return data, sr, duration


def seconds_to_clock(value: float) -> str:
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_summary(tracklist: dict) -> None:
    print("=" * 70)
    print("DJ Mix Playback Demo")
    print("=" * 70)
    print(f"Audio file: {tracklist.get('audio_path')}")
    print(f"Generated:  {tracklist.get('generated_at')}")
    print(f"Duration:   {seconds_to_clock(tracklist.get('duration_seconds', 0.0))}")
    print(f"Segments:   {len(tracklist.get('segments', []))}")
    print("=" * 70)
    for seg in tracklist.get("segments", []):
        print(
            f"{seg['index']:02d} | "
            f"{seconds_to_clock(seg['start_s'])} → {seconds_to_clock(seg['end_s'])} | "
            f"{seg['track_name']}"
        )
    print("=" * 70)


def demo_playback(audio_path: Path, tracklist_path: Path) -> None:
    tracklist = load_tracklist(tracklist_path)
    print_summary(tracklist)

    audio_data, sr, duration = load_audio(audio_path)

    print("Starting playback... (Ctrl+C to stop)")
    sd.play(audio_data, sr)
    start_time = time.time()
    current_index = None
    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            segment = None
            for seg in tracklist["segments"]:
                if seg["start_s"] <= elapsed < seg["end_s"]:
                    segment = seg
                    break
            if segment and segment["index"] != current_index:
                current_index = segment["index"]
                print(
                    f">>> Transition @ {seconds_to_clock(segment['start_s'])} → "
                    f"Track {segment['index']:02d}: {segment['track_name']}"
                )
            elif segment:
                print(
                    f"Time {seconds_to_clock(elapsed)} | "
                    f"Track {segment['index']:02d}: {segment['track_name']}",
                    end="\r",
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user.")
    finally:
        sd.stop()
        print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a mix with tracklist overlay.")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--tracklist",
        help="Path to .tracklist.json (defaults to audio_path + .tracklist.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    tracklist_path = Path(args.tracklist).resolve() if args.tracklist else Path(
        f"{audio_path}.tracklist.json"
    ).resolve()
    if not tracklist_path.exists():
        raise FileNotFoundError(
            f"Tracklist JSON not found: {tracklist_path}\n"
            "Run analyze_mix.py first to generate it."
        )

    demo_playback(audio_path, tracklist_path)


if __name__ == "__main__":
    main()


