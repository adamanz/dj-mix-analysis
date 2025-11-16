"""
Analyze a downloaded DJ set:
- detect likely transition boundaries
- split into segments
- ask LiquidAI LFM2-Audio-1.5B to identify each track
- save a JSON tracklist for demo playback

Usage:
    python analyze_mix.py path/to/mix.wav --out tracklist.json

Requirements:
    pip install torchaudio librosa soundfile liquid-audio
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchaudio

try:
    import librosa
except ImportError as exc:  # pragma: no cover - surface friendly error
    raise SystemExit(
        "librosa is required for transition detection.\n"
        "Install with: pip install librosa"
    ) from exc

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from liquid_audio import (
        ChatState,
        LFM2AudioModel,
        LFM2AudioProcessor,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "liquid-audio is required for track identification.\n"
        "Install with: pip install liquid-audio"
    ) from exc


@dataclass
class SegmentResult:
    index: int
    start_s: float
    end_s: float
    duration_s: float
    track_name: str
    notes: str | None = None


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    # Try torchaudio first, fallback to soundfile/librosa if it fails
    try:
        waveform, sr = torchaudio.load(path)
    except (ImportError, RuntimeError, Exception) as e:
        # Fallback to soundfile or librosa if torchaudio fails (e.g., missing torchcodec)
        if sf is not None:
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                # stereo → mono
                waveform = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        else:
            # Use librosa as last resort
            data, sr = librosa.load(str(path), sr=None, mono=False)
            if isinstance(data, np.ndarray) and data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.mean(axis=0)).unsqueeze(0)
    
    if waveform.dim() != 2:
        raise RuntimeError(f"Unexpected waveform shape {waveform.shape}")
    if waveform.size(0) > 1:  # stereo → mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    # normalize to avoid clipping
    peak = waveform.abs().max().item()
    if peak > 0:
        waveform = waveform / peak
    return waveform, sr


def detect_transition_candidates(
    waveform_np: np.ndarray,
    sr: int,
    hop_length: int,
    peak_delta: float,
    max_candidates: int,
) -> List[float]:
    """
    Use a simple onset-strength / spectral-flux heuristic to find transition peaks.
    Returns a sorted list of times in seconds.
    """
    onset_env = librosa.onset.onset_strength(
        y=waveform_np,
        sr=sr,
        hop_length=hop_length,
        aggregate=np.mean,
    )
    if onset_env.size == 0:
        return []
    onset_norm = (onset_env - onset_env.mean()) / (onset_env.std() + 1e-6)
    peaks = librosa.util.peak_pick(
        onset_norm,
        pre_max=8,
        post_max=8,
        pre_avg=16,
        post_avg=16,
        delta=peak_delta,
        wait=16,
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    times = times.tolist()
    times = sorted(t for t in times if t > 0)
    if max_candidates and len(times) > max_candidates:
        times = times[:max_candidates]
    return times


def merge_transitions(
    candidates: Sequence[float],
    duration_s: float,
    min_gap: float,
    edge_margin: float,
) -> List[float]:
    filtered: List[float] = []
    for t in candidates:
        if t < edge_margin or t > duration_s - edge_margin:
            continue
        if filtered and (t - filtered[-1]) < min_gap:
            continue
        filtered.append(t)
    return filtered


def build_segments(
    duration_s: float,
    transitions: Sequence[float],
    min_segment_dur: float,
) -> List[tuple[float, float]]:
    cuts = [0.0] + list(transitions) + [duration_s]
    raw_segments = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    merged: List[list[float]] = []
    for start, end in raw_segments:
        if not merged:
            merged.append([start, end])
            continue
        if (end - start) < min_segment_dur:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(round(seg[0], 3), round(seg[1], 3)) for seg in merged if seg[1] > seg[0]]


def slice_preview(
    waveform: torch.Tensor,
    sr: int,
    window: tuple[float, float],
    preview_duration: float,
    lead_in: float,
) -> torch.Tensor | None:
    start, end = window
    start = max(start + lead_in, start)
    if (end - start) < preview_duration:
        start = max(window[0], end - preview_duration)
    stop = min(end, start + preview_duration)
    if stop - start <= 0.5:
        return None
    start_idx = int(start * sr)
    end_idx = int(stop * sr)
    return waveform[:, start_idx:end_idx].contiguous()


def load_liquid_model(device: str) -> tuple[LFM2AudioModel, LFM2AudioProcessor]:
    repo = "LiquidAI/LFM2-Audio-1.5B"
    # Ensure device is valid - if CUDA not available, force CPU
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Force CPU if CUDA is not properly available
    import os
    original_cuda_available = None
    if device == "cpu":
        # Temporarily patch torch.cuda.is_available to return False
        # This prevents liquid_audio from trying to use CUDA internally
        if hasattr(torch.cuda, 'is_available'):
            original_cuda_available = torch.cuda.is_available
            # Monkey patch to force CPU
            torch.cuda.is_available = lambda: False
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    
    try:
        # Try with device parameter
        processor = LFM2AudioProcessor.from_pretrained(repo, device=device).eval()
    except (TypeError, ValueError, AssertionError, Exception):
        # Some versions don't accept device in from_pretrained or have issues
        processor = LFM2AudioProcessor.from_pretrained(repo).eval()
        try:
            processor = processor.to(device)
        except (AssertionError, RuntimeError):
            # If device move fails, keep on CPU
            device = "cpu"
            if hasattr(processor, 'to'):
                try:
                    processor = processor.to(device)
                except:
                    pass
    
    try:
        # Try with device parameter
        model = LFM2AudioModel.from_pretrained(repo, device=device).eval()
    except (TypeError, ValueError, AssertionError, Exception):
        # Fallback: load then move
        model = LFM2AudioModel.from_pretrained(repo).eval()
        try:
            if device != "cpu":
                model = model.to(device)
        except (AssertionError, RuntimeError):
            # If CUDA move fails, keep on CPU
            device = "cpu"
            model = model.to(device)
    finally:
        # Restore original cuda.is_available if we patched it
        if original_cuda_available is not None:
            torch.cuda.is_available = original_cuda_available
    
    return model, processor


def identify_track(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    segment: torch.Tensor,
    sr: int,
    max_new_tokens: int,
) -> str:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(
        "You are a DJ track ID assistant. "
        "Given a short excerpt from a DJ set, guess the main song name. "
        "Answer in the format 'Artist - Title (Remix)' or say 'Unknown track'."
    )
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(segment.cpu(), sr)
    chat.end_turn()

    chat.new_turn("assistant")
    text_tokens: List[int] = []
    try:
        for token in model.generate_sequential(
            **chat,
            max_new_tokens=max_new_tokens,
        ):
            if token.numel() == 1:
                # Convert scalar tensor to int
                text_tokens.append(int(token.cpu().item()))
            if len(text_tokens) >= max_new_tokens:
                break
    except Exception as exc:  # pragma: no cover
        return f"Unknown track (error: {exc})"

    if not text_tokens:
        return "Unknown track"

    # Decode list of token IDs
    decoded = processor.text.decode(text_tokens)
    return decoded.strip()


def seconds_to_clock(value: float) -> str:
    minutes, seconds = divmod(int(value), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def analyze_mix(args: argparse.Namespace) -> dict:
    audio_path = Path(args.audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print(f"[1/5] Loading audio: {audio_path}")
    waveform, sr = load_audio(audio_path, args.target_sr)
    duration_s = waveform.size(1) / sr
    print(f"       Duration: {seconds_to_clock(duration_s)} | Sample rate: {sr} Hz")

    print("[2/5] Detecting transition candidates...")
    candidates = detect_transition_candidates(
        waveform_np=waveform.squeeze(0).cpu().numpy(),
        sr=sr,
        hop_length=args.hop_length,
        peak_delta=args.peak_delta,
        max_candidates=args.max_candidates,
    )
    merged = merge_transitions(
        candidates=candidates,
        duration_s=duration_s,
        min_gap=args.min_transition_gap,
        edge_margin=args.edge_margin,
    )
    print(f"       Found {len(merged)} transitions after filtering")

    print("[3/5] Building segments...")
    segments = build_segments(
        duration_s=duration_s,
        transitions=merged,
        min_segment_dur=args.min_segment_duration,
    )
    print(f"       Created {len(segments)} segments")

    print("[4/5] Loading LiquidAI model...")
    # Check CUDA availability more carefully
    if args.force_cpu:
        device = "cpu"
    else:
        try:
            # Try to actually use CUDA to verify it works
            if torch.cuda.is_available():
                _ = torch.zeros(1).cuda()  # Test CUDA
                device = "cuda"
            else:
                device = "cpu"
        except (AssertionError, RuntimeError):
            device = "cpu"
    model, processor = load_liquid_model(device)
    print(f"       Model ready on {device}")

    print("[5/5] Identifying tracks...")
    results: List[SegmentResult] = []
    for idx, (start, end) in enumerate(segments, start=1):
        preview = slice_preview(
            waveform,
            sr,
            window=(start, end),
            preview_duration=args.preview_duration,
            lead_in=args.preview_lead_in,
        )
        if preview is None:
            track_name = "Unknown track"
        else:
            track_name = identify_track(
                model=model,
                processor=processor,
                segment=preview,
                sr=sr,
                max_new_tokens=args.max_new_tokens,
            )
        duration = end - start
        print(
            f"  Segment {idx:02d} | {seconds_to_clock(start)} → {seconds_to_clock(end)} "
            f"({duration:5.1f}s) | {track_name}"
        )
        results.append(
            SegmentResult(
                index=idx,
                start_s=start,
                end_s=end,
                duration_s=duration,
                track_name=track_name,
            )
        )

    return {
        "audio_path": str(audio_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sample_rate": sr,
        "duration_seconds": duration_s,
        "parameters": {
            "target_sr": args.target_sr,
            "hop_length": args.hop_length,
            "peak_delta": args.peak_delta,
            "max_candidates": args.max_candidates,
            "min_transition_gap": args.min_transition_gap,
            "edge_margin": args.edge_margin,
            "min_segment_duration": args.min_segment_duration,
            "preview_duration": args.preview_duration,
            "preview_lead_in": args.preview_lead_in,
        },
        "transitions_seconds": merged,
        "segments": [asdict(seg) for seg in results],
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a DJ mix and build a tracklist.")
    parser.add_argument("audio", help="Path to the downloaded audio file (wav/mp3/flac)")
    parser.add_argument("--out", help="Output JSON file (defaults to audio_path + .tracklist.json)")
    parser.add_argument("--target-sr", type=int, default=24_000, dest="target_sr")
    parser.add_argument("--hop-length", type=int, default=2048, dest="hop_length")
    parser.add_argument("--peak-delta", type=float, default=0.5, dest="peak_delta")
    parser.add_argument("--max-candidates", type=int, default=50, dest="max_candidates")
    parser.add_argument("--min-transition-gap", type=float, default=20.0, dest="min_transition_gap")
    parser.add_argument("--edge-margin", type=float, default=30.0, dest="edge_margin")
    parser.add_argument("--min-segment-duration", type=float, default=45.0, dest="min_segment_duration")
    parser.add_argument("--preview-duration", type=float, default=20.0, dest="preview_duration")
    parser.add_argument("--preview-lead-in", type=float, default=5.0, dest="preview_lead_in")
    parser.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    analysis = analyze_mix(args)

    out_path = Path(args.out) if args.out else Path(f"{args.audio}.tracklist.json")
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nSaved tracklist to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()


