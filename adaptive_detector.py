"""
Adaptive transition detector for DJ mixes
Uses multiple detection strategies and combines results
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import json
from pathlib import Path

def detect_transitions_multimethod(audio_path, min_track_duration=60):
    """
    Combined approach using multiple detection methods
    """
    print(f"Loading: {audio_path}")
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    print(f"Duration: {duration/60:.1f} minutes")

    hop_length = 512

    # Method 1: Tempogram-based (detect BPM changes)
    print("Method 1: Tempo analysis...")
    transitions_tempo = detect_tempo_changes(audio, sr, hop_length)

    # Method 2: Spectral contrast (detect timbral changes)
    print("Method 2: Spectral contrast...")
    transitions_spectral = detect_spectral_contrast_changes(audio, sr, hop_length)

    # Method 3: Self-similarity matrix (detect structural boundaries)
    print("Method 3: Self-similarity...")
    transitions_structure = detect_structural_boundaries(audio, sr, hop_length)

    # Combine all detected transitions
    all_transitions = sorted(set(
        list(transitions_tempo) +
        list(transitions_spectral) +
        list(transitions_structure)
    ))

    # Merge nearby transitions (within 10 seconds)
    merged = merge_nearby_transitions(all_transitions, threshold=10)

    # Filter by minimum track duration
    filtered = filter_by_duration(merged, min_track_duration)

    print(f"Found {len(filtered)} transitions after merging")

    return filtered, duration, sr

def detect_tempo_changes(audio, sr, hop_length):
    """Detect tempo/rhythm changes"""
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    # Compute tempogram
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)

    # Find dominant tempo over time
    dominant_tempo_idx = np.argmax(tempogram, axis=0)

    # Smooth to remove noise
    dominant_tempo_smooth = savgol_filter(dominant_tempo_idx, 101, 3)

    # Detect changes
    tempo_diff = np.abs(np.diff(dominant_tempo_smooth))
    tempo_diff = uniform_filter1d(tempo_diff, size=20)

    # Find peaks in tempo change
    threshold = np.percentile(tempo_diff, 85)
    peaks, _ = find_peaks(tempo_diff, height=threshold, distance=sr//hop_length * 30)  # Min 30 seconds

    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    return times

def detect_spectral_contrast_changes(audio, sr, hop_length):
    """Detect timbral changes using spectral contrast"""
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)

    # Compute frame-to-frame difference
    contrast_diff = np.sum(np.abs(np.diff(contrast, axis=1)), axis=0)

    # Smooth
    contrast_diff_smooth = uniform_filter1d(contrast_diff, size=50)

    # Adaptive thresholding
    threshold = np.mean(contrast_diff_smooth) + 1.5 * np.std(contrast_diff_smooth)

    # Find peaks
    peaks, _ = find_peaks(
        contrast_diff_smooth,
        height=threshold,
        distance=sr//hop_length * 45  # Min 45 seconds
    )

    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    return times

def detect_structural_boundaries(audio, sr, hop_length, max_duration=600):
    """Detect structural boundaries using self-similarity"""
    # Limit processing for long files
    if len(audio) / sr > max_duration:
        audio_segment = audio[:int(max_duration * sr)]
        print(f"  (Processing first {max_duration/60:.1f} minutes for structure)")
    else:
        audio_segment = audio

    # Compute chroma features
    chroma = librosa.feature.chroma_cqt(y=audio_segment, sr=sr, hop_length=hop_length)

    # Build self-similarity matrix
    sim_matrix = librosa.segment.recurrence_matrix(chroma, mode='affinity', metric='cosine')

    # Detect boundaries
    boundaries = librosa.segment.agglomerative(sim_matrix, 15)  # Target ~15 segments

    # Convert to time
    boundary_frames = np.where(np.diff(boundaries))[0]
    times = librosa.frames_to_time(boundary_frames, sr=sr, hop_length=hop_length)

    # Scale back to full duration if we processed a segment
    if len(audio) / sr > max_duration:
        scale_factor = len(audio) / len(audio_segment)
        # Only return boundaries from processed portion
        times = times[times < max_duration]

    return times

def merge_nearby_transitions(transitions, threshold=10):
    """Merge transitions within threshold seconds"""
    if not transitions:
        return []

    merged = [transitions[0]]
    for t in transitions[1:]:
        if t - merged[-1] > threshold:
            merged.append(t)
    return merged

def filter_by_duration(transitions, min_duration):
    """Filter transitions by minimum track duration"""
    if not transitions:
        return []

    filtered = []
    last = 0
    for t in transitions:
        if t - last >= min_duration:
            filtered.append(t)
            last = t
    return filtered

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python adaptive_detector.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    transitions, duration, sr = detect_transitions_multimethod(audio_file)

    print(f"\nResults:")
    print(f"  Transitions: {len(transitions)}")
    print(f"  Average track length: {duration/max(len(transitions), 1)/60:.1f} minutes")

    print(f"\nTransition times:")
    for i, t in enumerate(transitions, 1):
        mins, secs = divmod(t, 60)
        print(f"  {i:2d}. {int(mins):02d}:{secs:05.2f}")

    # Save results
    output = {
        "audio_file": audio_file,
        "duration_seconds": duration,
        "sample_rate": sr,
        "transitions": transitions,
        "num_segments": len(transitions) + 1
    }

    output_file = Path(audio_file).stem + "_adaptive_transitions.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_file}")