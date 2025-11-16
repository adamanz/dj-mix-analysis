"""
Optimized transition detector for orchestral electronic performances
Tuned specifically for Worakls Orchestra and similar live sets
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import json
from pathlib import Path

def detect_orchestral_transitions(
    audio_path,
    min_track_duration=240,  # 4 minutes minimum (orchestral pieces are longer)
    sensitivity=0.3  # Lower = more conservative
):
    """
    Specialized detector for orchestral electronic music
    """
    print(f"[1/5] Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    print(f"  Duration: {duration/60:.1f} minutes")

    hop_length = 512
    n_fft = 4096  # Larger window for orchestral analysis

    print("[2/5] Computing harmonic-percussive separation...")
    # Separate harmonic (orchestral) and percussive (electronic beats)
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=3.0)

    print("[3/5] Detecting key and harmony changes...")
    # Chromagram from harmonic content
    chroma = librosa.feature.chroma_cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length * 4  # Downsample for stability
    )

    # Compute chroma shift (key changes)
    chroma_diff = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)

    # Heavy smoothing for orchestral pieces
    chroma_smooth = uniform_filter1d(chroma_diff, size=500)

    # Normalize
    if chroma_smooth.std() > 0:
        chroma_norm = (chroma_smooth - chroma_smooth.mean()) / chroma_smooth.std()
    else:
        chroma_norm = chroma_smooth

    print("[4/5] Detecting structural boundaries...")
    # Use longer-term self-similarity
    # Downsample heavily for macro structure
    beat_sync_chroma = librosa.util.sync(chroma,
                                         librosa.beat.beat_track(y=audio, sr=sr)[1])

    # Self-similarity matrix
    sim = librosa.segment.recurrence_matrix(
        beat_sync_chroma,
        mode='affinity',
        metric='cosine',
        width=3,
        sym=True
    )

    # Novelty curve from similarity
    novelty = np.sum(np.diff(sim, axis=1), axis=0)
    novelty = uniform_filter1d(novelty, size=10)

    print("[5/5] Finding track boundaries...")
    # Combine features with weighting
    combined = (
        0.6 * chroma_norm[:len(novelty)] +  # Harmonic changes (most important)
        0.4 * novelty / novelty.std() if novelty.std() > 0 else novelty  # Structure
    )

    # Apply heavy smoothing
    combined_smooth = uniform_filter1d(combined, size=100)

    # Dynamic thresholding based on local statistics
    threshold = np.mean(combined_smooth) + sensitivity * np.std(combined_smooth)

    # Find peaks with constraints
    min_distance = int(min_track_duration * sr / (hop_length * 4))  # In downsampled frames

    peaks, properties = find_peaks(
        combined_smooth,
        height=threshold,
        distance=min_distance,
        prominence=np.std(combined_smooth) * 0.5
    )

    # Convert to time
    times = peaks * (hop_length * 4) / sr

    # Add manual corrections for common patterns
    # Orchestral pieces often have false peaks at crescendos
    filtered_times = []
    for i, t in enumerate(times):
        # Skip if too close to start/end
        if t < 30 or t > duration - 30:
            continue

        # Check if this is a real transition by looking at energy
        window_sec = 10
        start_idx = max(0, int((t - window_sec) * sr))
        end_idx = min(len(audio), int((t + window_sec) * sr))

        segment = audio[start_idx:end_idx]
        energy = librosa.feature.rms(y=segment, hop_length=hop_length)[0]

        # Real transitions often have energy valleys
        mid_point = len(energy) // 2
        mid_energy = np.mean(energy[mid_point-5:mid_point+5])
        avg_energy = np.mean(energy)

        if mid_energy < avg_energy * 1.2:  # Not a crescendo
            filtered_times.append(t)

    return filtered_times, duration, sr

def test_on_worakls():
    """Test on the Worakls Orchestra set"""
    audio_file = "/Users/adamanz/cercle-transition-dataset/audio/QRQwZDWz1Pw.mp3"

    # Known ground truth
    ground_truth = [0, 291, 610, 1007, 1381, 1719, 2116, 2538, 2854, 3312, 3844, 4296, 4759, 5131]

    print("\n" + "="*60)
    print("ORCHESTRAL DETECTOR TEST")
    print("="*60)

    transitions, duration, sr = detect_orchestral_transitions(
        audio_file,
        min_track_duration=240,  # 4 minutes
        sensitivity=0.3
    )

    print(f"\nResults:")
    print(f"  Ground truth: {len(ground_truth)} tracks")
    print(f"  Detected: {len(transitions)} transitions")

    print(f"\nDetected transitions:")
    for i, t in enumerate(transitions, 1):
        mins, secs = divmod(t, 60)

        # Find closest ground truth
        closest_gt = min(ground_truth[1:], key=lambda x: abs(x - t))
        error = t - closest_gt

        print(f"  {i:2d}. {int(mins):02d}:{secs:05.2f} (error from nearest: {error:+.1f}s)")

    # Calculate accuracy
    matches = 0
    for gt in ground_truth[1:]:  # Skip first at 0
        for dt in transitions:
            if abs(dt - gt) <= 20:  # 20 second tolerance
                matches += 1
                break

    precision = matches / len(transitions) if transitions else 0
    recall = matches / (len(ground_truth) - 1) if len(ground_truth) > 1 else 0

    print(f"\nAccuracy:")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall: {recall:.1%}")
    print(f"  F1: {2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0:.3f}")

if __name__ == "__main__":
    test_on_worakls()