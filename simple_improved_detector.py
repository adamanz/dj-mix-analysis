"""
Simplified but improved transition detector for DJ mixes
Focuses on spectral changes and adaptive thresholding
"""

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import json
from pathlib import Path

def detect_dj_transitions(audio_path, min_track_duration=90):
    """
    Improved transition detection using spectral novelty
    """
    print(f"[1/4] Loading audio: {audio_path}")
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    print(f"  Duration: {duration/60:.1f} minutes")

    print("[2/4] Computing spectral features...")
    hop_length = 512
    n_fft = 2048

    # Compute spectrogram
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)

    # Split into frequency bands
    n_bins = S.shape[0]
    bass = S[:n_bins//4, :]
    mids = S[n_bins//4:3*n_bins//4, :]
    highs = S[3*n_bins//4:, :]

    print("[3/4] Detecting transitions...")

    # Compute spectral novelty for each band
    def compute_novelty(band):
        # Spectral flux
        flux = np.sum(np.maximum(0, np.diff(band, axis=1)), axis=0)
        flux = np.pad(flux, (1, 0), mode='constant')

        # Smooth and normalize
        flux_smooth = uniform_filter1d(flux, size=50)
        if flux_smooth.std() > 0:
            flux_norm = (flux - flux_smooth) / flux_smooth.std()
        else:
            flux_norm = flux
        return flux_norm

    novelty_bass = compute_novelty(bass)
    novelty_mids = compute_novelty(mids)
    novelty_highs = compute_novelty(highs)

    # DJs often mix using bass/kick alignment
    combined_novelty = 0.5 * novelty_bass + 0.3 * novelty_mids + 0.2 * novelty_highs

    # Adaptive peak detection
    # Use different time scales to catch both cuts and crossfades
    short_window = uniform_filter1d(combined_novelty, size=100)  # ~1 second
    long_window = uniform_filter1d(combined_novelty, size=2000)  # ~20 seconds

    # Detect where short-term changes exceed long-term average significantly
    adaptive_threshold = long_window + 2 * np.std(combined_novelty)

    # Find peaks with adaptive thresholding
    peaks, properties = find_peaks(
        short_window,
        height=adaptive_threshold[:-1] if len(adaptive_threshold) > len(short_window) else adaptive_threshold,
        distance=int(min_track_duration * sr / hop_length),  # Minimum track duration
        prominence=np.std(combined_novelty)
    )

    # Convert to times
    transition_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # Add refinement: look for energy valleys near peaks (actual transition points)
    refined_times = []
    window_sec = 5  # Look within 5 seconds
    window_frames = int(window_sec * sr / hop_length)

    energy = librosa.feature.rms(S=S, hop_length=1)[0]

    for peak_frame in peaks:
        # Search for energy minimum around peak
        start = max(0, peak_frame - window_frames)
        end = min(len(energy), peak_frame + window_frames)

        if end > start:
            local_min = start + np.argmin(energy[start:end])
            refined_time = librosa.frames_to_time(local_min, sr=sr, hop_length=hop_length)
            refined_times.append(refined_time)

    print(f"  Found {len(refined_times)} transitions")

    # Create segments
    segments = []
    prev_time = 0

    for i, trans_time in enumerate(refined_times):
        segments.append({
            "index": i + 1,
            "start_s": prev_time,
            "end_s": trans_time,
            "duration_s": trans_time - prev_time,
        })
        prev_time = trans_time

    # Add final segment
    segments.append({
        "index": len(segments) + 1,
        "start_s": prev_time,
        "end_s": duration,
        "duration_s": duration - prev_time,
    })

    return {
        "transitions": refined_times,
        "segments": segments,
        "audio_duration": duration,
        "sample_rate": sr
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python simple_improved_detector.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    output_file = Path(audio_file).stem + "_transitions.json"

    results = detect_dj_transitions(audio_file)

    print(f"\n[4/4] Results:")
    print(f"  Total transitions: {len(results['transitions'])}")
    print(f"  Segments: {len(results['segments'])}")

    print(f"\nFirst 10 transitions:")
    for i, t in enumerate(results['transitions'][:10], 1):
        mins, secs = divmod(t, 60)
        print(f"  {i:2d}. {int(mins):02d}:{secs:05.2f}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")