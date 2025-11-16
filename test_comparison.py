"""
Compare different transition detection methods on the same audio file
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from pathlib import Path

def load_results():
    """Load all detection results"""
    results = {}

    # Original analyze_mix.py results
    with open('tracklist_full.json', 'r') as f:
        original = json.load(f)
        results['original'] = {
            'transitions': original['transitions_seconds'],
            'segments': original['segments'],
            'method': 'Original (onset detection)'
        }

    # Adaptive detector results
    try:
        with open('QRQwZDWz1Pw_adaptive_transitions.json', 'r') as f:
            adaptive = json.load(f)
            results['adaptive'] = {
                'transitions': adaptive['transitions'],
                'method': 'Adaptive (multi-method)'
            }
    except FileNotFoundError:
        print("Adaptive results not found")

    # Our initial test results
    try:
        with open('tracklist_test.json', 'r') as f:
            test = json.load(f)
            # Sample every 10th transition for readability
            results['test'] = {
                'transitions': test['transitions_seconds'][::10][:20],
                'method': 'Test (basic onset, sampled)'
            }
    except FileNotFoundError:
        print("Test results not found")

    return results

def visualize_transitions(audio_path, results):
    """Create visualization comparing different detection methods"""

    # Load a small portion of audio for visualization
    print("Loading audio for visualization...")
    y, sr = librosa.load(audio_path, duration=600, sr=22050)  # First 10 minutes

    # Compute spectrogram
    print("Computing spectrogram...")
    hop_length = 512
    D = librosa.stft(y, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create figure
    fig, axes = plt.subplots(nrows=len(results)+1, figsize=(14, 3*(len(results)+1)), sharex=True)

    # Plot spectrogram
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='log',
                                   sr=sr, hop_length=hop_length, ax=axes[0])
    axes[0].set_title('Spectrogram (First 10 minutes)', fontsize=12)
    axes[0].set_ylabel('Frequency (Hz)')

    # Plot transitions for each method
    colors = ['red', 'blue', 'green', 'orange']
    for idx, (name, data) in enumerate(results.items(), 1):
        ax = axes[idx]
        transitions = [t for t in data['transitions'] if t <= 600]  # Only first 10 mins

        # Plot onset strength for context
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
        ax.plot(times, onset_env / onset_env.max(), alpha=0.3, color='gray', label='Onset strength')

        # Plot detected transitions
        for t in transitions:
            ax.axvline(x=t, color=colors[idx-1], alpha=0.7, linestyle='--', linewidth=2)

        ax.set_title(f"{data['method']} - {len(transitions)} transitions in first 10 min", fontsize=11)
        ax.set_ylabel('Normalized')
        ax.set_xlim(0, 600)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('transition_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to transition_comparison.png")
    plt.show()

def print_statistics(results):
    """Print comparison statistics"""
    print("\n" + "="*60)
    print("TRANSITION DETECTION COMPARISON")
    print("="*60)

    for name, data in results.items():
        transitions = data['transitions']
        print(f"\n{data['method']}:")
        print(f"  Total transitions: {len(transitions)}")

        if len(transitions) > 1:
            intervals = np.diff(transitions)
            print(f"  Average interval: {np.mean(intervals):.1f}s ({np.mean(intervals)/60:.1f} min)")
            print(f"  Min interval: {np.min(intervals):.1f}s")
            print(f"  Max interval: {np.max(intervals):.1f}s")
            print(f"  Std deviation: {np.std(intervals):.1f}s")

        if 'segments' in data:
            durations = [s['duration_s'] for s in data['segments']]
            print(f"  Segments: {len(data['segments'])}")
            print(f"  Avg segment duration: {np.mean(durations)/60:.1f} min")

        # Show first 5 transitions
        print(f"  First 5 transitions:")
        for i, t in enumerate(transitions[:5], 1):
            mins, secs = divmod(t, 60)
            print(f"    {i}. {int(mins):02d}:{secs:05.2f}")

def create_better_detector():
    """Create an optimized detector based on analysis"""
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR BETTER DETECTION")
    print("="*60)

    recommendations = """
1. **Use Adaptive Method**: 50 transitions is more realistic than 5
   - Average track length of 2.1 minutes matches typical DJ set patterns

2. **Key Improvements Needed**:
   - Add crossfade detection (overlapping tracks)
   - Implement confidence scoring for each transition
   - Use genre-specific tuning (house vs techno vs drum & bass)

3. **For Track Identification**:
   - Sample 30-60 seconds from middle of each segment
   - Avoid transition zones (±15 seconds from boundaries)
   - Use multiple samples if segment > 3 minutes

4. **Validation Strategy**:
   - Manual verification on subset
   - Check against known tracklists if available
   - Use crowd-sourcing for difficult segments
"""
    print(recommendations)

if __name__ == "__main__":
    audio_path = "/Users/adamanz/cercle-transition-dataset/audio/QRQwZDWz1Pw.mp3"

    print("Loading detection results...")
    results = load_results()

    # Print statistics
    print_statistics(results)

    # Create visualization
    print("\nCreating visualization...")
    try:
        visualize_transitions(audio_path, results)
    except Exception as e:
        print(f"Visualization failed: {e}")

    # Show recommendations
    create_better_detector()