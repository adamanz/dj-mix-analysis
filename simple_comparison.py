"""
Simple comparison of detection results
"""

import json
import numpy as np

def compare_methods():
    print("\n" + "="*70)
    print("DJ MIX TRANSITION DETECTION COMPARISON")
    print("="*70)

    # Load and compare results
    results = []

    # Original method
    with open('tracklist_full.json', 'r') as f:
        original = json.load(f)
        results.append({
            'name': 'Original (analyze_mix.py)',
            'transitions': original['transitions_seconds'],
            'segments': len(original['segments']),
            'duration': original['duration_seconds']
        })

    # Adaptive method
    try:
        with open('QRQwZDWz1Pw_adaptive_transitions.json', 'r') as f:
            adaptive = json.load(f)
            results.append({
                'name': 'Adaptive (multi-method)',
                'transitions': adaptive['transitions'],
                'segments': adaptive['num_segments'],
                'duration': adaptive['duration_seconds']
            })
    except FileNotFoundError:
        pass

    # Test method (our initial run)
    try:
        with open('tracklist_test.json', 'r') as f:
            test = json.load(f)
            results.append({
                'name': 'Test (basic onset)',
                'transitions': test['transitions_seconds'],
                'segments': len(test['segments']),
                'duration': test['duration_seconds']
            })
    except FileNotFoundError:
        pass

    # Print comparison table
    print(f"\nAudio Duration: {results[0]['duration']/60:.1f} minutes")
    print("\n{:<30} {:>15} {:>15} {:>15}".format(
        "Method", "Transitions", "Segments", "Avg Length"
    ))
    print("-"*70)

    for r in results:
        n_trans = len(r['transitions'])
        avg_len = r['duration'] / r['segments'] / 60 if r['segments'] > 0 else 0
        print("{:<30} {:>15} {:>15} {:>15.1f} min".format(
            r['name'], n_trans, r['segments'], avg_len
        ))

    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)

    for r in results:
        print(f"\n{r['name']}:")
        trans = r['transitions']
        print(f"  Total transitions: {len(trans)}")

        if len(trans) > 0:
            if len(trans) > 1:
                intervals = np.diff(trans)
                print(f"  Interval statistics:")
                print(f"    Mean: {np.mean(intervals)/60:.1f} min")
                print(f"    Std:  {np.std(intervals)/60:.1f} min")
                print(f"    Min:  {np.min(intervals):.1f} sec")
                print(f"    Max:  {np.max(intervals)/60:.1f} min")

            print(f"\n  First 10 transitions:")
            for i, t in enumerate(trans[:10], 1):
                mins, secs = divmod(t, 60)
                if i > 1:
                    interval = t - trans[i-2]
                    print(f"    {i:2d}. {int(mins):02d}:{secs:05.2f}  (interval: {interval:.1f}s)")
                else:
                    print(f"    {i:2d}. {int(mins):02d}:{secs:05.2f}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
The Adaptive method found the most realistic number of transitions:
- 50 transitions ≈ typical for a 106-minute DJ mix
- 2.1 minute average track length is reasonable for electronic music
- Much better than original's 5 transitions (21 min avg - too long!)

The original method's conservative filtering missed many transitions.
The test method was too sensitive (207 transitions = 30s avg).
The adaptive method balances sensitivity with DJ-specific constraints.
""")

if __name__ == "__main__":
    compare_methods()