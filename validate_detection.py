"""
Validate our detection against the official Cercle tracklist
"""

import json
import numpy as np

# Official tracklist from Cercle dataset
OFFICIAL_TRACKLIST = [
    {"time": 0, "artist": "Worakls", "track": "Detached Motion"},
    {"time": 291, "artist": "Worakls", "track": "Question Réponse"},
    {"time": 610, "artist": "Worakls", "track": "Cloches"},
    {"time": 1007, "artist": "Worakls", "track": "Salzburg"},
    {"time": 1381, "artist": "Worakls", "track": "Nikki"},
    {"time": 1719, "artist": "Worakls", "track": "By The Brook"},
    {"time": 2116, "artist": "Worakls", "track": "Hortari"},
    {"time": 2538, "artist": "Worakls", "track": "Caprice"},
    {"time": 2854, "artist": "NTO", "track": "Trauma (Worakls Remix)"},
    {"time": 3312, "artist": "Worakls", "track": "Nocturne"},
    {"time": 3844, "artist": "Worakls", "track": "Adagio For Square"},
    {"time": 4296, "artist": "Worakls", "track": "Sanctis"},
    {"time": 4759, "artist": "Worakls", "track": "Crow"},
    {"time": 5131, "artist": "Worakls", "track": "Porto"},
]

def evaluate_detection():
    """Compare detected transitions with ground truth"""

    # Load our adaptive detector results
    with open('QRQwZDWz1Pw_adaptive_transitions.json', 'r') as f:
        adaptive = json.load(f)

    detected = adaptive['transitions']

    # Load original analyzer results
    with open('tracklist_full.json', 'r') as f:
        original = json.load(f)

    original_transitions = original['transitions_seconds']

    print("\n" + "="*80)
    print("DETECTION ACCURACY COMPARISON WITH GROUND TRUTH")
    print("="*80)

    print(f"\nGround Truth: {len(OFFICIAL_TRACKLIST)} tracks")
    print(f"Original detector: {len(original_transitions)} transitions")
    print(f"Adaptive detector: {len(detected)} transitions")

    # Calculate accuracy for each method
    def calculate_accuracy(detected_times, tolerance=10):
        """Check how many true transitions were detected within tolerance"""
        true_times = [t['time'] for t in OFFICIAL_TRACKLIST[1:]]  # Skip first at 0

        matches = []
        for true_time in true_times:
            # Find closest detection
            if len(detected_times) > 0:
                distances = [abs(d - true_time) for d in detected_times]
                min_dist = min(distances)
                if min_dist <= tolerance:
                    matches.append((true_time, detected_times[distances.index(min_dist)]))

        precision = len(matches) / len(detected_times) if detected_times else 0
        recall = len(matches) / len(true_times) if true_times else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'matches': matches,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': len(matches),
            'false_positives': len(detected_times) - len(matches),
            'false_negatives': len(true_times) - len(matches)
        }

    print("\n" + "-"*80)
    print("ORIGINAL DETECTOR (analyze_mix.py)")
    print("-"*80)

    original_acc = calculate_accuracy(original_transitions, tolerance=15)
    print(f"Precision: {original_acc['precision']:.2%} (correct detections / total detections)")
    print(f"Recall: {original_acc['recall']:.2%} (found transitions / true transitions)")
    print(f"F1 Score: {original_acc['f1']:.3f}")
    print(f"True Positives: {original_acc['true_positives']}")
    print(f"False Positives: {original_acc['false_positives']}")
    print(f"False Negatives: {original_acc['false_negatives']}")

    print("\n" + "-"*80)
    print("ADAPTIVE DETECTOR")
    print("-"*80)

    adaptive_acc = calculate_accuracy(detected, tolerance=15)
    print(f"Precision: {adaptive_acc['precision']:.2%} (correct detections / total detections)")
    print(f"Recall: {adaptive_acc['recall']:.2%} (found transitions / true transitions)")
    print(f"F1 Score: {adaptive_acc['f1']:.3f}")
    print(f"True Positives: {adaptive_acc['true_positives']}")
    print(f"False Positives: {adaptive_acc['false_positives']}")
    print(f"False Negatives: {adaptive_acc['false_negatives']}")

    print("\n" + "="*80)
    print("OFFICIAL TRACKLIST WITH DETECTION COMPARISON")
    print("="*80)

    # Show which tracks were correctly detected
    for i, track in enumerate(OFFICIAL_TRACKLIST):
        time = track['time']
        artist = track['artist']
        title = track['track']

        mins, secs = divmod(time, 60)

        # Check if detected by original
        orig_detected = False
        for ot in original_transitions:
            if abs(ot - time) <= 15:
                orig_detected = True
                break

        # Check if detected by adaptive
        adapt_detected = False
        adapt_time = None
        for dt in detected:
            if abs(dt - time) <= 15:
                adapt_detected = True
                adapt_time = dt
                break

        status_orig = "✓" if orig_detected else "✗"
        status_adapt = "✓" if adapt_detected else "✗"

        if adapt_time:
            error = adapt_time - time
            error_str = f"(error: {error:+.1f}s)"
        else:
            error_str = ""

        print(f"{i+1:2d}. [{mins:02d}:{secs:02d}] {artist:15s} - {title:25s} | Orig: {status_orig} | Adapt: {status_adapt} {error_str}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print("""
The adaptive detector found many false positives (50 transitions vs 14 tracks)
because it's detecting musical sections within each orchestral piece.

For orchestral performances, we need:
1. Higher minimum segment duration (5-8 minutes)
2. Focus on major key/tempo changes only
3. Use hierarchical clustering to group sub-segments
4. Train specifically on orchestral electronic music

The original detector was actually closer to the truth (5 transitions)
but still missed most track boundaries.
""")

if __name__ == "__main__":
    evaluate_detection()