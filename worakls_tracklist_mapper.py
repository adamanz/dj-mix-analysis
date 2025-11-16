"""
Map detected transitions to actual Worakls Orchestra tracklist
"""

import json
import numpy as np

# Official tracklist for Worakls Orchestra at Château La Coste
OFFICIAL_TRACKLIST = [
    "Detached Motion",
    "Question Réponse",
    "Cloches",
    "Salzburg",
    "Nikki",
    "By The Brook",
    "Hortari",
    "Caprice",
    "Toi",
    "Trauma (Worakls Remix)",
    "Nocturne",
    "Adagio For Square",
    "Sanctis",
    "Crow",
    # "Porto" - sometimes listed as track 15
]

def analyze_detection_accuracy():
    """
    Compare our detected transitions with expected track count
    """
    # Load our detection results
    with open('QRQwZDWz1Pw_adaptive_transitions.json', 'r') as f:
        adaptive = json.load(f)

    transitions = adaptive['transitions']
    duration = adaptive['duration_seconds']

    print("\n" + "="*70)
    print("WORAKLS ORCHESTRA - CHÂTEAU LA COSTE ANALYSIS")
    print("="*70)

    print(f"\nPerformance Details:")
    print(f"  Event: Worakls Orchestra live at Château La Coste")
    print(f"  Date: April 8, 2019")
    print(f"  Platform: Cercle")
    print(f"  Duration: {duration/60:.1f} minutes")
    print(f"  Official tracks: {len(OFFICIAL_TRACKLIST)}")

    print(f"\nDetection Results:")
    print(f"  Transitions detected: {len(transitions)}")
    print(f"  Average segment: {duration/len(transitions)/60:.1f} minutes")

    # Expected average track length
    expected_avg = duration / len(OFFICIAL_TRACKLIST) / 60
    print(f"\nExpected average track length: {expected_avg:.1f} minutes")

    # The issue: we detected 50 transitions for 14 tracks
    # This suggests the detector is finding:
    # 1. Song sections/movements (orchestral pieces have multiple parts)
    # 2. Dynamic changes within songs
    # 3. Orchestral arrangements creating false positives

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print("""
Why we detected 50 transitions instead of 14:

1. ORCHESTRAL MOVEMENTS: Each track likely has 3-4 distinct sections
   - Classical structure: Intro, Theme, Variation, Outro
   - Orchestral dynamics create strong spectral changes

2. LIVE PERFORMANCE DYNAMICS:
   - Build-ups and breakdowns within songs
   - Orchestral crescendos and diminuendos
   - Instrument solos and ensemble changes

3. DETECTION SENSITIVITY:
   - Tempo changes within pieces (common in orchestral works)
   - Spectral contrast changes when instruments enter/exit
   - Self-similarity detecting repeating motifs as boundaries

SOLUTION: For orchestral electronic performances, we need to:
- Increase minimum segment duration (4-8 minutes)
- Reduce sensitivity to spectral changes
- Focus on major tempo/key changes only
- Use hierarchical segmentation (songs -> sections)
""")

    # Try to map transitions to likely track boundaries
    print("\nLikely Track Boundaries (every 3-4 transitions):")

    estimated_tracks = []
    transitions_per_track = len(transitions) / len(OFFICIAL_TRACKLIST)

    for i, track_name in enumerate(OFFICIAL_TRACKLIST):
        # Estimate where this track should start
        transition_idx = int(i * transitions_per_track)
        if transition_idx < len(transitions):
            track_start = transitions[transition_idx]
        else:
            track_start = transitions[-1] if i > 0 else 0

        # Estimate where it ends
        next_idx = int((i + 1) * transitions_per_track)
        if next_idx < len(transitions):
            track_end = transitions[next_idx]
        else:
            track_end = duration

        mins_start, secs_start = divmod(track_start, 60)
        mins_end, secs_end = divmod(track_end, 60)
        duration_min = (track_end - track_start) / 60

        print(f"  {i+1:2d}. {track_name:25s} | {int(mins_start):02d}:{secs_start:05.2f} - {int(mins_end):02d}:{secs_end:05.2f} ({duration_min:.1f} min)")

        estimated_tracks.append({
            'index': i + 1,
            'name': track_name,
            'start_s': track_start,
            'end_s': track_end,
            'duration_s': track_end - track_start
        })

    # Save mapped tracklist
    output = {
        'performance': 'Worakls Orchestra at Château La Coste',
        'date': '2019-04-08',
        'platform': 'Cercle',
        'video_id': 'QRQwZDWz1Pw',
        'duration_seconds': duration,
        'official_track_count': len(OFFICIAL_TRACKLIST),
        'detected_transitions': len(transitions),
        'estimated_tracks': estimated_tracks,
        'note': 'Transitions mapped to official tracklist using proportional distribution'
    }

    with open('worakls_mapped_tracklist.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved mapped tracklist to worakls_mapped_tracklist.json")

if __name__ == "__main__":
    analyze_detection_accuracy()