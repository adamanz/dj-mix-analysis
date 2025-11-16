"""
Identify tracks using the improved transition detection
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
import torch

def identify_tracks_from_segments(audio_path, transitions_file, output_file):
    """
    Use adaptive transitions to identify tracks with LiquidAI
    """
    print("[1/4] Loading audio and transitions...")

    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr

    # Load transitions
    with open(transitions_file, 'r') as f:
        data = json.load(f)

    transitions = data['transitions']

    # Create segments
    segments = []
    prev_time = 0

    for i, trans_time in enumerate(transitions):
        segments.append({
            'index': i + 1,
            'start_s': prev_time,
            'end_s': trans_time,
            'duration_s': trans_time - prev_time,
        })
        prev_time = trans_time

    # Add final segment
    segments.append({
        'index': len(segments) + 1,
        'start_s': prev_time,
        'end_s': duration,
        'duration_s': duration - prev_time,
    })

    print(f"  Found {len(segments)} segments to identify")

    # Load model
    print("[2/4] Loading LiquidAI model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LFM2AudioModel.from_pretrained("LiquidAI/LFM2-Audio-1.5B")
    processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2-Audio-1.5B")
    model = model.to(device)
    model.eval()

    chat_state = ChatState()

    print("[3/4] Identifying tracks...")
    identified_segments = []

    for seg in segments:
        # Skip very short segments (likely transitions)
        if seg['duration_s'] < 30:
            seg['track_name'] = "Transition/Mix"
            seg['confidence'] = 0.0
            identified_segments.append(seg)
            continue

        # Sample from middle of segment to avoid transitions
        segment_middle = (seg['start_s'] + seg['end_s']) / 2
        sample_start = max(0, segment_middle - 15)  # 30 second sample
        sample_end = min(len(audio) / sr, segment_middle + 15)

        # Extract audio segment
        start_sample = int(sample_start * sr)
        end_sample = int(sample_end * sr)
        audio_segment = audio[start_sample:end_sample]

        # Multiple prompting strategies for better results
        prompts = [
            "Identify this electronic dance music track. What is the song name and artist?",
            "What song is playing in this DJ mix? Provide artist and track title.",
            "This is from a house/techno DJ set. Name the track and producer."
        ]

        responses = []

        for prompt in prompts:
            try:
                # Process with model
                inputs = processor(
                    audio=audio_segment,
                    sampling_rate=sr,
                    text=prompt,
                    return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100)

                response = processor.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)

            except Exception as e:
                print(f"  Error processing segment {seg['index']}: {e}")
                responses.append("Unknown")

        # Combine responses (could use voting or similarity)
        # For now, take the most detailed response
        best_response = max(responses, key=len)

        # Clean up response
        if "Unknown" in best_response or len(best_response) < 5:
            track_name = f"Track {seg['index']} (unidentified)"
            confidence = 0.2
        else:
            track_name = best_response.strip()
            # Simple confidence based on response consistency
            confidence = len(set(responses)) / len(responses)

        seg['track_name'] = track_name
        seg['confidence'] = confidence

        identified_segments.append(seg)

        mins, secs = divmod(seg['start_s'], 60)
        print(f"  Segment {seg['index']:2d} | {int(mins):02d}:{secs:05.2f} | {track_name} (conf: {confidence:.2f})")

    # Save results
    print(f"\n[4/4] Saving to {output_file}")

    output = {
        'audio_path': str(audio_path),
        'duration_seconds': duration,
        'sample_rate': sr,
        'segments': identified_segments,
        'identification_method': 'LiquidAI LFM2-Audio-1.5B with adaptive transitions'
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    high_conf = sum(1 for s in identified_segments if s.get('confidence', 0) > 0.7)
    med_conf = sum(1 for s in identified_segments if 0.3 < s.get('confidence', 0) <= 0.7)
    low_conf = sum(1 for s in identified_segments if s.get('confidence', 0) <= 0.3)

    print(f"\nSummary:")
    print(f"  Total segments: {len(identified_segments)}")
    print(f"  High confidence: {high_conf}")
    print(f"  Medium confidence: {med_conf}")
    print(f"  Low confidence: {low_conf}")

    return identified_segments

if __name__ == "__main__":
    audio_file = "/Users/adamanz/cercle-transition-dataset/audio/QRQwZDWz1Pw.mp3"
    transitions_file = "QRQwZDWz1Pw_adaptive_transitions.json"
    output_file = "QRQwZDWz1Pw_identified_tracks.json"

    segments = identify_tracks_from_segments(audio_file, transitions_file, output_file)