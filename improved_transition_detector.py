"""
Improved DJ Mix Transition Detection
Using multi-feature analysis for better boundary detection
"""

import numpy as np
import librosa
import scipy.signal
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import List, Tuple
import soundfile as sf

@dataclass
class TransitionCandidate:
    time: float
    confidence: float
    type: str  # 'cut', 'crossfade', 'eq_sweep'

class DJTransitionDetector:
    def __init__(
        self,
        sr: int = 44100,
        hop_length: int = 512,
        n_fft: int = 2048,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

    def detect_transitions(self, audio: np.ndarray) -> List[TransitionCandidate]:
        """
        Multi-feature transition detection combining:
        1. Spectral novelty (for track changes)
        2. Beat tracking discontinuities
        3. Harmonic-percussive separation
        4. Energy distribution changes
        """
        transitions = []

        # 1. Compute spectral features
        spectral_novelty = self._compute_spectral_novelty(audio)

        # 2. Track tempo/beat changes
        beat_changes = self._detect_beat_changes(audio)

        # 3. Detect crossfader patterns
        crossfade_points = self._detect_crossfades(audio)

        # 4. Harmonic content changes
        harmonic_changes = self._detect_harmonic_changes(audio)

        # 5. Combine all features with weighted voting
        all_candidates = self._combine_features(
            spectral_novelty,
            beat_changes,
            crossfade_points,
            harmonic_changes
        )

        # 6. Apply DJ-specific constraints
        transitions = self._filter_dj_constraints(all_candidates)

        return transitions

    def _compute_spectral_novelty(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral novelty using multiple frequency bands
        Better than simple onset detection for DJ mixes
        """
        # Compute spectrogram
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        S = np.abs(D)

        # Split into frequency bands (bass, mid, high)
        n_bins = S.shape[0]
        bass_idx = n_bins // 4
        mid_idx = 3 * n_bins // 4

        # Compute novelty for each band
        novelty_bass = self._local_novelty(S[:bass_idx, :])
        novelty_mid = self._local_novelty(S[bass_idx:mid_idx, :])
        novelty_high = self._local_novelty(S[mid_idx:, :])

        # Weight bass higher (DJs often mix using bass/kicks)
        novelty = 0.5 * novelty_bass + 0.3 * novelty_mid + 0.2 * novelty_high

        return novelty

    def _local_novelty(self, S: np.ndarray) -> np.ndarray:
        """
        Compute local spectral novelty (better than onset strength)
        """
        # Compute spectral flux with rectification
        flux = np.sum(np.maximum(0, np.diff(S, axis=1)), axis=0)
        flux = np.pad(flux, (1, 0), mode='constant')

        # Smooth and normalize
        flux_smooth = uniform_filter1d(flux, size=10)
        flux_norm = (flux - flux_smooth) / (flux_smooth + 1e-6)

        return flux_norm

    def _detect_beat_changes(self, audio: np.ndarray) -> List[float]:
        """
        Detect tempo/beat grid changes that indicate transitions
        """
        # Track beats with dynamic programming
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Compute inter-beat intervals
        beat_times = librosa.frames_to_time(beats, sr=self.sr, hop_length=self.hop_length)
        ibi = np.diff(beat_times)

        # Detect significant tempo changes
        tempo_changes = []
        window_size = 16  # ~4 bars at 120 BPM

        for i in range(window_size, len(ibi) - window_size):
            before = np.median(ibi[i-window_size:i])
            after = np.median(ibi[i:i+window_size])

            # Significant tempo change (>5%)
            if abs(after - before) / before > 0.05:
                tempo_changes.append(beat_times[i])

        return tempo_changes

    def _detect_crossfades(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """
        Detect crossfader movements by analyzing stereo field and energy
        """
        # Compute short-time energy
        frame_length = self.n_fft
        hop_length = self.hop_length

        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames**2, axis=0)

        # Smooth energy curve
        energy_smooth = uniform_filter1d(energy, size=20)

        # Find gradual energy changes (crossfades)
        crossfades = []
        min_crossfade_duration = int(2.0 * self.sr / hop_length)  # 2 seconds minimum

        # Compute derivative to find slopes
        energy_diff = np.gradient(energy_smooth)

        # Find sustained positive/negative slopes (fading in/out)
        threshold = np.std(energy_diff) * 0.5

        i = 0
        while i < len(energy_diff) - min_crossfade_duration:
            # Check for sustained slope
            window = energy_diff[i:i+min_crossfade_duration]

            if np.all(window > threshold) or np.all(window < -threshold):
                start_time = librosa.frames_to_time(i, sr=self.sr, hop_length=hop_length)
                end_time = librosa.frames_to_time(i + min_crossfade_duration, sr=self.sr, hop_length=hop_length)
                crossfades.append((start_time, end_time))
                i += min_crossfade_duration
            else:
                i += 1

        return crossfades

    def _detect_harmonic_changes(self, audio: np.ndarray) -> List[float]:
        """
        Use harmonic-percussive separation to detect melodic changes
        """
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)

        # Separate harmonic and percussive
        D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=3.0)

        # Compute chroma features from harmonic content
        chroma = librosa.feature.chroma_stft(S=np.abs(D_harmonic), sr=self.sr)

        # Compute chroma change (key/harmony changes)
        chroma_diff = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)

        # Find peaks in chroma change
        peaks, properties = scipy.signal.find_peaks(
            chroma_diff,
            height=np.percentile(chroma_diff, 90),
            distance=int(10 * self.sr / self.hop_length)  # Min 10 seconds between changes
        )

        times = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        return times.tolist()

    def _combine_features(self, *features) -> List[TransitionCandidate]:
        """
        Combine multiple detection features with confidence scoring
        """
        candidates = []

        # Convert all features to time-indexed events
        # ... (implementation for combining and scoring)

        return candidates

    def _filter_dj_constraints(self, candidates: List[TransitionCandidate]) -> List[TransitionCandidate]:
        """
        Apply DJ-specific constraints:
        - Minimum track length (usually 2-3 minutes)
        - Typical transition patterns
        - Remove false positives from breaks/drops
        """
        filtered = []
        min_track_duration = 120.0  # 2 minutes minimum

        last_transition = 0
        for candidate in sorted(candidates, key=lambda x: x.time):
            if candidate.time - last_transition >= min_track_duration:
                filtered.append(candidate)
                last_transition = candidate.time

        return filtered

def analyze_dj_mix(audio_path: str) -> List[TransitionCandidate]:
    """
    Main function to analyze a DJ mix
    """
    # Load audio
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Initialize detector
    detector = DJTransitionDetector(sr=sr)

    # Detect transitions
    transitions = detector.detect_transitions(audio)

    return transitions

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        transitions = analyze_dj_mix(audio_file)

        print(f"Detected {len(transitions)} transitions:")
        for i, t in enumerate(transitions, 1):
            mins, secs = divmod(t.time, 60)
            print(f"  {i:2d}. {int(mins):02d}:{secs:05.2f} - {t.type} (confidence: {t.confidence:.2f})")