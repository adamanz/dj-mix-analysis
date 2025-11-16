# DJ Mix Analysis Demo

Automated DJ mix analysis using LiquidAI LFM2-Audio-1.5B to detect transitions and identify tracks in DJ sets.

## Features

- **Automatic Transition Detection**: Uses onset detection and spectral analysis to find track transitions
- **Track Identification**: Leverages LiquidAI LFM2-Audio-1.5B model to identify tracks from audio segments
- **Interactive Playback**: Play analyzed mixes with real-time tracklist overlay

## Quick Start

### Prerequisites

- Python 3.12+
- PyTorch (CPU or CUDA/ROCm)
- Audio file (WAV, MP3, or FLAC format)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install liquid-audio package
pip install liquid-audio
```

### Usage

#### 1. Analyze a DJ Mix

```bash
python analyze_mix.py path/to/mix.wav --out tracklist.json
```

This will:
- Load and analyze the audio file
- Detect transition boundaries
- Split into segments
- Identify each track using LFM2-Audio-1.5B
- Save results to a JSON tracklist

#### 2. Play Mix with Tracklist

```bash
python demo_play_mix.py path/to/mix.wav --tracklist tracklist.json
```

This will:
- Load the audio and tracklist
- Play the mix with real-time track information
- Display transitions and track names as they occur

## Files

- **`analyze_mix.py`** - Main analysis script that processes DJ mixes
- **`demo_play_mix.py`** - Interactive playback with tracklist overlay
- **`test_liquid_audio.py`** - Quick test to verify audio model setup
- **`run_liquid_audio_newest.py`** - Examples and documentation for LiquidAI audio model

## Configuration

The `analyze_mix.py` script supports various parameters:

```bash
python analyze_mix.py mix.wav \
  --target-sr 24000 \
  --min-segment-duration 45.0 \
  --preview-duration 20.0 \
  --max-new-tokens 96 \
  --force-cpu  # Use CPU instead of GPU
```

## Model Information

This demo uses **LiquidAI LFM2-Audio-1.5B**, a 1.5B parameter audio model optimized for:
- Real-time speech-to-speech conversations
- Low-latency audio processing
- Multi-modal understanding (audio + text)

**Model Repository**: [LiquidAI/LFM2-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B)

## Requirements

See `requirements.txt` for full dependency list. Key packages:
- `liquid-audio` - LiquidAI audio model
- `torch` / `torchaudio` - PyTorch for audio processing
- `librosa` - Audio analysis and transition detection
- `soundfile` - Audio file I/O
- `sounddevice` - Real-time audio playback
- `numpy` - Numerical operations

## Troubleshooting

### Model Download Issues
- Ensure you have internet connection for first-time model download (~3-6GB)
- Check disk space availability

### CUDA/GPU Issues
- Use `--force-cpu` flag if GPU is not available
- For AMD GPUs, install PyTorch with ROCm support

### Audio Loading Issues
- Ensure audio file is in supported format (WAV, MP3, FLAC)
- Check file path is correct

## License

See individual file headers for license information.

## Acknowledgments

- [LiquidAI](https://liquid.ai/) for the LFM2-Audio-1.5B model
- [librosa](https://librosa.org/) for audio analysis tools

