"""
Liquid AI LFM2-Audio-1.5B - Latest Model Runner
Based on official documentation from Hugging Face and GitHub

This script demonstrates how to run the newest Liquid Audio model:
- Multi-turn, multi-modal chat (speech-to-speech)
- ASR (Automatic Speech Recognition)
- TTS (Text-to-Speech)
- Real-time interleaved generation
"""

import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def check_dependencies():
    """Check if required packages are installed"""
    print("="*60)
    print("Checking Dependencies")
    print("="*60)
    
    missing = []
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} installed")
    except ImportError:
        missing.append("torch")
        print("[X] PyTorch not installed")
    
    try:
        import torchaudio
        print(f"[OK] torchaudio installed")
    except ImportError:
        missing.append("torchaudio")
        print("[X] torchaudio not installed")
    
    try:
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
        print("[OK] liquid-audio installed")
        return True, None
    except ImportError:
        print("[X] liquid-audio not installed")
        print("\nTo install liquid-audio:")
        print("  pip install liquid-audio")
        print("  pip install 'liquid-audio[demo]'  # for demo interface")
        print("\nNote: liquid-audio requires Python 3.12+")
        return False, missing
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False, missing
    
    return True, None

def demo_gradio():
    """Launch Gradio demo interface"""
    print("\n" + "="*60)
    print("Gradio Demo Interface")
    print("="*60)
    print("\nTo launch the Gradio demo, run:")
    print("  liquid-audio-demo")
    print("\nThis will start a webserver on http://localhost:7860")
    print("\nMake sure you installed demo dependencies:")
    print("  pip install 'liquid-audio[demo]'")

def demo_multi_turn_chat():
    """Demo: Multi-turn, multi-modal chat with interleaved generation"""
    print("\n" + "="*60)
    print("Multi-turn, Multi-modal Chat Example")
    print("="*60)
    
    try:
        import torch
        import torchaudio
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
        
        # Load models
        HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
        print(f"\nLoading model: {HF_REPO}")
        print("This may take a few minutes on first run...")
        
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        model = LFM2AudioModel.from_pretrained(HF_REPO).eval()
        
        # Check for GPU
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            model = model.to(device)
        else:
            device = "cpu"
            print("Using CPU")
        
        print("[OK] Model loaded successfully")
        
        # Set up inputs for the model
        chat = ChatState(processor)
        
        # System prompt for interleaved generation
        chat.new_turn("system")
        chat.add_text("Respond with interleaved text and audio.")
        chat.end_turn()
        
        print("\n[Example] Multi-turn chat setup complete!")
        print("\nTo use with audio input:")
        print("  chat.new_turn('user')")
        print("  wav, sampling_rate = torchaudio.load('your_audio.wav')")
        print("  chat.add_audio(wav, sampling_rate)")
        print("  chat.end_turn()")
        print("\n  chat.new_turn('assistant')")
        print("  for t in model.generate_interleaved(**chat, max_new_tokens=512):")
        print("      if t.numel() == 1:")
        print("          print(processor.text.decode(t), end='', flush=True)")
        print("      else:")
        print("          # Handle audio tokens")
        
        return model, processor, chat
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. liquid-audio is installed: pip install liquid-audio")
        print("2. You have Python 3.12+")
        print("3. You have sufficient disk space for model download (~3-6GB)")
        return None, None, None

def demo_asr():
    """Demo: Automatic Speech Recognition (ASR)"""
    print("\n" + "="*60)
    print("ASR (Automatic Speech Recognition) Example")
    print("="*60)
    
    try:
        import torch
        import torchaudio
        from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
        
        # Load models
        HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
        processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
        model = LFM2AudioModel.from_pretrained(HF_REPO).eval()
        
        if torch.cuda.is_available():
            model = model.to("cuda")
        
        # Set up for ASR
        chat = ChatState(processor)
        
        chat.new_turn("system")
        chat.add_text("Perform ASR.")
        chat.end_turn()
        
        print("\n[Example] ASR setup complete!")
        print("\nTo use with audio file:")
        print("  chat.new_turn('user')")
        print("  wav, sampling_rate = torchaudio.load('audio.wav')")
        print("  chat.add_audio(wav, sampling_rate)")
        print("  chat.end_turn()")
        print("\n  chat.new_turn('assistant')")
        print("  for t in model.generate_sequential(**chat, max_new_tokens=512):")
        print("      if t.numel() == 1:")
        print("          print(processor.text.decode(t), end='', flush=True)")
        
    except Exception as e:
        print(f"\nError: {e}")

def main():
    print("="*60)
    print("Liquid AI LFM2-Audio-1.5B - Latest Model Runner")
    print("="*60)
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    
    if not deps_ok:
        print("\n" + "="*60)
        print("Installation Instructions")
        print("="*60)
        print("\n1. Install liquid-audio:")
        print("   pip install liquid-audio")
        print("   pip install 'liquid-audio[demo]'  # for demo interface")
        print("\n2. Optional: Install flash-attn for better performance:")
        print("   pip install flash-attn --no-build-isolation")
        print("\n3. Make sure you have Python 3.12+")
        print("   Check with: python --version")
        return
    
    print("\n" + "="*60)
    print("Available Demos")
    print("="*60)
    print("\n1. Gradio Demo Interface")
    print("2. Multi-turn, Multi-modal Chat")
    print("3. ASR (Automatic Speech Recognition)")
    
    print("\n" + "="*60)
    print("Quick Start")
    print("="*60)
    print("\nOption 1: Launch Gradio Demo")
    print("  liquid-audio-demo")
    print("  Then open http://localhost:7860 in your browser")
    
    print("\nOption 2: Use in Python Code")
    print("""
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

# Load models
HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
model = LFM2AudioModel.from_pretrained(HF_REPO).eval()

# For interleaved generation (speech-to-speech)
chat = ChatState(processor)
chat.new_turn("system")
chat.add_text("Respond with interleaved text and audio.")
chat.end_turn()

# Add user audio input
chat.new_turn("user")
# wav, sr = torchaudio.load("audio.wav")
# chat.add_audio(wav, sr)
chat.end_turn()

# Generate response
chat.new_turn("assistant")
for t in model.generate_interleaved(**chat, max_new_tokens=512, 
                                    audio_temperature=1.0, audio_top_k=4):
    if t.numel() == 1:
        print(processor.text.decode(t), end="", flush=True)
    else:
        # Handle audio tokens (8 codebooks)
        pass
""")
    
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    print("\nModel: LFM2-Audio-1.5B")
    print("Repository: LiquidAI/LFM2-Audio-1.5B")
    print("Hugging Face: https://huggingface.co/LiquidAI/LFM2-Audio-1.5B")
    print("GitHub: https://github.com/Liquid4All/liquid-audio")
    print("\nFeatures:")
    print("- Real-time speech-to-speech conversations")
    print("- Low-latency audio processing")
    print("- Interleaved generation (text + audio)")
    print("- Sequential generation (ASR/TTS)")
    print("- 1.5B parameters, optimized for edge devices")
    
    print("\n" + "="*60)
    print("Generation Modes")
    print("="*60)
    print("\n1. Interleaved Generation:")
    print("   - For real-time speech-to-speech")
    print("   - System prompt: 'Respond with interleaved text and audio.'")
    print("   - Method: model.generate_interleaved()")
    print("   - Outputs text and audio tokens in fixed pattern")
    
    print("\n2. Sequential Generation:")
    print("   - For ASR or TTS tasks")
    print("   - System prompt: 'Perform ASR.' (for ASR)")
    print("   - Method: model.generate_sequential()")
    print("   - Model decides when to switch modalities")
    
    # Run demos
    print("\n" + "="*60)
    print("Running Examples")
    print("="*60)
    
    demo_gradio()
    
    model, processor, chat = demo_multi_turn_chat()
    
    if model is not None:
        demo_asr()
    
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("\n1. Try the Gradio demo: liquid-audio-demo")
    print("2. Check the official documentation:")
    print("   https://huggingface.co/LiquidAI/LFM2-Audio-1.5B")
    print("   https://github.com/Liquid4All/liquid-audio")
    print("3. See example code in this script above")
    print("="*60)

if __name__ == "__main__":
    main()

