"""
Quick test script for LFM2-Audio-1.5B
This script verifies the model can be loaded and shows basic usage
"""

import sys
import torch

print("="*60)
print("LFM2-Audio-1.5B Quick Test")
print("="*60)

# Check dependencies
print("\n[1/3] Checking dependencies...")
try:
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
    print("[OK] liquid-audio imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import liquid-audio: {e}")
    sys.exit(1)

print(f"[OK] PyTorch {torch.__version__}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\n[2/3] Loading model (this will download on first run)...")
print("Model: LiquidAI/LFM2-Audio-1.5B")
print("This may take several minutes and requires ~3-6GB download...")

try:
    HF_REPO = "LiquidAI/LFM2-Audio-1.5B"
    
    print("\nLoading processor...")
    # Explicitly set device to CPU if CUDA not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LFM2AudioProcessor.from_pretrained(HF_REPO, device=device).eval()
    print(f"[OK] Processor loaded on {device.upper()}")
    
    print("\nLoading model...")
    # Explicitly set device to CPU if CUDA not available
    model = LFM2AudioModel.from_pretrained(HF_REPO, device=device).eval()
    print(f"[OK] Model loaded on {device.upper()}")
    
    print("[OK] Model loaded successfully!")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    print("\nFull error traceback:")
    traceback.print_exc()
    print("\nPossible issues:")
    print("1. Internet connection required for first download")
    print("2. Sufficient disk space needed (~10GB)")
    print("3. Check Hugging Face access")
    print("4. Model may require CUDA-enabled PyTorch (try installing PyTorch with CUDA/ROCm)")
    sys.exit(1)

# Test basic functionality
print("\n[3/3] Testing basic functionality...")

try:
    # Create a simple chat state
    chat = ChatState(processor)
    
    # Set up system prompt for interleaved generation
    chat.new_turn("system")
    chat.add_text("Respond with interleaved text and audio.")
    chat.end_turn()
    
    print("[OK] ChatState created successfully")
    print("[OK] Model is ready to use!")
    
    print("\n" + "="*60)
    print("SUCCESS! LFM2-Audio-1.5B is working!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use the Gradio demo: python -m liquid_audio.demo")
    print("   Or run: liquid-audio-demo")
    print("2. See run_liquid_audio_newest.py for code examples")
    print("3. Check documentation: https://huggingface.co/LiquidAI/LFM2-Audio-1.5B")
    
except Exception as e:
    print(f"\n[ERROR] Failed to test functionality: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)

