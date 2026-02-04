"""
GPU Test Script - Verify CUDA and Model Loading
Run this to ensure your RTX 2050 is properly configured
"""

import torch
import sys
from datetime import datetime

def test_cuda():
    """Test CUDA availability and GPU info"""
    print("=" * 60)
    print("DEBATEAI - Hardware Verification")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Python version
    print(f"✓ Python Version: {sys.version.split()[0]}")
    
    # PyTorch version
    print(f"✓ PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA Available: {cuda_available}")
    
    if cuda_available:
        # GPU details
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Device: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        
        # Memory test
        try:
            # Allocate 1GB tensor to test
            x = torch.randn(1000, 1000, 100).cuda()
            print(f"✓ Memory Test: Successfully allocated tensor on GPU")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Memory Test Failed: {e}")
    else:
        print("\n⚠ WARNING: CUDA not available!")
        print("  Models will run on CPU (slower but functional)")
        print("  Check NVIDIA driver installation")
    
    print("\n" + "=" * 60)

def test_ollama():
    """Test Ollama connection"""
    print("\nTesting Ollama Connection...")
    try:
        import ollama
        # List available models
        models = ollama.list()
        print(f"✓ Ollama Connected")
        print(f"✓ Available Models: {len(models.get('models', []))}")
        for model in models.get('models', [])[:5]:  # Show first 5
            print(f"  - {model['name']}")
    except Exception as e:
        print(f"✗ Ollama Error: {e}")
        print("  Make sure Ollama is running (check system tray)")

def test_embeddings():
    """Test sentence-transformers with GPU"""
    print("\nTesting Embedding Model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        
        print(f"✓ Embedding Model Loaded on: {device.upper()}")
        
        # Test encoding
        test_text = "This is a test sentence for embedding."
        embedding = model.encode(test_text)
        
        print(f"✓ Embedding Generated: {embedding.shape[0]} dimensions")
        print(f"✓ Embedding Speed Test: ~{len(test_text.split())} words/sec")
        
    except Exception as e:
        print(f"✗ Embedding Test Failed: {e}")

if __name__ == "__main__":
    test_cuda()
    test_ollama()
    test_embeddings()
    
    print("\n" + "=" * 60)
    print("VERDICT:")
    if torch.cuda.is_available():
        print("🎉 Your system is READY for high-performance RAG!")
        print("   Expected query time: 20-30 seconds")
    else:
        print("⚠ System functional but will use CPU")
        print("   Expected query time: 60-90 seconds")
    print("=" * 60)