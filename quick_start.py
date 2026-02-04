"""
DEBATEAI - Quick Start Verification
====================================
Run this script to verify everything is set up correctly.

Usage: python quick_start.py
"""

import sys
from pathlib import Path

def check_file_structure():
    """Check if all required files exist"""
    print("\n" + "=" * 60)
    print("Checking File Structure")
    print("=" * 60)
    
    required_files = {
        'core/interfaces.py': 'Base interfaces',
        'core/orchestrator.py': 'Debate orchestrator',
        'core/hybrid_retriever.py': 'Hybrid search',
        'core/document_processor.py': 'Document processing',
        'retrievers/faiss_retriever.py': 'FAISS search',
        'retrievers/bm25_retriever.py': 'BM25 search',
        'agents/pro.py': 'Pro agent',
        'agents/con.py': 'Con agent',
        'agents/judge.py': 'Judge agent',
        'agents/reporter.py': 'Reporter agent',
        'config/agents_config.yaml': 'Agent config',
        'cli.py': 'Command-line interface',
    }
    
    missing = []
    found = []
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            print(f"✅ {file_path}")
            found.append(file_path)
        else:
            print(f"❌ {file_path} - MISSING!")
            missing.append(file_path)
    
    print(f"\n📊 Status: {len(found)}/{len(required_files)} files found")
    
    if missing:
        print("\n⚠️  Missing files:")
        for f in missing:
            print(f"   - {f}")
        return False
    else:
        print("\n✅ All required files present!")
        return True


def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 60)
    print("Checking Dependencies")
    print("=" * 60)
    
    required_packages = [
        ('llama_index', 'llama-index'),
        ('faiss', 'faiss-cpu or faiss-gpu'),
        ('rank_bm25', 'rank-bm25'),
        ('sentence_transformers', 'sentence-transformers'),
        ('ollama', 'ollama'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('pypdf', 'pypdf'),
        ('yaml', 'pyyaml'),
    ]
    
    missing = []
    
    for package, install_name in required_packages:
        try:
            __import__(package)
            print(f"✅ {install_name}")
        except ImportError:
            print(f"❌ {install_name} - NOT INSTALLED!")
            missing.append(install_name)
    
    if missing:
        print("\n⚠️  Missing packages. Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_ollama():
    """Check if Ollama is running and models are available"""
    print("\n" + "=" * 60)
    print("Checking Ollama")
    print("=" * 60)
    
    try:
        import ollama
        
        # Check connection
        try:
            models = ollama.list()
            print("✅ Ollama is running")
            
            # Check required models
            required_models = [
                'llama3.1:8b',
                'mistral:7b',
                'phi3:medium',
                'phi3:mini'
            ]
            
            available_models = [m['name'] for m in models.get('models', [])]
            
            print(f"\n📦 Available models ({len(available_models)}):")
            for model in available_models[:10]:  # Show first 10
                print(f"   - {model}")
            
            missing_models = []
            for model in required_models:
                if not any(model in m for m in available_models):
                    missing_models.append(model)
            
            if missing_models:
                print(f"\n⚠️  Missing models:")
                for model in missing_models:
                    print(f"   - {model}")
                print(f"\nDownload with: ollama pull {missing_models[0]}")
                return False
            else:
                print(f"\n✅ All required models available!")
                return True
                
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("\nStart Ollama with: ollama serve")
            return False
            
    except ImportError:
        print("❌ Ollama Python package not installed")
        print("Install with: pip install ollama")
        return False


def check_gpu():
    """Check if GPU is available"""
    print("\n" + "=" * 60)
    print("Checking GPU")
    print("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA Available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print(f"⚠️  GPU not available, will use CPU")
            print(f"   (This is fine, just slower)")
            return True
    except ImportError:
        print(f"⚠️  PyTorch not installed, cannot check GPU")
        return True


def check_data():
    """Check if data folders exist"""
    print("\n" + "=" * 60)
    print("Checking Data Folders")
    print("=" * 60)
    
    data_folders = ['data/raw', 'data/processed', 'outputs']
    
    for folder in data_folders:
        path = Path(folder)
        if path.exists():
            files = list(path.glob('*'))
            print(f"✅ {folder}/ ({len(files)} files)")
        else:
            print(f"⚠️  {folder}/ - creating...")
            path.mkdir(parents=True, exist_ok=True)
    
    # Check if raw data exists
    raw_files = list(Path('data/raw').rglob('*.*'))
    if raw_files:
        print(f"\n📄 Found {len(raw_files)} files in data/raw/")
        print("   Ready to process!")
    else:
        print(f"\n⚠️  No files in data/raw/")
        print("   Add PDFs, CSVs, or TXT files to get started")
    
    return True


def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("🤖 DEBATEAI - QUICK START VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("GPU", check_gpu),
        ("Data", check_data),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou're ready to run DEBATEAI!")
        print("\nNext steps:")
        print("  1. Add documents to data/raw/")
        print("  2. Run: python cli.py")
        print("  3. Choose option 1 to index documents")
        print("  4. Choose option 2 to run your first debate!")
    else:
        print("\n" + "=" * 60)
        print("⚠️  SOME CHECKS FAILED")
        print("=" * 60)
        print("\nFix the issues above and run this script again.")
        print("See INTEGRATION_GUIDE.md for detailed instructions.")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
