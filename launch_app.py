"""
DEBATEAI Streamlit App Launcher
================================
One-click launcher for the web application.

Usage: python launch_app.py
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required = ['streamlit', 'plotly', 'pandas']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_ollama():
    """Check if Ollama is running"""
    print("\nChecking Ollama...")
    
    try:
        import ollama
        ollama.list()
        print("  ✅ Ollama is running")
        return True
    except Exception as e:
        print(f"  ❌ Ollama not running: {e}")
        print("\nStart Ollama with:")
        print("  ollama serve")
        return False


def check_data():
    """Check if documents are indexed"""
    print("\nChecking data...")
    
    chunks_file = Path("data/processed/chunks.json")
    if chunks_file.exists():
        print(f"  ✅ Found processed data")
        return True
    else:
        print(f"  ⚠️  No processed data found")
        print("  You'll need to upload documents in the app")
        return True  # Not critical, just a warning


def launch_app():
    """Launch the Streamlit app"""
    print("\n" + "="*60)
    print("🚀 Launching DEBATEAI Web Application...")
    print("="*60)
    
    # Check app.py exists
    if not Path("app.py").exists():
        print("\n❌ Error: app.py not found!")
        print("Make sure you're running this from D:\\FSP\\")
        return
    
    # Launch Streamlit
    print("\n📱 Opening web browser...")
    print("   URL: http://localhost:8501")
    print("\n⚠️  To stop the app, press Ctrl+C in this terminal\n")
    
    # Small delay before opening browser
    time.sleep(2)
    
    # Open browser automatically
    webbrowser.open("http://localhost:8501")
    
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down DEBATEAI...")
        print("="*60)


def main():
    """Main launcher"""
    print("\n" + "="*60)
    print("🤖 DEBATEAI - Web Application Launcher")
    print("="*60)
    
    # Run checks
    deps_ok = check_dependencies()
    ollama_ok = check_ollama()
    check_data()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first!")
        return
    
    if not ollama_ok:
        response = input("\n⚠️  Ollama not running. Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Exiting. Please start Ollama first.")
            return
    
    # Ask to proceed
    print("\n" + "="*60)
    response = input("Ready to launch? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        launch_app()
    else:
        print("\n👋 Launch cancelled.")


if __name__ == "__main__":
    main()
