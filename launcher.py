import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'tweepy', 'pandas', 'numpy', 'plotly',
        'textblob', 'wordcloud', 'matplotlib', 'seaborn',
        'Pillow', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Setup the environment for the dashboard"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    
    # Initialize TextBlob if needed
    try:
        import textblob
        print("✅ TextBlob is ready")
    except Exception as e:
        print(f"⚠️ TextBlob setup issue: {e}")
    
    print("✅ Environment setup complete")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Social Media Sentiment Analysis Dashboard...")
    print("📱 The dashboard will open in your default browser")
    print("🛑 Press Ctrl+C to stop the application")
    
    try:
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("=" * 60)
    print("📊 Social Media Sentiment Analysis Dashboard")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Setup environment
    setup_environment()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()