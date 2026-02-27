import sys
import os
import subprocess

# Set the path to your Python 3.10
PYTHON_PATH = r"C:\Users\nahid_vv0xche\AppData\Local\Programs\Python\Python310"
PYTHON_EXE = os.path.join(PYTHON_PATH, "python.exe")

def check_packages():
    """Check if required packages are installed"""
    packages = ['biopython', 'scipy', 'scikit-learn', 'networkx', 
                'plotly', 'streamlit', 'pandas', 'numpy']
    
    print("Checking installed packages...")
    try:
        import pkg_resources
        for package in packages:
            try:
                version = pkg_resources.get_distribution(package).version
                print(f"✅ {package}: {version}")
            except pkg_resources.DistributionNotFound:
                print(f"❌ {package}: NOT INSTALLED")
    except:
        # Fallback method
        import importlib
        for package in packages:
            spec = importlib.util.find_spec(package)
            if spec:
                print(f"✅ {package}: Found")
            else:
                print(f"❌ {package}: NOT FOUND")

def main():
    print("=" * 60)
    print("Quantum Intelligent Viral Genomics Suite Launcher")
    print("=" * 60)
    
    # Check current Python
    print(f"\nCurrent Python: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Check packages
    check_packages()
    
    print("\n" + "=" * 60)
    print("Starting Streamlit App...")
    print("=" * 60)
    
    # Run streamlit with the correct Python
    streamlit_cmd = [PYTHON_EXE, "-m", "streamlit", "run", "script.py"]
    
    print(f"\nCommand: {' '.join(streamlit_cmd)}")
    
    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print("\n\nApp stopped by user")
    except Exception as e:
        print(f"\nError starting app: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()