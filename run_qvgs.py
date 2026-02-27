#!/usr/bin/env python3
"""
Quantum Intelligent Viral Genomics Suite - Launcher
This script ensures the app runs with the correct Python environment
"""

import sys
import os
import subprocess
import importlib.util


def find_correct_python():
    """Find the Python installation with all required packages"""
    # Try the known path where packages are installed
    known_paths = [
        r"C:\Users\nahid_vv0xche\AppData\Local\Programs\Python\Python310\python.exe",
        r"C:\Python310\python.exe",
        r"C:\Program Files\Python310\python.exe",
        sys.executable  # Current Python
    ]

    for python_path in known_paths:
        if os.path.exists(python_path):
            print(f"Found Python at: {python_path}")

            # Check if this Python has BioPython
            try:
                result = subprocess.run(
                    [python_path, "-c", "from Bio import SeqIO; print('BioPython: OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "BioPython: OK" in result.stdout:
                    print(f"✅ This Python has BioPython installed")
                    return python_path
                else:
                    print(f"❌ This Python doesn't have BioPython")
            except:
                continue

    return None


def install_missing_packages(python_path):
    """Install required packages"""
    print("\n" + "=" * 60)
    print("Installing missing packages...")
    print("=" * 60)

    packages = [
        "biopython",
        "scipy",
        "scikit-learn",
        "networkx",
        "plotly",
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn"
    ]

    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run(
                [python_path, "-m", "pip", "install", package],
                check=True
            )
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

    print("\n" + "=" * 60)
    print("All packages installed!")
    print("=" * 60)


def run_streamlit(python_path):
    """Run the Streamlit app"""
    print("\n" + "=" * 60)
    print("Starting Quantum Intelligent Viral Genomics Suite")
    print("=" * 60)

    # Check if streamlit is available
    try:
        result = subprocess.run(
            [python_path, "-c", "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except:
        print("Streamlit not found, installing...")
        subprocess.run([python_path, "-m", "pip", "install", "streamlit"])

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "script.py")

    if not os.path.exists(script_path):
        print(f"❌ Error: Cannot find script.py at {script_path}")
        return

    print(f"Running script: {script_path}")

    # Run streamlit
    try:
        subprocess.run([
            python_path,
            "-m",
            "streamlit",
            "run",
            script_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\nApp stopped by user")
    except Exception as e:
        print(f"\nError running app: {e}")


def main():
    print("\n" + "=" * 60)
    print("QUANTUM INTELLIGENT VIRAL GENOMICS SUITE v6.0")
    print("=" * 60)

    # Find correct Python
    python_path = find_correct_python()

    if not python_path:
        print("❌ Could not find a Python with BioPython installed")
        response = input("\nDo you want to install BioPython and other packages? (y/n): ")
        if response.lower() == 'y':
            # Use current Python
            python_path = sys.executable
            install_missing_packages(python_path)
        else:
            print("Exiting...")
            return

    # Verify all packages are installed
    print("\nVerifying required packages...")
    required_packages = ["biopython", "scipy", "scikit-learn", "networkx", "plotly", "pandas", "numpy"]

    for package in required_packages:
        try:
            subprocess.run(
                [python_path, "-c", f"import {package.split('-')[0]}; print(f'✅ {package}: OK')"],
                capture_output=True,
                text=True
            )
        except:
            print(f"❌ {package}: Missing")
            response = input(f"\nPackage {package} is missing. Install it now? (y/n): ")
            if response.lower() == 'y':
                install_missing_packages(python_path)
                break

    # Run the app
    run_streamlit(python_path)


if __name__ == "__main__":
    main()