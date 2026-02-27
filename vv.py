import sys
import subprocess

# ============================================================
# AUTO INSTALL REQUIRED PACKAGES FOR
# Quantum Intelligent Viral Genomics Suite
# ============================================================

REQUIRED_PACKAGES = [
    # Core
    "streamlit",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "plotly",

    # Bioinformatics
    "biopython",

    # Machine learning
    "scikit-learn",

    # Network & graph
    "networkx",

    # Scientific utilities
    "statsmodels",

    # Plotly image export
    "kaleido",

    # Optional but useful
    "tqdm",
    "joblib"
]

def install(package):
    try:
        print(f"📦 Installing {package} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            stdout=subprocess.DEVNULL
        )
        print(f"✅ Installed {package}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")

if __name__ == "__main__":
    print("🚀 Installing all dependencies...\n")

    # Upgrade pip first
    subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    for pkg in REQUIRED_PACKAGES:
        install(pkg)

    print("\n🎉 All packages installation attempted.")
    print("👉 If any failed, run again or install manually.")
