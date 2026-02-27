#!/bin/bash

echo "Creating virtual environment..."
python -m venv q_maple_env
source q_maple_env/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing core scientific stack..."
pip install numpy scipy scikit-learn matplotlib pandas

echo "Installing quantum machine learning libraries..."
pip install pennylane
pip install qiskit
pip install qiskit-machine-learning

echo "Installing optional accelerators..."
pip install networkx tqdm

echo "Installation complete."
echo "Activate environment with: source q_maple_env/bin/activate"
