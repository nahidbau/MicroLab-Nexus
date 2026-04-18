# ============================================================================
# FILE: app.py - Complete PRRSV Genome Intelligence Engine (Local PC version)
# ============================================================================
import os
import uuid
import shutil
import json
from datetime import datetime
from pathlib import Path

# FastAPI
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Celery
from celery import Celery

# QML
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# Other
from pydantic import BaseSettings

# ============================================================================
# Configuration
# ============================================================================
class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./prrsv.db"  # SQLite for local testing
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "local-dev-key-change-me"
    QML_N_QUBITS: int = 4
    QML_N_LAYERS: int = 3

settings = Settings()

# ============================================================================
# Database Setup
# ============================================================================
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")
    step = Column(String(100), default="")
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# Celery Setup
# ============================================================================
celery_app = Celery(__name__, broker=settings.REDIS_URL, backend=settings.REDIS_URL)

# ============================================================================
# Quantum Machine Learning Model
# ============================================================================
class QuantumMutationPredictor(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.n_layers = n_layers
        self.qlayer = qml.qnn.TorchLayer(self._circuit, {"weights": (n_layers, n_qubits)})
        self.out_layer = nn.Linear(n_qubits, 1)

    def _circuit(self, inputs, weights):
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        x = x.float()
        quantum_out = self.qlayer(x)
        return torch.sigmoid(self.out_layer(quantum_out))

qml_model = QuantumMutationPredictor(
    n_qubits=settings.QML_N_QUBITS,
    n_layers=settings.QML_N_LAYERS
)

def predict_mutation_impact(assembly_fasta_path):
    """Extract features from assembly and run QML prediction."""
    # Mock feature extraction: in reality, you'd parse the FASTA, compute k-mers, etc.
    # Here we generate a random 4D vector for demo
    features = torch.randn(1, settings.QML_N_QUBITS)
    with torch.no_grad():
        score = qml_model(features).item()
    return {
        "escape_potential": round(score, 4),
        "interpretation": "High risk" if score > 0.5 else "Low risk",
        "quantum_circuit": f"{settings.QML_N_QUBITS} qubits, {settings.QML_N_LAYERS} layers"
    }

# ============================================================================
# Bioinformatics Mock Functions (replace with real tools later)
# ============================================================================
def run_fastqc(fastq_path):
    return {"quality": "good", "gc_percent": 52.3, "total_reads": 125000}

def run_trimmomatic(fastq_path):
    trimmed = fastq_path.replace(".fastq", "_trimmed.fastq")
    Path(trimmed).touch()
    return trimmed

def run_spades(trimmed_fastq):
    assembly = trimmed_fastq.replace("_trimmed.fastq", "_assembly.fasta")
    with open(assembly, "w") as f:
        f.write(">PRRSV_contig\nATCGATCGATCGATCG\n")
    return assembly

def run_prodigal(assembly_fasta):
    return {"genes": ["ORF1a", "ORF1b", "ORF2", "ORF3", "ORF4", "ORF5", "ORF6", "ORF7"]}

# ============================================================================
# Celery Task
# ============================================================================
@celery_app.task(bind=True)
def run_prrsv_pipeline(self, job_id: int, file_path: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise ValueError("Job not found")

        job.status = "running"
        db.commit()

        job.step = "FastQC quality assessment"
        db.commit()
        qc = run_fastqc(file_path)

        job.step = "Trimming adapters"
        db.commit()
        trimmed = run_trimmomatic(file_path)

        job.step = "Genome assembly with SPAdes"
        db.commit()
        assembly = run_spades(trimmed)

        job.step = "Gene annotation with Prodigal"
        db.commit()
        annotation = run_prodigal(assembly)

        job.step = "Quantum ML mutation prediction"
        db.commit()
        qml_result = predict_mutation_impact(assembly)

        job.status = "completed"
        job.result = {
            "fastqc": qc,
            "trimmed_file": trimmed,
            "assembly_file": assembly,
            "annotation": annotation,
            "qml": qml_result
        }
        db.commit()
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.commit()
        raise
    finally:
        db.close()
    return {"job_id": job_id, "status": "completed"}

# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(title="PRRSV Genome Intelligence Engine")
templates = Jinja2Templates(directory="templates")
os.makedirs("uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Create the HTML template file
with open("templates/index.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/daisyui@4.12.10/dist/full.css" rel="stylesheet">
    <script src="https://unpkg.com/htmx.org@2.0.1"></script>
    <title>PRRSV Analyzer</title>
</head>
<body class="bg-base-200">
    <div class="navbar bg-primary text-primary-content shadow-lg">
        <div class="flex-1">
            <a class="btn btn-ghost normal-case text-xl">🧬 PRRSV Genome Intelligence Engine</a>
        </div>
    </div>
    <div class="container mx-auto p-4">
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title">Upload PRRSV Reads (FASTQ)</h2>
                <form hx-post="/api/upload" hx-target="#results" hx-indicator="#spinner" enctype="multipart/form-data">
                    <input type="file" name="file" accept=".fastq,.fastq.gz" class="file-input file-input-bordered w-full" required>
                    <button type="submit" class="btn btn-secondary mt-4">Run Full Pipeline + QML</button>
                    <span id="spinner" class="loading loading-spinner loading-md htmx-indicator ml-2"></span>
                </form>
            </div>
        </div>
        <div id="results" class="mt-8"></div>
    </div>
</body>
</html>
    """)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file
    file_id = str(uuid.uuid4())
    original_name = file.filename
    safe_name = f"{file_id}_{original_name}"
    file_path = os.path.join("uploads", safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create job
    job = Job(filename=original_name, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue task
    run_prrsv_pipeline.delay(job.id, file_path)

    # Return HTML fragment for HTMX
    return HTMLResponse(f"""
    <div class="card bg-base-100 shadow-xl mt-4" id="job-{job.id}">
        <div class="card-body">
            <h2 class="card-title">Job #{job.id}: {original_name}</h2>
            <div hx-get="/api/job/{job.id}/status" hx-trigger="load delay:1s" hx-swap="outerHTML"></div>
        </div>
    </div>
    """)

@app.get("/api/job/{job_id}/status")
async def job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return HTMLResponse("Job not found", status_code=404)

    if job.status == "completed":
        return HTMLResponse(f"""
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title text-success">✅ Job #{job.id} Completed</h2>
                <div class="overflow-x-auto">
                    <pre class="bg-gray-100 p-2 rounded">{json.dumps(job.result, indent=2)}</pre>
                </div>
            </div>
        </div>
        """)
    elif job.status == "failed":
        return HTMLResponse(f"""
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title text-error">❌ Job #{job.id} Failed</h2>
                <p class="text-error">{job.error}</p>
            </div>
        </div>
        """)
    else:
        return HTMLResponse(f"""
        <div class="card bg-base-100 shadow-xl">
            <div class="card-body">
                <h2 class="card-title">⏳ Job #{job.id} - {job.status}</h2>
                <progress class="progress progress-primary w-full" value="50" max="100"></progress>
                <p>Current step: {job.step or "initializing..."}</p>
                <div hx-get="/api/job/{job_id}/status" hx-trigger="load delay:2s" hx-swap="outerHTML"></div>
            </div>
        </div>
        """)

@app.get("/api/jobs")
async def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(10).all()
    return [
        {
            "id": j.id,
            "filename": j.filename,
            "status": j.status,
            "step": j.step,
            "created_at": j.created_at.isoformat()
        }
        for j in jobs
    ]

# ============================================================================
# Run locally: uvicorn app:app --reload
# Celery: celery -A app worker --loglevel=info
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)