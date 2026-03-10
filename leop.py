import os
import re
import csv
import json
import uuid
import shutil
import sqlite3
import zipfile
import asyncio
import subprocess
import shlex
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# ==============================================================================
# CONFIGURATION & SYSTEM SETUP
# ==============================================================================

APP_TITLE = "GenomeOps Workbench"
APP_VERSION = "1.2.2"  # fixed Prokka PATH issue

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
LOG_DIR = DATA_DIR / "logs"
WORKSPACE_DIR = DATA_DIR / "workspace"
DB_PATH = DATA_DIR / "app.db"

for d in [DATA_DIR, UPLOAD_DIR, RESULT_DIR, LOG_DIR, WORKSPACE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

APP_PASSWORD = os.getenv("APP_PASSWORD", "changeme")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "10000"))

# Global store for running subprocesses (job_id -> Popen)
running_jobs: Dict[str, subprocess.Popen] = {}

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# ------------------------------------------------------------------------------
# System preparation – run once at startup (ignore warnings)
# ------------------------------------------------------------------------------
def setup_system():
    """Install basic build tools (skip universe if not available)."""
    try:
        subprocess.run(
            "apt-get update && apt-get install -y wget git python3-pip build-essential ncbi-blast+",
            shell=True,
            check=True,
            capture_output=True,
            timeout=120
        )
        print("System setup completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"System setup warning: {e.stderr.decode()}")
    except Exception as e:
        print(f"System setup exception: {e}")

setup_system()

# ==============================================================================
# TOOL CATALOG (with robust install commands)
# ==============================================================================

TOOLS = {
    # ---------- QC ----------
    "fastqc": {
        "name": "FastQC",
        "category": "QC",
        "description": "Quality control for FASTQ files",
        "install_command": "apt-get update && apt-get install -y fastqc",
        "version_command": "fastqc --version",
        "parameters": [
            {"name": "input", "label": "FASTQ file", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "Raw sequencing reads"}
        ]
    },
    "multiqc": {
        "name": "MultiQC",
        "category": "QC",
        "description": "Aggregate QC reports from a directory",
        "install_command": "pip install multiqc",
        "version_command": "multiqc --version",
        "parameters": [
            {"name": "input_dir", "label": "Directory (choose any file inside it)", "type": "file", "required": True,
             "extensions": [], "help": "Any file inside the target directory; the parent folder will be used"}
        ]
    },
    "fastp": {
        "name": "fastp",
        "category": "QC",
        "description": "Fast all-in-one preprocessor for FASTQ data",
        "install_command": "apt-get update && apt-get install -y fastp",
        "version_command": "fastp --version",
        "parameters": [
            {"name": "in1", "label": "Read 1 FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "Forward reads"},
            {"name": "in2", "label": "Read 2 FASTQ (optional for PE)", "type": "file", "required": False,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "Reverse reads for paired-end"},
            {"name": "out1", "label": "Output clean read 1", "type": "text", "default": "clean_R1.fastq.gz"},
            {"name": "out2", "label": "Output clean read 2", "type": "text", "default": "clean_R2.fastq.gz"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4},
            {"name": "qualified_quality_phred", "label": "Qualified quality (Q)", "type": "number", "default": 15},
            {"name": "length_required", "label": "Minimum read length", "type": "number", "default": 30}
        ]
    },
    "trimmomatic": {
        "name": "Trimmomatic",
        "category": "QC",
        "description": "Flexible read trimming",
        "install_command": "apt-get update && apt-get install -y trimmomatic",
        "version_command": "trimmomatic -version",
        "parameters": [
            {"name": "r1", "label": "Read 1 FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Forward reads"},
            {"name": "r2", "label": "Read 2 FASTQ (optional for PE)", "type": "file", "required": False,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Reverse reads for paired-end"},
            {"name": "adapter", "label": "Adapter FASTA", "type": "file", "required": False,
             "extensions": [".fa", ".fasta"], "help": "Adapter sequences"},
            {"name": "leading", "label": "Leading quality", "type": "number", "default": 3},
            {"name": "trailing", "label": "Trailing quality", "type": "number", "default": 3},
            {"name": "minlen", "label": "Minimum length", "type": "number", "default": 36}
        ]
    },
    "nanoplot": {
        "name": "NanoPlot",
        "category": "QC",
        "description": "QC for Nanopore reads",
        "install_command": "pip install nanoplot",
        "version_command": "NanoPlot --version",
        "parameters": [
            {"name": "reads", "label": "Nanopore FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Long reads for QC"},
            {"name": "outdir", "label": "Output directory", "type": "text", "default": "nanoplot_out"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4}
        ]
    },

    # ---------- Assembly ----------
    "spades": {
        "name": "SPAdes",
        "category": "Assembly",
        "description": "Paired-end genome assembly",
        "install_command": "apt-get update && apt-get install -y spades",
        "version_command": "spades.py --version",
        "parameters": [
            {"name": "r1", "label": "Read 1 (FASTQ)", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "Forward reads"},
            {"name": "r2", "label": "Read 2 (FASTQ)", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "Reverse reads"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4},
            {"name": "memory", "label": "Memory limit (GB)", "type": "number", "default": 16}
        ]
    },
    "canu": {
        "name": "Canu",
        "category": "Assembly",
        "description": "Long-read assembler (PacBio/ONT)",
        "install_command": "apt-get update && apt-get install -y canu",
        "version_command": "canu --version",
        "parameters": [
            {"name": "input", "label": "Long-read FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".fastq.gz", ".fq.gz"], "help": "PacBio or Nanopore reads"},
            {"name": "genomeSize", "label": "Estimated genome size (e.g. 4.8m)", "type": "text", "required": True,
             "help": "e.g. 4.8m for 4.8 Mbp"},
            {"name": "useGrid", "label": "Use grid (false for local run)", "type": "flag", "default": False},
            {"name": "maxThreads", "label": "Max threads", "type": "number", "default": 8}
        ]
    },
    "flye": {
        "name": "Flye",
        "category": "Assembly",
        "description": "Long-read assembler (viral genomes)",
        "install_command": "pip install flye",
        "version_command": "flye --version",
        "parameters": [
            {"name": "reads", "label": "Long reads FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Nanopore/PacBio reads"},
            {"name": "genome-size", "label": "Estimated genome size", "type": "text", "required": True,
             "help": "e.g. 5m for 5 Mbp"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4},
            {"name": "out-dir", "label": "Output directory", "type": "text", "default": "flye_out"}
        ]
    },
    "quast": {
        "name": "QUAST",
        "category": "Assembly",
        "description": "Quality assessment of genome assemblies",
        "install_command": "apt-get update && apt-get install -y python3-pip && pip3 install quast",
        "version_command": "quast --version",
        "parameters": [
            {"name": "input", "label": "Assembly FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fasta"], "help": "Assembly to evaluate"},
            {"name": "reference", "label": "Reference genome (optional)", "type": "file", "required": False,
             "extensions": [".fa", ".fasta"], "help": "Reference for comparative metrics"},
            {"name": "genes", "label": "Gene coordinates (GFF)", "type": "file", "required": False,
             "extensions": [".gff", ".gff3"], "help": "Annotation to evaluate gene content"}
        ]
    },

    # ---------- Annotation ----------
    "prokka": {
        "name": "Prokka",
        "category": "Annotation",
        "description": "Rapid prokaryotic genome annotation",
        # Install: remove conda version, use GitHub master, ensure correct PATH
        "install_command": (
            "apt-get update && apt-get install -y git perl bioperl ncbi-blast+ && "
            "rm -f /opt/conda/bin/prokka || true && "
            "git clone https://github.com/tseemann/prokka.git /opt/prokka && "
            "cd /opt/prokka && /opt/prokka/bin/prokka --setupdb && "
            "ln -sf /opt/prokka/bin/prokka /usr/local/bin/prokka && "
            "ln -sf /usr/bin/blastp /usr/local/bin/blastp || true"
        ),
        "version_command": "prokka --version",
        "parameters": [
            {"name": "input", "label": "Genome FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fna", ".fasta", ".fas"], "help": "Assembly in FASTA format"},
            {"name": "prefix", "label": "Output prefix", "type": "text", "default": "prokka"},
            {"name": "kingdom", "label": "Kingdom", "type": "select",
             "options": ["Bacteria", "Archaea", "Mitochondria", "Viruses"], "default": "Bacteria"},
            {"name": "cpus", "label": "CPUs", "type": "number", "default": 4}
        ]
    },
    "prodigal": {
        "name": "Prodigal",
        "category": "Annotation",
        "description": "Prokaryotic gene finding (works for viruses)",
        "install_command": "apt-get update && apt-get install -y prodigal",
        "version_command": "prodigal -v",
        "parameters": [
            {"name": "input", "label": "Genome FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fasta", ".fna"], "help": "Assembly to predict genes"},
            {"name": "output", "label": "Output GFF file", "type": "text", "default": "genes.gff"},
            {"name": "format", "label": "Output format", "type": "select",
             "options": ["gff", "gbk", "sqn"], "default": "gff"}
        ]
    },

    # ---------- AMR / Virulence ----------
    "abricate": {
        "name": "ABRicate",
        "category": "AMR / Virulence",
        "description": "Mass screening of contigs for antimicrobial resistance genes",
        "install_command": (
            "apt-get update && apt-get install -y git ncbi-blast+ perl && "
            "git clone https://github.com/tseemann/abricate.git /opt/abricate && "
            "/opt/abricate/bin/abricate --setupdb && "
            "ln -sf /opt/abricate/bin/abricate /usr/local/bin/abricate"
        ),
        "version_command": "abricate --version",
        "parameters": [
            {"name": "input", "label": "Genome FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fna", ".fasta", ".fas"], "help": "Assembly to screen"},
            {"name": "db", "label": "Database", "type": "select",
             "options": ["ncbi", "card", "argannot", "ecoh", "plasmidfinder", "vfdb"], "default": "ncbi"},
            {"name": "minid", "label": "Minimum DNA identity (%)", "type": "number", "default": 80},
            {"name": "mincov", "label": "Minimum coverage (%)", "type": "number", "default": 60}
        ]
    },

    # ---------- Typing ----------
    "mlst": {
        "name": "MLST",
        "category": "Typing",
        "description": "Multi-locus sequence typing",
        "install_command": "apt-get update && apt-get install -y mlst",
        "version_command": "mlst --version",
        "parameters": [
            {"name": "input", "label": "Genome FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fna", ".fasta", ".fas"], "help": "Assembly or contigs"}
        ]
    },

    # ---------- Phylogeny ----------
    "iqtree": {
        "name": "IQ-TREE",
        "category": "Phylogeny",
        "description": "Efficient phylogenomic inference",
        "install_command": "wget -O /tmp/iqtree.tar.gz https://github.com/iqtree/iqtree2/releases/download/v2.3.6/iqtree-2.3.6-Linux.tar.gz && tar -xzf /tmp/iqtree.tar.gz -C /usr/local && ln -s /usr/local/iqtree-2.3.6-Linux/bin/iqtree2 /usr/local/bin/iqtree",
        "version_command": "iqtree --version",
        "parameters": [
            {"name": "input", "label": "Alignment file (PHYLIP/FASTA)", "type": "file", "required": True,
             "extensions": [".phy", ".fa", ".fasta"], "help": "Multiple sequence alignment"},
            {"name": "model", "label": "Model", "type": "text", "default": "GTR+G"},
            {"name": "bootstrap", "label": "Bootstrap replicates", "type": "number", "default": 1000},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4}
        ]
    },

    # ---------- Alignment ----------
    "bwa": {
        "name": "BWA",
        "category": "Alignment",
        "description": "Burrows-Wheeler Aligner (index + mem)",
        "install_command": "apt-get update && apt-get install -y bwa",
        "version_command": "bwa 2>&1 | head -n 2",
        "parameters": [
            {"name": "index", "label": "Reference FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fasta"], "help": "Reference genome (will be indexed if needed)"},
            {"name": "r1", "label": "Read 1 FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Forward reads"},
            {"name": "r2", "label": "Read 2 FASTQ (optional)", "type": "file", "required": False,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Reverse reads for paired-end"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4}
        ]
    },
    "minimap2": {
        "name": "Minimap2",
        "category": "Alignment",
        "description": "Long-read aligner (viral mapping)",
        "install_command": "apt-get update && apt-get install -y minimap2",
        "version_command": "minimap2 --version",
        "parameters": [
            {"name": "target", "label": "Reference genome", "type": "file", "required": True,
             "extensions": [".fa", ".fasta", ".mmi"], "help": "Reference (or index)"},
            {"name": "query", "label": "Reads FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Long reads to align"},
            {"name": "out", "label": "Output SAM", "type": "text", "default": "aln.sam"},
            {"name": "threads", "label": "Threads", "type": "number", "default": 4}
        ]
    },

    # ---------- Utilities ----------
    "samtools": {
        "name": "samtools",
        "category": "Utilities",
        "description": "Utilities for SAM/BAM files",
        "install_command": "apt-get update && apt-get install -y samtools",
        "version_command": "samtools --version",
        "parameters": [
            {"name": "command", "label": "Command (e.g., flagstat, sort)", "type": "text", "required": True,
             "help": "Subcommand to run"},
            {"name": "input", "label": "BAM/SAM file", "type": "file", "required": True,
             "extensions": [".bam", ".sam"], "help": "Alignment file"}
        ]
    },
    "bcftools": {
        "name": "bcftools",
        "category": "Utilities",
        "description": "Tools for VCF/BCF files",
        "install_command": "apt-get update && apt-get install -y bcftools",
        "version_command": "bcftools --version",
        "parameters": [
            {"name": "command", "label": "Command (e.g., view, stats, query)", "type": "text", "required": True,
             "help": "bcftools subcommand"},
            {"name": "input", "label": "VCF/BCF file", "type": "file", "required": True,
             "extensions": [".vcf", ".vcf.gz", ".bcf"], "help": "Variant file"}
        ]
    },
    "bedtools": {
        "name": "bedtools",
        "category": "Utilities",
        "description": "Genome arithmetic (intersect, merge, etc.)",
        "install_command": "apt-get update && apt-get install -y bedtools",
        "version_command": "bedtools --version",
        "parameters": [
            {"name": "command", "label": "Subcommand (e.g., intersect, merge)", "type": "text", "required": True,
             "help": "bedtools subcommand"},
            {"name": "a", "label": "File A (BED/GFF/VCF)", "type": "file", "required": True,
             "extensions": [".bed", ".gff", ".vcf"], "help": "First interval file"},
            {"name": "b", "label": "File B", "type": "file", "required": True,
             "extensions": [".bed", ".gff", ".vcf"], "help": "Second interval file"}
        ]
    },
    "seqkit": {
        "name": "seqkit",
        "category": "Utilities",
        "description": "FASTA/FASTQ manipulation toolkit",
        "install_command": "apt-get update && apt-get install -y seqkit",
        "version_command": "seqkit version",
        "parameters": [
            {"name": "command", "label": "Subcommand (e.g., stats, grep)", "type": "text", "required": True,
             "help": "seqkit subcommand"},
            {"name": "input", "label": "FASTA/FASTQ file", "type": "file", "required": True,
             "extensions": [".fa", ".fasta", ".fq", ".fastq", ".gz"], "help": "Sequence file"}
        ]
    },

    # ---------- Virus-specific tools ----------
    "diamond": {
        "name": "DIAMOND",
        "category": "Virus / Alignment",
        "description": "Protein sequence aligner (for viral protein searches)",
        "install_command": "apt-get update && apt-get install -y diamond-aligner",
        "version_command": "diamond --version",
        "parameters": [
            {"name": "query", "label": "Protein FASTA", "type": "file", "required": True,
             "extensions": [".faa", ".fa"], "help": "Protein sequences to search"},
            {"name": "db", "label": "Database (pre-formatted .dmnd)", "type": "file", "required": True,
             "extensions": [".dmnd"], "help": "DIAMOND database file"},
            {"name": "out", "label": "Output file", "type": "text", "default": "matches.tsv"},
            {"name": "evalue", "label": "E-value threshold", "type": "number", "default": 0.001}
        ]
    },
    "hmmer": {
        "name": "HMMER",
        "category": "Virus / Profile Search",
        "description": "Profile hidden Markov models for viral protein families",
        "install_command": "apt-get update && apt-get install -y hmmer",
        "version_command": "hmmscan -h",
        "parameters": [
            {"name": "hmm", "label": "HMM profile", "type": "file", "required": True,
             "extensions": [".hmm"], "help": "Profile HMM database"},
            {"name": "seq", "label": "Protein FASTA", "type": "file", "required": True,
             "extensions": [".faa", ".fa"], "help": "Sequences to search against profile"},
            {"name": "out", "label": "Output file", "type": "text", "default": "hits.txt"}
        ]
    },
    "infernal": {
        "name": "Infernal",
        "category": "Virus / RNA Search",
        "description": "RNA structure search (viral ncRNAs)",
        "install_command": "apt-get update && apt-get install -y infernal",
        "version_command": "cmsearch -h",
        "parameters": [
            {"name": "cm", "label": "Covariance model", "type": "file", "required": True,
             "extensions": [".cm"], "help": "RNA family model"},
            {"name": "seq", "label": "Nucleotide FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fasta", ".fna"], "help": "Genome or contigs to search"},
            {"name": "out", "label": "Output file", "type": "text", "default": "results.txt"}
        ]
    },
    "racon": {
        "name": "Racon",
        "category": "Virus / Polishing",
        "description": "Polisher for long-read assemblies",
        "install_command": "apt-get update && apt-get install -y racon",
        "version_command": "racon --version",
        "parameters": [
            {"name": "reads", "label": "Long reads FASTQ", "type": "file", "required": True,
             "extensions": [".fastq", ".fq", ".gz"], "help": "Reads used for assembly"},
            {"name": "overlaps", "label": "Overlaps (PAF)", "type": "file", "required": True,
             "extensions": [".paf"], "help": "Overlap mapping from minimap2"},
            {"name": "target", "label": "Assembly FASTA", "type": "file", "required": True,
             "extensions": [".fa", ".fasta"], "help": "Draft assembly to polish"},
            {"name": "out", "label": "Polished assembly", "type": "text", "default": "polished.fa"}
        ]
    }
}

# ==============================================================================
# DATABASE (SQLite)
# ==============================================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()

    # Files table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            original_name TEXT,
            saved_name TEXT,
            path TEXT,
            extension TEXT,
            size_bytes INTEGER,
            uploaded_at TEXT
        )
    """)

    # Jobs table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            job_type TEXT,
            title TEXT,
            command TEXT,
            status TEXT,
            created_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            log_file TEXT,
            result_dir TEXT,
            stdout TEXT,
            stderr TEXT,
            returncode INTEGER
        )
    """)

    # Tools table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tools (
            tool_key TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            description TEXT,
            install_command TEXT,
            version_command TEXT,
            installed INTEGER DEFAULT 0,
            install_job_id TEXT,
            parameters TEXT
        )
    """)

    # Add version_command column if missing (for upgrades)
    cur.execute("PRAGMA table_info(tools)")
    columns = [col[1] for col in cur.fetchall()]
    if "version_command" not in columns:
        cur.execute("ALTER TABLE tools ADD COLUMN version_command TEXT")

    # Upsert tools from catalog
    for key, t in TOOLS.items():
        cur.execute("""
            INSERT OR REPLACE INTO tools
            (tool_key, name, category, description, install_command, version_command, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (key, t["name"], t["category"], t["description"],
              t["install_command"], t.get("version_command", ""),
              json.dumps(t["parameters"])))

    conn.commit()
    conn.close()

init_db()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or f"file_{uuid.uuid4().hex[:8]}"

def run_shell(command: str, cwd: Optional[Path] = None, timeout: int = 300) -> dict:
    """Run a shell command and return result dict."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": -1,
            "stdout": e.stdout.decode() if e.stdout else "",
            "stderr": f"Timeout after {timeout}s"
        }
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": str(e)}

def tool_status(tool_key: str) -> dict:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM tools WHERE tool_key = ?", (tool_key,)).fetchone()
    conn.close()
    if not row:
        return {"error": "tool not found"}
    d = dict(row)
    d["parameters"] = json.loads(d["parameters"])
    return d

def all_tool_statuses() -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM tools ORDER BY category, name").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["parameters"] = json.loads(d["parameters"])
        out.append(d)
    return out

def insert_file(meta: dict):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO files (file_id, original_name, saved_name, path, extension, size_bytes, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (meta["file_id"], meta["original_name"], meta["saved_name"], meta["path"],
          meta["extension"], meta["size_bytes"], meta["uploaded_at"]))
    conn.commit()
    conn.close()

def get_files() -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM files ORDER BY uploaded_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_file(file_id: str) -> Optional[dict]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def create_job(job_type: str, title: str, command: str = "") -> str:
    job_id = uuid.uuid4().hex
    log_file = LOG_DIR / f"{job_id}.log"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[{now_str()}] Job created: {title}\n")
        if command:
            f.write(f"[{now_str()}] Command: {command}\n")

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO jobs (
            job_id, job_type, title, command, status, created_at, started_at, finished_at,
            log_file, result_dir, stdout, stderr, returncode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (job_id, job_type, title, command, "queued", now_str(), None, None,
          str(log_file), "", "", "", None))
    conn.commit()
    conn.close()
    return job_id

def update_job(job_id: str, **kwargs):
    if not kwargs:
        return
    conn = db()
    cur = conn.cursor()
    cols = []
    vals = []
    for k, v in kwargs.items():
        cols.append(f"{k} = ?")
        vals.append(v)
    vals.append(job_id)
    cur.execute(f"UPDATE jobs SET {', '.join(cols)} WHERE job_id = ?", vals)
    conn.commit()
    conn.close()

def get_jobs() -> List[dict]:
    conn = db()
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_job(job_id: str) -> Optional[dict]:
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def append_log(job_id: str, text: str):
    job = get_job(job_id)
    if not job:
        return
    with open(job["log_file"], "a", encoding="utf-8") as f:
        f.write(text)

def zip_folder(folder: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                full = Path(root) / file
                zf.write(full, arcname=str(full.relative_to(folder)))

def fasta_stats(fasta_path: Path) -> dict:
    seqs = []
    current = []
    with open(fasta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    seqs.append("".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            seqs.append("".join(current))

    if not seqs:
        return {"contigs": 0, "total_length": 0, "largest_contig": 0, "gc_percent": 0.0, "n50": 0}

    lengths = sorted((len(s) for s in seqs), reverse=True)
    total = sum(lengths)
    largest = lengths[0]
    gc = sum(s.upper().count("G") + s.upper().count("C") for s in seqs)
    gc_percent = round((gc / total) * 100, 2) if total else 0.0

    half = total / 2
    running = 0
    n50 = 0
    for l in lengths:
        running += l
        if running >= half:
            n50 = l
            break

    return {
        "contigs": len(seqs),
        "total_length": total,
        "largest_contig": largest,
        "gc_percent": gc_percent,
        "n50": n50
    }

def metadata_summary(csv_path: Path) -> dict:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"n_rows": 0, "columns": []}

    cols = list(rows[0].keys())

    def counts_for(name_candidates: List[str]):
        for c in cols:
            if c.lower() in [x.lower() for x in name_candidates]:
                freq = {}
                for r in rows:
                    v = (r.get(c) or "").strip()
                    if v:
                        freq[v] = freq.get(v, 0) + 1
                return {"column": c, "counts": dict(sorted(freq.items(), key=lambda x: (-x[1], x[0])))}
        return {"column": None, "counts": {}}

    return {
        "n_rows": len(rows),
        "columns": cols,
        "country": counts_for(["country", "location_country"]),
        "host": counts_for(["host"]),
        "source": counts_for(["source", "sample_source", "origin"]),
        "year": counts_for(["year", "collection_year"]),
        "st": counts_for(["st", "mlst", "sequence_type"])
    }

def system_info() -> dict:
    return {
        "title": APP_TITLE,
        "version": APP_VERSION,
        "workspace": str(WORKSPACE_DIR),
        "uploads": str(UPLOAD_DIR),
        "results": str(RESULT_DIR),
        "n_files": len(get_files()),
        "n_jobs": len(get_jobs()),
    }

def write_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ==============================================================================
# RESTRICTED TERMINAL
# ==============================================================================

BLOCKED_PATTERNS = [
    r"\bsudo\b", r"\bapt\b", r"\bapt-get\b", r"\byum\b", r"\bdnf\b", r"\bapk\b",
    r"\bpacman\b", r"\bcurl\b", r"\bwget\b", r"\bssh\b", r"\bscp\b", r"\brm\s+-rf\b",
    r"\bmkfs\b", r"\bmount\b", r"\bumount\b", r"\bshutdown\b", r"\breboot\b",
    r"\buseradd\b", r"\bchmod\s+777\b", r"\bchown\b", r">\s*/",
]

SAFE_CD_PREFIX = str(WORKSPACE_DIR.resolve())

def terminal_allowed(command: str) -> (bool, str):
    cmd = command.strip()
    if not cmd:
        return False, "Empty command."
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, cmd, flags=re.IGNORECASE):
            return False, f"Blocked command pattern: {pat}"
    return True, ""

async def run_terminal_command(command: str, cwd: Path) -> dict:
    ok, reason = terminal_allowed(command)
    if not ok:
        return {"returncode": 1, "stdout": "", "stderr": reason, "cwd": str(cwd)}

    if command.strip().startswith("cd "):
        target = command.strip()[3:].strip()
        new_cwd = (cwd / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
        try:
            new_cwd.relative_to(WORKSPACE_DIR.resolve())
        except Exception:
            return {"returncode": 1, "stdout": "", "stderr": "cd outside workspace is not allowed.", "cwd": str(cwd)}
        if not new_cwd.exists() or not new_cwd.is_dir():
            return {"returncode": 1, "stdout": "", "stderr": "Target directory does not exist.", "cwd": str(cwd)}
        return {"returncode": 0, "stdout": "", "stderr": "", "cwd": str(new_cwd)}

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "HOME": str(WORKSPACE_DIR), "PWD": str(cwd)}
    )
    stdout, stderr = await proc.communicate()
    return {
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="ignore"),
        "stderr": stderr.decode(errors="ignore"),
        "cwd": str(cwd)
    }

# ==============================================================================
# REQUEST MODELS
# ==============================================================================

class PasswordPayload(BaseModel):
    password: str

class AssemblyStatsRequest(BaseModel):
    file_id: str

class MetadataSummaryRequest(BaseModel):
    file_id: str

class ToolInstallRequest(BaseModel):
    password: str
    tool_key: str
    force: bool = False   # allow reinstall

class ToolRunDynamicRequest(BaseModel):
    password: str
    tool_key: str
    values: Dict[str, Any]

class CancelJobRequest(BaseModel):
    password: str

# ==============================================================================
# ASYNC JOB RUNNERS
# ==============================================================================

async def run_job_command(job_id: str, command: str, work_dir: Path, env: Optional[Dict] = None):
    update_job(job_id, status="running", started_at=now_str(), result_dir=str(work_dir))
    append_log(job_id, f"[{now_str()}] Running in {work_dir}\n")

    if env is None:
        env = os.environ.copy()

    loop = asyncio.get_event_loop()
    proc = await loop.run_in_executor(None, lambda: subprocess.Popen(
        command,
        shell=True,
        cwd=str(work_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    ))
    running_jobs[job_id] = proc

    stdout, stderr = await loop.run_in_executor(None, proc.communicate)
    returncode = proc.returncode

    running_jobs.pop(job_id, None)

    (work_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (work_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")

    append_log(job_id, "\n=== STDOUT ===\n")
    append_log(job_id, stdout or "")
    append_log(job_id, "\n=== STDERR ===\n")
    append_log(job_id, stderr or "")

    update_job(
        job_id,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        finished_at=now_str(),
        status="completed" if returncode == 0 else "failed"
    )

async def run_install_job(job_id: str, tool_key: str, install_cmd: str, version_cmd: str):
    work_dir = RESULT_DIR / f"install_{tool_key}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # First, check if tool already exists in PATH
    check_cmd = f"which {tool_key} 2>/dev/null || echo 'not found'"
    check = run_shell(check_cmd)
    if check["returncode"] == 0 and "not found" not in check["stdout"]:
        append_log(job_id, f"[{now_str()}] {tool_key} already present in PATH. Verifying...\n")
        if version_cmd:
            ver = run_shell(version_cmd, work_dir)
            if ver["returncode"] == 0:
                append_log(job_id, f"[{now_str()}] Verification passed: {ver['stdout']}\n")
                conn = db()
                cur = conn.cursor()
                cur.execute("UPDATE tools SET installed = 1, install_job_id = ? WHERE tool_key = ?", (job_id, tool_key))
                conn.commit()
                conn.close()
                update_job(job_id, status="completed", finished_at=now_str(), returncode=0)
                return
            else:
                append_log(job_id, f"[{now_str()}] Verification failed, will attempt installation.\n")
        else:
            conn = db()
            cur = conn.cursor()
            cur.execute("UPDATE tools SET installed = 1, install_job_id = ? WHERE tool_key = ?", (job_id, tool_key))
            conn.commit()
            conn.close()
            update_job(job_id, status="completed", finished_at=now_str(), returncode=0)
            return

    # Run installation
    await run_job_command(job_id, install_cmd, work_dir)

    job = get_job(job_id)
    if job and job["returncode"] == 0 and version_cmd:
        append_log(job_id, f"[{now_str()}] Verifying installation...\n")
        ver = run_shell(version_cmd, work_dir)
        if ver["returncode"] == 0:
            append_log(job_id, f"[{now_str()}] Verification passed: {ver['stdout']}\n")
            conn = db()
            cur = conn.cursor()
            cur.execute("UPDATE tools SET installed = 1, install_job_id = ? WHERE tool_key = ?", (job_id, tool_key))
            conn.commit()
            conn.close()
        else:
            append_log(job_id, f"[{now_str()}] Verification failed: {ver['stderr']}\n")
            conn = db()
            cur = conn.cursor()
            cur.execute("UPDATE tools SET installed = 0, install_job_id = ? WHERE tool_key = ?", (job_id, tool_key))
            conn.commit()
            conn.close()
    elif job and job["returncode"] == 0 and not version_cmd:
        conn = db()
        cur = conn.cursor()
        cur.execute("UPDATE tools SET installed = 1, install_job_id = ? WHERE tool_key = ?", (job_id, tool_key))
        conn.commit()
        conn.close()

async def run_builtin_assembly_stats(job_id: str, fasta_path: Path):
    work_dir = RESULT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    update_job(job_id, status="running", started_at=now_str(), result_dir=str(work_dir))
    append_log(job_id, f"[{now_str()}] Calculating assembly statistics for {fasta_path.name}\n")

    stats = await asyncio.to_thread(fasta_stats, fasta_path)
    write_json(work_dir / "assembly_stats.json", stats)
    (work_dir / "assembly_stats.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in stats.items()),
        encoding="utf-8"
    )
    append_log(job_id, json.dumps(stats, indent=2) + "\n")
    update_job(
        job_id,
        stdout=json.dumps(stats, indent=2),
        stderr="",
        returncode=0,
        finished_at=now_str(),
        status="completed"
    )

async def run_builtin_metadata_summary(job_id: str, csv_path: Path):
    work_dir = RESULT_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    update_job(job_id, status="running", started_at=now_str(), result_dir=str(work_dir))
    append_log(job_id, f"[{now_str()}] Summarizing metadata for {csv_path.name}\n")

    summary = await asyncio.to_thread(metadata_summary, csv_path)
    write_json(work_dir / "metadata_summary.json", summary)
    (work_dir / "metadata_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_log(job_id, json.dumps(summary, indent=2) + "\n")
    update_job(
        job_id,
        stdout=json.dumps(summary, indent=2),
        stderr="",
        returncode=0,
        finished_at=now_str(),
        status="completed"
    )

# ==============================================================================
# API ROUTES
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.get("/health")
async def health():
    return {"status": "ok", "time": now_str()}

@app.get("/api/system")
async def api_system():
    return system_info()

@app.post("/api/auth")
async def api_auth(payload: PasswordPayload):
    return {"ok": payload.password == APP_PASSWORD}

@app.get("/api/tools")
async def api_tools():
    return {"tools": all_tool_statuses()}

@app.post("/api/tools/install")
async def api_install_tool(payload: ToolInstallRequest):
    if payload.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password.")
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT installed, install_job_id, install_command, version_command FROM tools WHERE tool_key = ?", (payload.tool_key,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Tool not found")

    # If not forcing and already installed, return early
    if not payload.force and row["installed"]:
        conn.close()
        return {"message": "Tool already installed", "installed": True}

    # If there's an ongoing installation job, reuse it (unless forcing)
    if not payload.force and row["install_job_id"]:
        job = get_job(row["install_job_id"])
        if job and job["status"] in ("queued", "running"):
            conn.close()
            return {"message": "Installation already in progress", "job_id": row["install_job_id"]}

    # Create a new job
    job_id = create_job("install", f"Install {payload.tool_key}", row["install_command"])
    cur.execute("UPDATE tools SET install_job_id = ? WHERE tool_key = ?", (job_id, payload.tool_key))
    conn.commit()
    conn.close()

    # Reset installed flag to 0 while installing
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE tools SET installed = 0 WHERE tool_key = ?", (payload.tool_key,))
    conn.commit()
    conn.close()

    asyncio.create_task(run_install_job(job_id, payload.tool_key, row["install_command"], row["version_command"] or ""))
    return {"job_id": job_id}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    original_name = file.filename or "upload.bin"
    file_id = uuid.uuid4().hex[:12]
    saved_name = f"{file_id}_{safe_name(original_name)}"
    save_path = UPLOAD_DIR / saved_name

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = {
        "file_id": file_id,
        "original_name": original_name,
        "saved_name": saved_name,
        "path": str(save_path),
        "extension": save_path.suffix.lower(),
        "size_bytes": save_path.stat().st_size,
        "uploaded_at": now_str()
    }
    insert_file(meta)
    return meta

@app.get("/api/files")
async def api_files():
    return {"files": get_files()}

@app.get("/api/files/{file_id}/preview")
async def api_file_preview(file_id: str):
    f = get_file(file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    path = Path(f["path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
            lines = [fp.readline() for _ in range(20)]
        return {"content": "".join(lines), "truncated": True}
    except:
        return {"content": "[Binary file – preview not available]", "truncated": False}

@app.post("/api/run/assembly-stats")
async def api_run_assembly_stats(req: AssemblyStatsRequest):
    f = get_file(req.file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    p = Path(f["path"])
    if p.suffix.lower() not in [".fa", ".fna", ".fasta", ".fas"]:
        raise HTTPException(status_code=400, detail="This utility requires a FASTA file.")

    job_id = create_job("builtin", f"Assembly stats: {p.name}")
    asyncio.create_task(run_builtin_assembly_stats(job_id, p))
    return {"job_id": job_id}

@app.post("/api/run/metadata-summary")
async def api_run_metadata_summary(req: MetadataSummaryRequest):
    f = get_file(req.file_id)
    if not f:
        raise HTTPException(status_code=404, detail="File not found")
    p = Path(f["path"])
    if p.suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="This utility requires a CSV file.")

    job_id = create_job("builtin", f"Metadata summary: {p.name}")
    asyncio.create_task(run_builtin_metadata_summary(job_id, p))
    return {"job_id": job_id}

@app.post("/api/run/tool2")
async def api_run_tool_dynamic(req: ToolRunDynamicRequest):
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password.")
    conn = db()
    cur = conn.cursor()
    row = cur.execute("SELECT * FROM tools WHERE tool_key = ?", (req.tool_key,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Tool not found")
    tool = dict(row)
    if not tool["installed"]:
        raise HTTPException(status_code=400, detail="Tool not installed. Please install it first.")
    parameters = json.loads(tool["parameters"])

    work_dir = RESULT_DIR / f"{req.tool_key}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # For Prokka, use full path to avoid conda version and set PATH to prioritize system blastp
    if req.tool_key == "prokka":
        cmd_parts = ["/usr/local/bin/prokka"]
        # Override PATH to ensure system blastp is found first
        env = os.environ.copy()
        env["PATH"] = "/usr/local/bin:/usr/bin:" + env.get("PATH", "")
    else:
        cmd_parts = [req.tool_key]
        env = None

    for param in parameters:
        pname = param["name"]
        if pname not in req.values:
            if param.get("required", False):
                raise HTTPException(status_code=400, detail=f"Missing required parameter: {pname}")
            continue
        value = req.values[pname]
        if value is None or value == "":
            continue
        if param["type"] == "file":
            file_rec = get_file(value)
            if not file_rec:
                raise HTTPException(status_code=400, detail=f"File not found for parameter {pname}")
            file_path = Path(file_rec["path"])
            if req.tool_key == "multiqc" and pname == "input_dir":
                file_path = file_path.parent
            cmd_parts.append(shlex.quote(str(file_path)))
        elif param["type"] in ("text", "number", "select"):
            cmd_parts.append(f"--{pname}")
            cmd_parts.append(shlex.quote(str(value)))
        elif param["type"] == "flag":
            if value:
                cmd_parts.append(f"--{pname}")

    command = " ".join(cmd_parts)

    job_id = create_job("tool", f"Run {tool['name']}", command)
    update_job(job_id, result_dir=str(work_dir))
    asyncio.create_task(run_job_command(job_id, command, work_dir, env=env))
    return {"job_id": job_id, "command": command}

@app.get("/api/jobs")
async def api_jobs():
    return {"jobs": get_jobs()}

@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_text = ""
    if job["log_file"] and Path(job["log_file"]).exists():
        log_text = Path(job["log_file"]).read_text(encoding="utf-8", errors="ignore")
    job["log"] = log_text
    return job

@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str, req: CancelJobRequest):
    if req.password != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Wrong password.")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    proc = running_jobs.get(job_id)
    if not proc:
        raise HTTPException(status_code=404, detail="Running process not found")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    update_job(job_id, status="cancelled", finished_at=now_str())
    running_jobs.pop(job_id, None)
    append_log(job_id, f"[{now_str()}] Job cancelled by user.\n")
    return {"status": "cancelled"}

@app.get("/api/jobs/{job_id}/download")
async def api_job_download(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result_dir = job.get("result_dir")
    if not result_dir or not Path(result_dir).exists():
        raise HTTPException(status_code=404, detail="No result directory found")

    zip_path = RESULT_DIR / f"{job_id}.zip"
    zip_folder(Path(result_dir), zip_path)
    return FileResponse(path=zip_path, filename=f"{job_id}_results.zip", media_type="application/zip")

@app.websocket("/ws/job/{job_id}")
async def ws_job_log(ws: WebSocket, job_id: str):
    await ws.accept()
    job = get_job(job_id)
    if not job:
        await ws.close(code=1008, reason="Job not found")
        return
    log_path = Path(job["log_file"]) if job["log_file"] else None
    if not log_path or not log_path.exists():
        await ws.close(code=1008, reason="Log file not found")
        return

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        await ws.send_text(f.read())

    last_size = log_path.stat().st_size
    try:
        while True:
            await asyncio.sleep(1)
            current_size = log_path.stat().st_size
            if current_size > last_size:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    new_data = f.read()
                    await ws.send_text(new_data)
                last_size = current_size
            job = get_job(job_id)
            if job["status"] in ("completed", "failed", "cancelled"):
                await ws.close()
                break
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/terminal")
async def ws_terminal(ws: WebSocket):
    await ws.accept()
    state = {
        "authenticated": False,
        "cwd": WORKSPACE_DIR.resolve()
    }
    await ws.send_json({
        "type": "welcome",
        "message": "GenomeOps terminal ready. Authenticate first."
    })

    try:
        while True:
            msg = await ws.receive_json()

            if msg.get("type") == "auth":
                if msg.get("password") == APP_PASSWORD:
                    state["authenticated"] = True
                    await ws.send_json({
                        "type": "auth",
                        "ok": True,
                        "cwd": str(state["cwd"])
                    })
                else:
                    await ws.send_json({"type": "auth", "ok": False, "error": "Wrong password."})
                continue

            if not state["authenticated"]:
                await ws.send_json({"type": "error", "error": "Authenticate first."})
                continue

            if msg.get("type") == "run":
                command = msg.get("command", "")
                result = await run_terminal_command(command, state["cwd"])
                if result.get("cwd"):
                    state["cwd"] = Path(result["cwd"])
                await ws.send_json({
                    "type": "result",
                    "cwd": str(state["cwd"]),
                    "returncode": result["returncode"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"]
                })

    except WebSocketDisconnect:
        return

# ==============================================================================
# FRONTEND (HTML) – Enhanced version (unchanged except version)
# ==============================================================================

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>GenomeOps Workbench v1.2.2</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<style>
:root{
  --bg:#f4f7fb; --card:#fff; --line:#d8e1ec; --text:#213042;
  --blue:#1d4ed8; --green:#0f766e; --orange:#b45309; --red:#b91c1c; --muted:#64748b;
  --hover-shadow:0 8px 20px rgba(0,0,0,0.08);
}
*{box-sizing:border-box}
body{margin:0;font-family:'Inter',Arial,Helvetica,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
.header{padding:20px 24px;background:linear-gradient(135deg,#17324d,#1d4ed8);color:#fff}
.header h1{margin:0;font-size:30px;font-weight:600}
.header p{margin:6px 0 0 0;opacity:0.9}
.wrap{max-width:1600px;margin:16px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:1.2fr .9fr;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:18px;box-shadow:0 4px 16px rgba(0,0,0,.05);transition:box-shadow 0.2s}
.card:hover{box-shadow:var(--hover-shadow)}
h2{margin:0 0 12px 0;font-size:1.4rem;font-weight:600}
.small{font-size:12px;color:var(--muted)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
button{border:none;padding:9px 14px;border-radius:10px;color:#fff;background:var(--blue);cursor:pointer;font-weight:600;transition:transform 0.1s, filter 0.1s;display:inline-flex;align-items:center;gap:6px}
button:hover{filter:brightness(0.95);transform:scale(1.02)}
button:active{transform:scale(0.98)}
button.green{background:var(--green)} button.orange{background:var(--orange)} button.red{background:var(--red)} button.gray{background:#64748b}
button:disabled{background:#cbd5e1;cursor:not-allowed;transform:none}
table{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}
th,td{padding:12px 10px;border-bottom:1px solid #e5eaf2;text-align:left;vertical-align:middle;font-size:14px}
th{background:#f8fafc;font-weight:600}
tr:last-child td{border-bottom:none}
input,select,textarea{width:100%;padding:10px;border:1px solid #cbd5e1;border-radius:10px;background:#fff;margin-bottom:10px;font-family:inherit;transition:border 0.2s}
input:focus,select:focus,textarea:focus{outline:none;border-color:var(--blue)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:12px;max-height:450px;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:13px}
#termOut{height:360px;max-height:none}
.tag{display:inline-block;border:1px solid var(--line);padding:2px 8px;border-radius:999px;font-size:12px;margin-right:4px;background:#f8fafc}
.ok{color:#15803d;font-weight:700}.bad{color:#b91c1c;font-weight:700}
.notice{padding:12px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;font-size:13px}
.help-text{font-size:12px;color:#6b7280;margin-top:-8px;margin-bottom:12px;font-style:italic}
fieldset{border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
legend{font-weight:bold;padding:0 8px}
footer{background:#1e293b;color:#cbd5e1;padding:24px;border-radius:16px 16px 0 0;margin-top:32px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
.avatar{width:80px;height:80px;border-radius:50%;background:#2d3b4f;display:flex;align-items:center;justify-content:center;font-size:40px;color:#94a3b8}
.dev-info{flex:1}
.dev-info a{color:#7ab3ff;text-decoration:none}
.dev-info a:hover{text-decoration:underline}
.modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);align-items:center;justify-content:center;z-index:1000}
.modal-content{background:white;border-radius:16px;padding:24px;max-width:800px;width:90%;max-height:80vh;overflow:auto}
.modal-close{float:right;font-size:24px;cursor:pointer}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,0.3);border-radius:50%;border-top-color:#fff;animation:spin 1s ease-in-out infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.tool-search{margin-bottom:12px;padding:8px;border-radius:20px;border:1px solid var(--line);width:100%}
@media(max-width:1100px){.grid{grid-template-columns:1fr}.row2{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
  <h1><i class="fas fa-dna"></i> GenomeOps Workbench v1.2.2</h1>
  <p>Fixed Prokka – now forces system blastp via PATH override</p>
</div>

<div class="wrap">
  <div class="grid">
    <div>
      <div class="card">
        <h2>1. App status</h2>
        <div class="toolbar">
          <button onclick="loadSystem()"><i class="fas fa-sync-alt"></i> Refresh System</button>
          <button class="gray" onclick="loadTools()"><i class="fas fa-tools"></i> Refresh Tools</button>
          <button class="gray" onclick="loadFiles()"><i class="fas fa-file"></i> Refresh Files</button>
        </div>
        <div id="systemBox"></div>
      </div>

      <div class="card">
        <h2>2. Login</h2>
        <input id="password" type="password" placeholder="App password"/>
        <button onclick="checkPassword()"><i class="fas fa-lock-open"></i> Unlock protected actions</button>
        <div id="authMsg" class="small" style="margin-top:8px;"></div>
      </div>

      <div class="card">
        <h2>3. Tool catalog <i class="fas fa-search" style="margin-left:8px;"></i></h2>
        <input type="text" id="toolSearch" class="tool-search" placeholder="Filter tools by name or category...">
        <div id="toolsArea"></div>
      </div>

      <div class="card">
        <h2>4. Uploads</h2>
        <input id="uploadFile" type="file"/>
        <button class="green" onclick="uploadFile()"><i class="fas fa-upload"></i> Upload</button>
        <div id="uploadMsg" class="small" style="margin-top:8px;"></div>
        <div id="filesArea" style="margin-top:12px;"></div>
      </div>

      <div class="card">
        <h2>5. Built-in analyses</h2>
        <div class="row2">
          <div>
            <div class="small">Assembly statistics from FASTA</div>
            <select id="assemblyFile"></select>
            <button class="orange" onclick="runAssemblyStats()"><i class="fas fa-chart-bar"></i> Run</button>
          </div>
          <div>
            <div class="small">Metadata summary from CSV</div>
            <select id="metaFile"></select>
            <button class="orange" onclick="runMetadataSummary()"><i class="fas fa-table"></i> Run</button>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>6. Install & run external tools</h2>
        <select id="toolSelect" onchange="onToolSelect()"></select>
        <div id="toolInstallStatus" class="small"></div>
        <button class="green" id="installBtn" onclick="installToolDirect(selectedTool.tool_key, true)" style="display:none;"><i class="fas fa-download"></i> Install / Reinstall</button>
        <div id="toolFormContainer" style="margin-top:12px;"></div>
        <button class="green" id="runToolBtn" onclick="runDynamicTool()" style="display:none;"><i class="fas fa-play"></i> Run tool</button>
        <div id="toolRunMsg" class="small"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>7. Jobs and results</h2>
        <button onclick="loadJobs()"><i class="fas fa-sync-alt"></i> Refresh Jobs</button>
        <div id="jobsArea" style="margin-top:12px;"></div>
      </div>

      <div class="card">
        <h2>8. Job log <span id="liveIndicator" class="small" style="display:none;">(live)</span></h2>
        <div id="selectedJobLabel" class="small">No job selected.</div>
        <pre id="jobLog">No log loaded.</pre>
      </div>

      <div class="card">
        <h2>9. Linux workspace console</h2>
        <div class="small">Restricted workspace shell. No sudo, package managers, downloads, or destructive commands.</div>
        <div class="row2">
          <div><input id="termCmd" placeholder="e.g. ls -lah"/></div>
          <div><button onclick="termRun()"><i class="fas fa-terminal"></i> Run command</button></div>
        </div>
        <div class="small" id="termCwd"></div>
        <pre id="termOut"></pre>
      </div>
    </div>
  </div>

  <footer>
    <div class="avatar"><i class="fas fa-user-circle"></i></div>
    <div class="dev-info">
      <h3>Nahiduzzaman</h3>
      <p>URA, Department of Microbiology and Hygiene, Bangladesh Agricultural University</p>
      <p>
        <i class="fas fa-globe"></i> <a href="https://sites.google.com/view/nahiduzzaman-bau/home" target="_blank">sites.google.com/view/nahiduzzaman-bau</a><br>
        <i class="fas fa-envelope"></i> <a href="mailto:nahiduzzaman.2001055@bau.edu.bd">nahiduzzaman.2001055@bau.edu.bd</a>
      </p>
      <p><small>Version 1.2.2 – GenomeOps Workbench</small></p>
    </div>
  </footer>
</div>

<!-- File preview modal -->
<div id="previewModal" class="modal">
  <div class="modal-content">
    <span class="modal-close" onclick="closePreview()">&times;</span>
    <h3>File preview</h3>
    <pre id="previewContent" style="white-space:pre-wrap;"></pre>
  </div>
</div>

<script>
let APP_OK = false;
let currentJobId = null;
let tools = [];
let files = [];
let selectedTool = null;
let termSocket = null;
let termReady = false;
let jobLogSocket = null;

async function api(url, method="GET", body=null){
  const opts = {method, headers:{}};
  if(body){opts.headers["Content-Type"]="application/json"; opts.body=JSON.stringify(body);}
  const res = await fetch(url, opts);
  let data = await res.json();
  if(!res.ok){ throw new Error(data.detail || data.error || "Request failed");}
  return data;
}

async function loadSystem(){
  const d = await api("/api/system");
  document.getElementById("systemBox").innerHTML = `
    <div class="small"><b>Version:</b> ${d.version}</div>
    <div class="small"><b>Workspace:</b> ${d.workspace}</div>
    <div class="small"><b>Uploads:</b> ${d.uploads}</div>
    <div class="small"><b>Results:</b> ${d.results}</div>
    <div class="small"><b>Files:</b> ${d.n_files} | <b>Jobs:</b> ${d.n_jobs}</div>
  `;
}

async function checkPassword(){
  try{
    const password = document.getElementById("password").value;
    const d = await api("/api/auth", "POST", {password});
    APP_OK = !!d.ok;
    document.getElementById("authMsg").innerText = APP_OK ? "Unlocked." : "Wrong password.";
    connectTerminal();
  }catch(e){
    document.getElementById("authMsg").innerText = e.message;
  }
}

function filterTools(){
  const query = document.getElementById("toolSearch").value.toLowerCase();
  const cats = {};
  for (const t of tools){
    if(t.name.toLowerCase().includes(query) || t.category.toLowerCase().includes(query) || t.description.toLowerCase().includes(query)){
      if(!cats[t.category]) cats[t.category] = [];
      cats[t.category].push(t);
    }
  }
  let html = "";
  for (const cat of Object.keys(cats).sort()){
    html += `<h3>${cat}</h3><table><thead><tr><th>Tool</th><th>Description</th><th>Status</th><th>Action</th></tr></thead><tbody>`;
    for (const t of cats[cat]){
      html += `
        <tr>
          <td><b>${t.name}</b></td>
          <td>${t.description}</td>
          <td class="${t.installed ? 'ok' : 'bad'}">${t.installed ? 'Installed' : 'Not installed'}</td>
          <td>${t.installed ? '' : `<button class="gray" onclick="installToolDirect('${t.tool_key}', true)"><i class="fas fa-download"></i> Install</button>`}</td>
        </tr>
      `;
    }
    html += "</tbody></table>";
  }
  document.getElementById("toolsArea").innerHTML = html || "<p>No tools match your filter.</p>";
}

async function loadTools(){
  const d = await api("/api/tools");
  tools = d.tools;
  filterTools();

  const sel = document.getElementById("toolSelect");
  sel.innerHTML = '<option value="">-- select tool --</option>';
  for(const t of tools){
    const opt = document.createElement("option");
    opt.value = t.tool_key;
    opt.textContent = `${t.name} | ${t.category}`;
    sel.appendChild(opt);
  }
}

document.getElementById("toolSearch").addEventListener("input", filterTools);

async function installToolDirect(toolKey, force = false){
  if(!APP_OK){ alert("Unlock protected actions first."); return; }
  const password = document.getElementById("password").value;
  try{
    const d = await api("/api/tools/install", "POST", {password, tool_key: toolKey, force});
    if(d.job_id){
      alert(`Installation started. Job ID: ${d.job_id}`);
      loadJobs();
    } else {
      alert(d.message);
    }
    loadTools();
  }catch(e){
    alert(e.message);
  }
}

async function onToolSelect(){
  const key = document.getElementById("toolSelect").value;
  if(!key) return;
  selectedTool = tools.find(t => t.tool_key === key);
  if(!selectedTool) return;
  document.getElementById("toolInstallStatus").innerText = selectedTool.installed ? "Installed" : "Not installed";
  document.getElementById("installBtn").style.display = "inline-block"; // always show for reinstall
  if(selectedTool.installed){
    renderToolForm(selectedTool);
    document.getElementById("runToolBtn").style.display = "inline-block";
  } else {
    document.getElementById("toolFormContainer").innerHTML = "";
    document.getElementById("runToolBtn").style.display = "none";
  }
}

function renderToolForm(tool){
  let html = '<h3>Parameters</h3>';
  for (let p of tool.parameters){
    if(p.type === 'file'){
      html += `<div><label><b>${p.label}</b> ${p.required ? '*' : ''}</label>`;
      if(p.help) html += `<div class="help-text">${p.help}</div>`;
      html += `<select id="param_${p.name}" data-required="${p.required}">`;
      html += '<option value="">-- select file --</option>';
      for (let f of files){
        if(p.extensions && p.extensions.length > 0){
          const extMatch = p.extensions.some(ext => f.original_name.toLowerCase().endsWith(ext.toLowerCase()));
          if(!extMatch) continue;
        }
        html += `<option value="${f.file_id}">${f.original_name}</option>`;
      }
      html += '</select></div>';
    } else if(p.type === 'select'){
      html += `<div><label><b>${p.label}</b> ${p.required ? '*' : ''}</label>`;
      if(p.help) html += `<div class="help-text">${p.help}</div>`;
      html += `<select id="param_${p.name}" data-required="${p.required}">`;
      for (let opt of p.options){
        html += `<option value="${opt}" ${opt === p.default ? 'selected' : ''}>${opt}</option>`;
      }
      html += '</select></div>';
    } else if(p.type === 'number'){
      html += `<div><label><b>${p.label}</b> ${p.required ? '*' : ''}</label>`;
      if(p.help) html += `<div class="help-text">${p.help}</div>`;
      html += `<input type="number" id="param_${p.name}" value="${p.default || ''}" data-required="${p.required}"></div>`;
    } else if(p.type === 'text'){
      html += `<div><label><b>${p.label}</b> ${p.required ? '*' : ''}</label>`;
      if(p.help) html += `<div class="help-text">${p.help}</div>`;
      html += `<input type="text" id="param_${p.name}" value="${p.default || ''}" data-required="${p.required}"></div>`;
    } else if(p.type === 'flag'){
      html += `<div><label><input type="checkbox" id="param_${p.name}" ${p.default ? 'checked' : ''}> ${p.label}</label>`;
      if(p.help) html += `<div class="help-text">${p.help}</div>`;
      html += '</div>';
    }
  }
  document.getElementById("toolFormContainer").innerHTML = html;
}

async function runDynamicTool(){
  if(!APP_OK || !selectedTool){ alert("Select a tool and unlock first."); return; }
  const password = document.getElementById("password").value;
  const values = {};
  for (let p of selectedTool.parameters){
    const el = document.getElementById(`param_${p.name}`);
    if(!el) continue;
    if(p.type === 'flag'){
      values[p.name] = el.checked;
    } else {
      const val = el.value;
      if(p.required && !val){
        alert(`Parameter ${p.label} is required.`);
        return;
      }
      values[p.name] = val;
    }
  }
  try{
    const d = await api("/api/run/tool2", "POST", {password, tool_key: selectedTool.tool_key, values});
    alert(`Job submitted: ${d.job_id}`);
    loadJobs();
  }catch(e){
    alert(e.message);
  }
}

async function uploadFile(){
  const input = document.getElementById("uploadFile");
  if(!input.files.length){ alert("Choose a file first."); return; }
  const form = new FormData();
  form.append("file", input.files[0]);
  const res = await fetch("/api/upload", {method:"POST", body:form});
  const d = await res.json();
  document.getElementById("uploadMsg").innerText = `Uploaded: ${d.original_name} (${d.file_id})`;
  input.value = "";
  await loadFiles();
  await loadSystem();
}

async function loadFiles(){
  const d = await api("/api/files");
  files = d.files;

  let html = "<table><thead><tr><th>File ID</th><th>Name</th><th>Type</th><th>Size</th><th>Uploaded</th><th></th></tr></thead><tbody>";
  for (const f of files){
    html += `<tr>
      <td>${f.file_id}</td>
      <td>${f.original_name}</td>
      <td>${f.extension || '-'}</td>
      <td>${f.size_bytes}</td>
      <td>${f.uploaded_at}</td>
      <td><button class="gray" onclick="previewFile('${f.file_id}')"><i class="fas fa-eye"></i></button></td>
    </tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("filesArea").innerHTML = html;

  for (const id of ["assemblyFile","metaFile"]){
    const sel = document.getElementById(id);
    sel.innerHTML = '<option value="">-- select file --</option>';
    for(const f of files){
      const opt = document.createElement("option");
      opt.value = f.file_id;
      opt.textContent = `${f.original_name} [${f.file_id}]`;
      sel.appendChild(opt);
    }
  }
  if(selectedTool && selectedTool.installed){
    renderToolForm(selectedTool);
  }
}

async function previewFile(fileId){
  try{
    const d = await api(`/api/files/${fileId}/preview`);
    document.getElementById("previewContent").innerText = d.content;
    document.getElementById("previewModal").style.display = "flex";
  }catch(e){
    alert("Preview failed: " + e.message);
  }
}

function closePreview(){
  document.getElementById("previewModal").style.display = "none";
}

async function runAssemblyStats(){
  const file_id = document.getElementById("assemblyFile").value;
  if(!file_id){ alert("Select a FASTA file."); return; }
  const d = await api("/api/run/assembly-stats", "POST", {file_id});
  alert("Job submitted: " + d.job_id);
  loadJobs();
}

async function runMetadataSummary(){
  const file_id = document.getElementById("metaFile").value;
  if(!file_id){ alert("Select a CSV file."); return; }
  const d = await api("/api/run/metadata-summary", "POST", {file_id});
  alert("Job submitted: " + d.job_id);
  loadJobs();
}

async function cancelJob(jobId){
  if(!APP_OK){ alert("Unlock first."); return; }
  const password = document.getElementById("password").value;
  try{
    await api(`/api/jobs/${jobId}/cancel`, "POST", {password});
    alert("Job cancelled.");
    loadJobs();
    if(currentJobId === jobId){
      if(jobLogSocket) jobLogSocket.close();
      showLog(jobId);
    }
  }catch(e){
    alert(e.message);
  }
}

async function loadJobs(){
  const d = await api("/api/jobs");
  let html = '<table><thead><tr><th>Job ID</th><th>Title</th><th>Status</th><th>Created</th><th>Action</th></tr></thead><tbody>';
  for(const j of d.jobs){
    html += `<tr>
      <td>${j.job_id}</td>
      <td>${j.title}</td>
      <td><b>${j.status}</b></td>
      <td>${j.created_at}</td>
      <td>
        <button onclick="showLog('${j.job_id}')"><i class="fas fa-file-alt"></i> Log</button>
        ${j.status === 'running' ? `<button class="red" onclick="cancelJob('${j.job_id}')"><i class="fas fa-ban"></i> Cancel</button>` : ''}
        ${j.result_dir ? `<a href="/api/jobs/${j.job_id}/download" target="_blank"><button class="gray"><i class="fas fa-download"></i> Result</button></a>` : ''}
      </td>
    </tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("jobsArea").innerHTML = html;
}

async function showLog(jobId){
  currentJobId = jobId;
  document.getElementById("selectedJobLabel").innerText = `Selected Job: ${jobId}`;
  document.getElementById("jobLog").innerText = "Loading...";
  document.getElementById("liveIndicator").style.display = "inline";
  if(jobLogSocket) jobLogSocket.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  jobLogSocket = new WebSocket(`${proto}://${location.host}/ws/job/${jobId}`);
  jobLogSocket.onmessage = (event) => {
    document.getElementById("jobLog").innerText += event.data;
    const pre = document.getElementById("jobLog");
    pre.scrollTop = pre.scrollHeight;
  };
  jobLogSocket.onclose = () => {
    document.getElementById("liveIndicator").style.display = "none";
  };
}

function connectTerminal(){
  if(termSocket && termSocket.readyState === WebSocket.OPEN) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  termSocket = new WebSocket(`${proto}://${location.host}/ws/terminal`);

  termSocket.onopen = () => {
    termReady = true;
    document.getElementById("termOut").textContent += "Terminal connected.\n";
    const password = document.getElementById("password").value;
    if(password){
      termSocket.send(JSON.stringify({type:"auth", password}));
    }
  };

  termSocket.onmessage = (event) => {
    const d = JSON.parse(event.data);
    if(d.type === "welcome"){
      document.getElementById("termOut").textContent += d.message + "\n";
    } else if(d.type === "auth"){
      if(d.ok){
        document.getElementById("termOut").textContent += "Terminal authenticated.\n";
        document.getElementById("termCwd").innerText = "cwd: " + d.cwd;
      } else {
        document.getElementById("termOut").textContent += "Authentication failed.\n";
      }
    } else if(d.type === "result"){
      document.getElementById("termCwd").innerText = "cwd: " + d.cwd;
      if(d.stdout) document.getElementById("termOut").textContent += d.stdout + "\n";
      if(d.stderr) document.getElementById("termOut").textContent += d.stderr + "\n";
      document.getElementById("termOut").textContent += `[exit=${d.returncode}]\n`;
    } else if(d.type === "error"){
      document.getElementById("termOut").textContent += d.error + "\n";
    }
    const pre = document.getElementById("termOut");
    pre.scrollTop = pre.scrollHeight;
  };

  termSocket.onclose = () => {
    termReady = false;
    document.getElementById("termOut").textContent += "Terminal disconnected.\n";
  };
}

function termRun(){
  if(!termReady || !termSocket || termSocket.readyState !== WebSocket.OPEN){
    connectTerminal();
    // Wait a moment for connection
    setTimeout(() => {
      if(termSocket && termSocket.readyState === WebSocket.OPEN){
        const cmd = document.getElementById("termCmd").value.trim();
        if(!cmd) return;
        termSocket.send(JSON.stringify({type:"run", command:cmd}));
        document.getElementById("termOut").textContent += `$ ${cmd}\n`;
        document.getElementById("termCmd").value = "";
      } else {
        alert("Terminal not connected. Try again.");
      }
    }, 500);
    return;
  }
  const cmd = document.getElementById("termCmd").value.trim();
  if(!cmd) return;
  termSocket.send(JSON.stringify({type:"run", command:cmd}));
  document.getElementById("termOut").textContent += `$ ${cmd}\n`;
  document.getElementById("termCmd").value = "";
}

window.onload = async function(){
  await loadSystem();
  await loadTools();
  await loadFiles();
  await loadJobs();
  connectTerminal();
};
</script>
</body>
</html>
"""

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)