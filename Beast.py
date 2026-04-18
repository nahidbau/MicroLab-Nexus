from __future__ import annotations

import asyncio
import csv
import html
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator


# ============================================================
# Core paths and persistent settings
# ============================================================
APP_TITLE = "BayesPhylo Studio"

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "app_settings.json"

DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", str(BASE_DIR / "runs"))).resolve()
DEFAULT_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

JOB_LOCK = threading.Lock()
JOBS: Dict[str, dict] = {}

app = FastAPI(title=APP_TITLE, version="5.0.0")


# ============================================================
# Settings helpers
# ============================================================
def default_settings() -> dict:
    return {
        "beast_bin": os.environ.get("BEAST_BIN", "").strip(),
        "treeannotator_bin": os.environ.get("TREEANNOTATOR_BIN", "").strip(),
        "muscle_bin": os.environ.get("MUSCLE_BIN", "").strip(),
        "runs_root": str(DEFAULT_RUNS_ROOT),
        "java_opts": os.environ.get("JAVA_OPTS", "").strip(),
    }


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            base = default_settings()
            base.update(data)
            return base
        except Exception:
            return default_settings()
    return default_settings()


def save_settings(data: dict) -> dict:
    base = default_settings()
    base.update(data)
    SETTINGS_FILE.write_text(json.dumps(base, indent=2), encoding="utf-8")
    return base


def get_runs_root() -> Path:
    settings = load_settings()
    root = Path(settings.get("runs_root", str(DEFAULT_RUNS_ROOT))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


# ============================================================
# Models
# ============================================================
class BeastConfig(BaseModel):
    job_name: str = Field(..., min_length=3, max_length=120)
    chain_length: int = Field(1_000_000, ge=10_000, le=500_000_000)
    log_every: int = Field(1_000, ge=100, le=10_000_000)
    screen_every: int = Field(1_000, ge=100, le=10_000_000)
    threads: int = Field(2, ge=1, le=64)

    substitution_model: str = Field("HKY", pattern="^(HKY|GTR)$")
    clock_model: str = Field("strict", pattern="^(strict|relaxed_lognormal)$")
    tree_prior: str = Field(
        "coalescent_constant",
        pattern="^(coalescent_constant|coalescent_skyline|birth_death)$",
    )

    gamma_categories: int = Field(4, ge=1, le=8)
    use_tip_dates: bool = True
    estimate_base_freqs: bool = True
    sample_prior_only: bool = False
    treeannotator_burnin_percent: int = Field(10, ge=0, le=90)

    # Alignment controls
    run_alignment: bool = False
    assume_already_aligned: bool = True

    @field_validator("job_name")
    @classmethod
    def clean_job_name(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v.strip())
        if not v:
            raise ValueError("job_name cannot be empty")
        return v


class AppSettingsModel(BaseModel):
    beast_bin: str = ""
    treeannotator_bin: str = ""
    muscle_bin: str = ""
    runs_root: str = ""
    java_opts: str = ""


# ============================================================
# Utilities
# ============================================================
def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(n)
    idx = 0
    while val >= 1024 and idx < len(units) - 1:
        val /= 1024
        idx += 1
    return f"{val:.1f} {units[idx]}"


def safe_read_text(path: Path, limit: int = 300_000) -> str:
    if not path.exists():
        return ""
    txt = path.read_text(encoding="utf-8", errors="replace")
    return txt[-limit:] if len(txt) > limit else txt


def is_executable_file(path_str: str) -> bool:
    if not path_str:
        return False
    p = Path(path_str)
    return p.exists() and p.is_file()


def resolve_bin(explicit_path: str, candidates: List[str]) -> Optional[str]:
    if explicit_path and is_executable_file(explicit_path):
        return str(Path(explicit_path).resolve())
    for cand in candidates:
        found = shutil.which(cand)
        if found:
            return found
    return None


def resolve_beast_bin() -> Optional[str]:
    settings = load_settings()
    return resolve_bin(settings.get("beast_bin", ""), ["beast", "beast.bat"])


def resolve_treeannotator_bin() -> Optional[str]:
    settings = load_settings()
    return resolve_bin(settings.get("treeannotator_bin", ""), ["treeannotator", "treeannotator.bat"])


def resolve_muscle_bin() -> Optional[str]:
    settings = load_settings()
    return resolve_bin(settings.get("muscle_bin", ""), ["muscle", "muscle.exe", "muscle5"])


def get_java_opts() -> str:
    return load_settings().get("java_opts", "").strip()


# ============================================================
# Persistence helpers
# ============================================================
def persist_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return
    run_dir = Path(job["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job_state.json").write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_job(job_id: str) -> dict:
    with JOB_LOCK:
        if job_id in JOBS:
            return JOBS[job_id]

    runs_root = get_runs_root()
    state_file = runs_root / job_id / "job_state.json"
    if not state_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = json.loads(state_file.read_text(encoding="utf-8"))
    with JOB_LOCK:
        JOBS[job_id] = job
    return job


def set_job(job_id: str, **updates) -> None:
    with JOB_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id].update(updates)
        JOBS[job_id]["updated_at"] = now_ts()
        persist_job(job_id)


def append_event(job_id: str, message: str) -> None:
    job = load_job(job_id)
    line = f"[{iso_now()}] {message}\n"
    event_log = Path(job["paths"]["event_log"])
    with event_log.open("a", encoding="utf-8") as f:
        f.write(line)
    job["events"].append({"time": iso_now(), "message": message})
    persist_job(job_id)


def make_job_dirs(job_id: str, original_ext: str) -> dict:
    runs_root = get_runs_root()
    run_dir = runs_root / job_id
    input_dir = run_dir / "input"
    config_dir = run_dir / "config"
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"
    tree_dir = run_dir / "trees"

    for p in [run_dir, input_dir, config_dir, output_dir, logs_dir, tree_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "config_dir": str(config_dir),
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "tree_dir": str(tree_dir),

        "raw_input": str(input_dir / f"raw_input{original_ext}"),
        "aligned_input": str(input_dir / "aligned_input.fasta"),
        "metadata_csv": str(input_dir / "metadata.csv"),

        "xml": str(config_dir / "analysis.xml"),

        "event_log": str(logs_dir / "events.log"),
        "runner_stdout": str(logs_dir / "runner_stdout.log"),
        "runner_stderr": str(logs_dir / "runner_stderr.log"),
        "alignment_stdout": str(logs_dir / "alignment_stdout.log"),
        "alignment_stderr": str(logs_dir / "alignment_stderr.log"),

        "trace_log": str(output_dir / "analysis.log"),
        "trees": str(output_dir / "analysis.trees"),
        "mcc_tree": str(output_dir / "analysis_mcc.tree"),
        "summary_json": str(output_dir / "summary.json"),
        "summary_txt": str(output_dir / "summary.txt"),
        "pairwise_csv": str(output_dir / "pairwise_distance_matrix.csv"),
        "pairwise_json": str(output_dir / "pairwise_distance_matrix.json"),

        "upgma_nwk": str(tree_dir / "upgma_tree.nwk"),
        "upgma_nexus": str(tree_dir / "upgma_tree.nex"),
        "edited_tree_nwk": str(tree_dir / "edited_tree.nwk"),
        "edited_tree_nexus": str(tree_dir / "edited_tree.nex"),
    }


# ============================================================
# Alignment parsing
# ============================================================
def parse_fasta(text: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_parts = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts).upper().replace(" ", "")))
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)

    if header is not None:
        records.append((header, "".join(seq_parts).upper().replace(" ", "")))

    return records


def parse_nexus(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    in_matrix = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("matrix"):
            in_matrix = True
            continue
        if in_matrix:
            if line.endswith(";"):
                line = line[:-1].strip()
                if line:
                    rows.append(line)
                break
            rows.append(line)

    recs = []
    for row in rows:
        if row.startswith("["):
            continue
        parts = row.split()
        if len(parts) >= 2:
            recs.append((parts[0], "".join(parts[1:]).upper().replace(" ", "")))
    return recs


def load_alignment_records(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    if text.lstrip().startswith(">") or suffix in {".fa", ".fas", ".fasta", ".txt"}:
        records = parse_fasta(text)
    elif "#NEXUS" in text[:500].upper() or suffix in {".nex", ".nexus"}:
        records = parse_nexus(path)
    else:
        raise ValueError("Unsupported sequence format. Use FASTA or NEXUS.")
    if not records:
        raise ValueError("No sequences found in input.")
    return records


def write_fasta(records: List[Tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for taxon, seq in records:
            f.write(f">{taxon}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def validate_taxa_names(records: List[Tuple[str, str]]) -> None:
    names = [t for t, _ in records]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate taxon names detected in alignment.")
    for t in names:
        if not t.strip():
            raise ValueError("Empty taxon name detected.")
        if re.search(r"[(),:;]", t):
            raise ValueError(f"Taxon name contains illegal Newick characters: {t}")


def is_rectangular_alignment(records: List[Tuple[str, str]]) -> bool:
    lengths = {len(seq) for _, seq in records}
    return len(lengths) == 1


def validate_alignment(records: List[Tuple[str, str]]) -> None:
    if len(records) < 3:
        raise ValueError("At least 3 taxa are required.")
    validate_taxa_names(records)

    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise ValueError("All sequences must have the same aligned length.")

    allowed = set("ACGTN?-")
    for taxon, seq in records:
        bad = set(seq) - allowed
        if bad:
            raise ValueError(f"Unsupported characters in sequence '{taxon}': {sorted(bad)}")


def parse_metadata_csv(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.exists() or path.stat().st_size == 0:
        return out

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Metadata CSV is empty.")
        field_map = {x.lower().strip(): x for x in reader.fieldnames}
        if "taxon" not in field_map or "date" not in field_map:
            raise ValueError("Metadata CSV must contain columns: taxon,date")
        taxon_col = field_map["taxon"]
        date_col = field_map["date"]

        for row in reader:
            taxon = row[taxon_col].strip()
            if not taxon:
                continue
            out[taxon] = dict(row)
            out[taxon]["date"] = row[date_col].strip()

    return out


# ============================================================
# MUSCLE alignment
# ============================================================
def build_muscle_command(inp: Path, outp: Path) -> List[str]:
    muscle_bin = resolve_muscle_bin()
    if not muscle_bin:
        raise RuntimeError("MUSCLE executable not found. Set it in Settings or add it to PATH.")
    name = Path(muscle_bin).name.lower()

    # Handle common variants
    if "muscle" in name:
        return [muscle_bin, "-align", str(inp), "-output", str(outp)]
    return [muscle_bin, "-in", str(inp), "-out", str(outp)]


def run_subprocess(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    env = os.environ.copy()
    java_opts = get_java_opts()
    if java_opts:
        env["JAVA_OPTS"] = java_opts

    with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
        out.write(f"\n[{iso_now()}] COMMAND: {' '.join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=out,
            stderr=err,
            text=True,
            env=env,
        )
        return proc.wait()


# ============================================================
# Analysis
# ============================================================
def gc_content(seq: str) -> float:
    seq2 = [b for b in seq if b in {"A", "C", "G", "T"}]
    if not seq2:
        return 0.0
    gc = sum(1 for b in seq2 if b in {"G", "C"})
    return 100.0 * gc / len(seq2)


def base_composition(records: List[Tuple[str, str]]) -> dict:
    counts = {"A": 0, "C": 0, "G": 0, "T": 0, "N": 0, "-": 0, "?": 0}
    for _, seq in records:
        for ch in seq:
            if ch in counts:
                counts[ch] += 1
    total = sum(counts.values()) or 1
    freqs = {k: round(v / total, 6) for k, v in counts.items()}
    return {"counts": counts, "freqs": freqs}


def site_classification(records: List[Tuple[str, str]]) -> dict:
    seqs = [s for _, s in records]
    L = len(seqs[0])

    variable = 0
    singleton = 0
    parsimony_informative = 0
    constant = 0
    ambiguous_sites = 0

    for i in range(L):
        chars = [s[i] for s in seqs]
        if any(c in {"N", "-", "?"} for c in chars):
            ambiguous_sites += 1

        clean = [c for c in chars if c not in {"N", "-", "?"}]
        uniq = set(clean)

        if len(uniq) <= 1:
            constant += 1
            continue

        variable += 1
        counts = Counter(clean)
        repeated = sum(1 for v in counts.values() if v >= 2)
        if repeated >= 2:
            parsimony_informative += 1
        else:
            singleton += 1

    return {
        "constant_sites": constant,
        "variable_sites": variable,
        "singleton_sites": singleton,
        "parsimony_informative_sites": parsimony_informative,
        "sites_with_ambiguity_or_gap": ambiguous_sites,
    }


def pairwise_p_distance(seq1: str, seq2: str) -> float:
    comparable = 0
    diff = 0
    for a, b in zip(seq1, seq2):
        if a in {"N", "-", "?"} or b in {"N", "-", "?"}:
            continue
        comparable += 1
        if a != b:
            diff += 1
    return (diff / comparable) if comparable else 0.0


def pairwise_distance_matrix(records: List[Tuple[str, str]]) -> Tuple[List[str], List[List[float]]]:
    taxa = [t for t, _ in records]
    seqs = {t: s for t, s in records}
    n = len(taxa)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = pairwise_p_distance(seqs[taxa[i]], seqs[taxa[j]])
            mat[i][j] = d
            mat[j][i] = d
    return taxa, mat


def pairwise_pdist_summary_from_matrix(taxa: List[str], mat: List[List[float]]) -> dict:
    vals = []
    n = len(taxa)
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(mat[i][j])

    if not vals:
        return {"min": None, "mean": None, "max": None, "n_pairs": 0, "stdev": None}

    return {
        "min": min(vals),
        "mean": statistics.mean(vals),
        "max": max(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n_pairs": len(vals),
    }


def save_pairwise_matrix(taxa: List[str], mat: List[List[float]], csv_path: Path, json_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["taxon"] + taxa)
        for t, row in zip(taxa, mat):
            writer.writerow([t] + [f"{x:.8f}" for x in row])

    json_path.write_text(
        json.dumps({"taxa": taxa, "matrix": mat}, indent=2),
        encoding="utf-8",
    )


def alignment_summary(records: List[Tuple[str, str]], meta: Dict[str, dict]) -> Tuple[dict, List[str], List[List[float]]]:
    validate_alignment(records)

    taxa = [t for t, _ in records]
    seq_len = len(records[0][1])
    gc_vals = [gc_content(s) for _, s in records]
    site_stats = site_classification(records)
    bc = base_composition(records)
    taxa2, mat = pairwise_distance_matrix(records)
    pd = pairwise_pdist_summary_from_matrix(taxa2, mat)

    matched_rows = sum(1 for t in taxa if t in meta)
    dates = [meta[t]["date"] for t in taxa if t in meta and meta[t].get("date")]

    ambiguity_chars = bc["counts"]["N"] + bc["counts"]["-"] + bc["counts"]["?"]
    total_chars = sum(bc["counts"].values()) or 1

    summary = {
        "taxa_count": len(records),
        "alignment_length": seq_len,
        "gc_mean": round(statistics.mean(gc_vals), 4),
        "gc_min": round(min(gc_vals), 4),
        "gc_max": round(max(gc_vals), 4),
        "ambiguity_characters": ambiguity_chars,
        "ambiguity_fraction": round(ambiguity_chars / total_chars, 6),
        "base_counts": bc["counts"],
        "base_freqs": bc["freqs"],
        **site_stats,
        "pairwise_distance_min": None if pd["min"] is None else round(pd["min"], 6),
        "pairwise_distance_mean": None if pd["mean"] is None else round(pd["mean"], 6),
        "pairwise_distance_max": None if pd["max"] is None else round(pd["max"], 6),
        "pairwise_distance_stdev": None if pd["stdev"] is None else round(pd["stdev"], 6),
        "pairwise_distance_n_pairs": pd["n_pairs"],
        "metadata_rows_matched": matched_rows,
        "metadata_coverage_fraction": round(matched_rows / len(taxa), 6),
        "metadata_date_min": min(dates) if dates else None,
        "metadata_date_max": max(dates) if dates else None,
        "taxa_preview": taxa[:20],
    }
    return summary, taxa2, mat


# ============================================================
# UPGMA tree
# ============================================================
class UPGMANode:
    def __init__(self, name: str, height: float = 0.0, left=None, right=None):
        self.name = name
        self.height = height
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


def upgma_tree_newick(taxa: List[str], mat: List[List[float]]) -> str:
    clusters = {i: [i] for i in range(len(taxa))}
    nodes = {i: UPGMANode(taxa[i], 0.0) for i in range(len(taxa))}
    active = set(range(len(taxa)))
    next_id = len(taxa)

    def avg_dist(c1, c2):
        vals = []
        for i in clusters[c1]:
            for j in clusters[c2]:
                vals.append(mat[i][j])
        return sum(vals) / len(vals)

    while len(active) > 1:
        best = None
        best_pair = None
        active_list = sorted(active)
        for i_idx in range(len(active_list)):
            for j_idx in range(i_idx + 1, len(active_list)):
                a = active_list[i_idx]
                b = active_list[j_idx]
                d = avg_dist(a, b)
                if best is None or d < best:
                    best = d
                    best_pair = (a, b)

        a, b = best_pair
        h = best / 2.0
        new_node = UPGMANode(f"cluster_{next_id}", h, nodes[a], nodes[b])

        clusters[next_id] = clusters[a] + clusters[b]
        nodes[next_id] = new_node
        active.remove(a)
        active.remove(b)
        active.add(next_id)
        next_id += 1

    root_id = next(iter(active))
    root = nodes[root_id]

    def to_newick(node: UPGMANode, parent_height: Optional[float] = None) -> str:
        if node.is_leaf:
            bl = 0.0 if parent_height is None else max(parent_height - node.height, 0.0)
            return f"{node.name}:{bl:.8f}"
        left_str = to_newick(node.left, node.height)
        right_str = to_newick(node.right, node.height)
        bl = "" if parent_height is None else f":{max(parent_height - node.height, 0.0):.8f}"
        return f"({left_str},{right_str}){bl}"

    return to_newick(root) + ";"


def newick_to_nexus(newick: str, title: str = "tree1") -> str:
    return f"""#NEXUS
Begin trees;
    Tree {title} = {newick}
End;
"""


# ============================================================
# Newick parser for browser support and backend validation
# ============================================================
def validate_newick_text(newick: str) -> None:
    s = newick.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    if s.count("(") != s.count(")"):
        raise ValueError("Unbalanced parentheses in Newick.")
    if "(" not in s or ")" not in s:
        raise ValueError("Tree must contain at least one internal node.")


# ============================================================
# Manifest
# ============================================================
def build_manifest(job: dict) -> List[dict]:
    labels = {
        "raw_input": "Raw input",
        "aligned_input": "Aligned input FASTA",
        "metadata_csv": "Metadata CSV",
        "xml": "Generated XML",
        "event_log": "Event log",
        "runner_stdout": "Runner stdout",
        "runner_stderr": "Runner stderr",
        "alignment_stdout": "Alignment stdout",
        "alignment_stderr": "Alignment stderr",
        "trace_log": "BEAST trace log",
        "trees": "Posterior trees",
        "mcc_tree": "Annotated MCC tree",
        "summary_json": "Summary JSON",
        "summary_txt": "Summary text",
        "pairwise_csv": "Pairwise matrix CSV",
        "pairwise_json": "Pairwise matrix JSON",
        "upgma_nwk": "UPGMA tree Newick",
        "upgma_nexus": "UPGMA tree NEXUS",
        "edited_tree_nwk": "Edited tree Newick",
        "edited_tree_nexus": "Edited tree NEXUS",
    }
    files = []
    for key, path_str in job["paths"].items():
        if key in {"run_dir", "input_dir", "config_dir", "output_dir", "logs_dir", "tree_dir"}:
            continue
        p = Path(path_str)
        if p.exists() and p.is_file():
            files.append({
                "key": key,
                "label": labels.get(key, key),
                "filename": p.name,
                "size_bytes": p.stat().st_size,
                "size_human": human_size(p.stat().st_size),
            })
    return files


# ============================================================
# XML generation
# ============================================================
def generate_alignment_block(records: List[Tuple[str, str]]) -> str:
    lines = ['<data id="alignment" spec="Alignment" dataType="nucleotide">']
    for taxon, seq in records:
        t = html.escape(taxon, quote=True)
        s = html.escape(seq, quote=True)
        lines.append(
            f'  <sequence id="seq_{t}" spec="Sequence" taxon="{t}" totalcount="4" value="{s}"/>'
        )
    lines.append("</data>")
    return "\n".join(lines)


def generate_taxon_block(records: List[Tuple[str, str]], meta: Dict[str, dict], use_tip_dates: bool) -> str:
    lines = ['<taxonset id="TaxonSet.alignment" spec="TaxonSet">']
    for taxon, _ in records:
        t = html.escape(taxon, quote=True)
        if use_tip_dates and taxon in meta and meta[taxon].get("date"):
            date_val = html.escape(meta[taxon]["date"], quote=True)
            lines.append(
                f'  <taxon id="{t}" spec="Taxon"><date value="{date_val}" direction="forwards" units="years"/></taxon>'
            )
        else:
            lines.append(f'  <taxon id="{t}" spec="Taxon"/>')
    lines.append("</taxonset>")
    return "\n".join(lines)


def beast_model_bits(cfg: BeastConfig) -> dict:
    if cfg.substitution_model == "HKY":
        site_model = f"""
        <siteModel id="SiteModel.s:alignment" spec="SiteModel" gammaCategoryCount="{cfg.gamma_categories}" shape="@gammaShape.s:alignment" proportionInvariant="0.0">
            <substModel id="hky.s:alignment" spec="HKY" kappa="@kappa.s:alignment">
                <frequencies id="freqs.s:alignment" spec="Frequencies" data="@alignment" estimate="{str(cfg.estimate_base_freqs).lower()}"/>
            </substModel>
        </siteModel>
        """
        params = """
        <parameter id="kappa.s:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">2.0</parameter>
        <parameter id="gammaShape.s:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        """
        ops = """
        <operator id="KappaScaler.s:alignment" spec="ScaleOperator" parameter="@kappa.s:alignment" scaleFactor="0.5" weight="1.0"/>
        <operator id="GammaShapeScaler.s:alignment" spec="ScaleOperator" parameter="@gammaShape.s:alignment" scaleFactor="0.5" weight="1.0"/>
        """
        logs = """
        <log idref="kappa.s:alignment"/>
        <log idref="gammaShape.s:alignment"/>
        """
    else:
        site_model = f"""
        <siteModel id="SiteModel.s:alignment" spec="SiteModel" gammaCategoryCount="{cfg.gamma_categories}" shape="@gammaShape.s:alignment" proportionInvariant="0.0">
            <substModel id="gtr.s:alignment" spec="GTR">
                <parameter id="rateAC.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <parameter id="rateAG.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <parameter id="rateAT.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <parameter id="rateCG.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <parameter id="rateCT.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <parameter id="rateGT.s:alignment" spec="parameter.RealParameter" estimate="true" lower="0.0" name="rates">1.0</parameter>
                <frequencies id="freqs.s:alignment" spec="Frequencies" data="@alignment" estimate="{str(cfg.estimate_base_freqs).lower()}"/>
            </substModel>
        </siteModel>
        """
        params = """
        <parameter id="gammaShape.s:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        """
        ops = """
        <operator id="GammaShapeScaler.s:alignment" spec="ScaleOperator" parameter="@gammaShape.s:alignment" scaleFactor="0.5" weight="1.0"/>
        """
        logs = """
        <log idref="gammaShape.s:alignment"/>
        """

    if cfg.clock_model == "strict":
        clock_model = """
        <branchRateModel id="StrictClock.c:alignment" spec="beast.base.evolution.branchratemodel.StrictClockModel" clock.rate="@clockRate.c:alignment"/>
        """
        clock_params = """
        <parameter id="clockRate.c:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        """
        clock_ops = """
        <operator id="ClockRateScaler.c:alignment" spec="ScaleOperator" parameter="@clockRate.c:alignment" scaleFactor="0.5" weight="3.0"/>
        """
        clock_logs = """
        <log idref="clockRate.c:alignment"/>
        """
        clock_ref = "@StrictClock.c:alignment"
    else:
        clock_model = """
        <branchRateModel id="RelaxedClock.c:alignment" spec="beast.base.evolution.branchratemodel.UCRelaxedClockModel" clock.rate="@clockRate.c:alignment" rateCategories="@rateCategories.c:alignment" tree="@Tree.t:alignment">
            <LogNormal id="LogNormalDistributionModel.c:alignment" S="@ucldStdev.c:alignment" meanInRealSpace="true"/>
        </branchRateModel>
        """
        clock_params = """
        <parameter id="clockRate.c:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        <parameter id="ucldStdev.c:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">0.333</parameter>
        <stateNode id="rateCategories.c:alignment" spec="parameter.IntegerParameter" dimension="0"/>
        """
        clock_ops = """
        <operator id="ClockRateScaler.c:alignment" spec="ScaleOperator" parameter="@clockRate.c:alignment" scaleFactor="0.5" weight="3.0"/>
        <operator id="ucldStdevScaler.c:alignment" spec="ScaleOperator" parameter="@ucldStdev.c:alignment" scaleFactor="0.5" weight="3.0"/>
        """
        clock_logs = """
        <log idref="clockRate.c:alignment"/>
        <log idref="ucldStdev.c:alignment"/>
        """
        clock_ref = "@RelaxedClock.c:alignment"

    if cfg.tree_prior == "coalescent_constant":
        tree_prior_block = """
        <distribution id="CoalescentConstant.t:alignment" spec="Coalescent" tree="@Tree.t:alignment">
            <populationModel id="ConstantPopulation0.t:alignment" spec="ConstantPopulation" popSize="@popSize.t:alignment"/>
        </distribution>
        """
        prior_params = """
        <parameter id="popSize.t:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        """
        prior_ops = """
        <operator id="PopSizeScaler.t:alignment" spec="ScaleOperator" parameter="@popSize.t:alignment" scaleFactor="0.5" weight="3.0"/>
        """
        prior_logs = """
        <log idref="popSize.t:alignment"/>
        """
    elif cfg.tree_prior == "coalescent_skyline":
        tree_prior_block = """
        <distribution id="BayesianSkyline.t:alignment" spec="BayesianSkyline" tree="@Tree.t:alignment" groupSizes="@bPopSizesIndicator.t:alignment" popSizes="@bPopSizes.t:alignment"/>
        """
        prior_params = """
        <parameter id="bPopSizes.t:alignment" spec="parameter.RealParameter" dimension="5" lower="0.0" name="stateNode">1.0</parameter>
        <stateNode id="bPopSizesIndicator.t:alignment" spec="parameter.IntegerParameter" dimension="5">1</stateNode>
        """
        prior_ops = """
        <operator id="bPopSizesScaler.t:alignment" spec="ScaleOperator" parameter="@bPopSizes.t:alignment" scaleFactor="0.5" weight="3.0"/>
        """
        prior_logs = """
        <log idref="bPopSizes.t:alignment"/>
        """
    else:
        tree_prior_block = """
        <distribution id="BirthDeathModel.t:alignment" spec="BirthDeathGernhard08Model" birthDiffRate="@birthRate.t:alignment" relativeDeathRate="@deathRate.t:alignment" sampleProbability="@samplingProportion.t:alignment" tree="@Tree.t:alignment"/>
        """
        prior_params = """
        <parameter id="birthRate.t:alignment" spec="parameter.RealParameter" lower="0.0" name="stateNode">1.0</parameter>
        <parameter id="deathRate.t:alignment" spec="parameter.RealParameter" lower="0.0" upper="1.0" name="stateNode">0.5</parameter>
        <parameter id="samplingProportion.t:alignment" spec="parameter.RealParameter" lower="0.0" upper="1.0" name="stateNode">0.5</parameter>
        """
        prior_ops = """
        <operator id="BirthRateScaler.t:alignment" spec="ScaleOperator" parameter="@birthRate.t:alignment" scaleFactor="0.5" weight="3.0"/>
        <operator id="DeathRateScaler.t:alignment" spec="ScaleOperator" parameter="@deathRate.t:alignment" scaleFactor="0.5" weight="3.0"/>
        <operator id="SamplingPropScaler.t:alignment" spec="ScaleOperator" parameter="@samplingProportion.t:alignment" scaleFactor="0.5" weight="3.0"/>
        """
        prior_logs = """
        <log idref="birthRate.t:alignment"/>
        <log idref="deathRate.t:alignment"/>
        <log idref="samplingProportion.t:alignment"/>
        """

    return {
        "site_model": site_model,
        "clock_model": clock_model,
        "clock_ref": clock_ref,
        "params": params + clock_params + prior_params,
        "ops": ops + clock_ops + prior_ops,
        "logs": logs + clock_logs + prior_logs,
        "tree_prior_block": tree_prior_block,
    }


def build_beast_xml(records: List[Tuple[str, str]], meta: Dict[str, dict], cfg: BeastConfig, xml_path: Path) -> None:
    taxa = [t for t, _ in records]
    if cfg.use_tip_dates:
        missing = [t for t in taxa if t not in meta or not meta[t].get("date")]
        if missing:
            raise ValueError(f"Tip dates missing for taxa: {', '.join(missing[:10])}")

    alignment_block = generate_alignment_block(records)
    taxon_block = generate_taxon_block(records, meta, cfg.use_tip_dates)
    bits = beast_model_bits(cfg)

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<beast version="2.7" namespace="beast.base.core:beast.base.inference:beast.base.evolution.alignment:beast.base.evolution.tree:beast.base.evolution.sitemodel:beast.base.evolution.substitutionmodel:beast.base.evolution.branchratemodel:beast.base.evolution.likelihood:beast.base.evolution.operator:beast.base.evolution.speciation:beast.base.inference.operator:beast.base.inference.distribution">

{alignment_block}

{taxon_block}

<run id="mcmc" spec="MCMC" chainLength="{cfg.chain_length}" sampleFromPrior="{str(cfg.sample_prior_only).lower()}">
    <state id="state" storeEvery="{cfg.log_every}">
        {bits["params"]}
        <tree id="Tree.t:alignment" spec="beast.base.evolution.tree.TreeParser" IsLabelledNewick="false" taxa="@TaxonSet.alignment" newick=""/>
    </state>

    <init id="RandomTree.t:alignment" spec="beast.base.evolution.tree.RandomTree" taxa="@TaxonSet.alignment" estimate="false">
        <populationModel id="ConstantPopulationInit.t:alignment" spec="ConstantPopulation">
            <parameter id="randomPopSize.t:alignment" spec="parameter.RealParameter" name="popSize">1.0</parameter>
        </populationModel>
    </init>

    {bits["site_model"]}

    {bits["clock_model"]}

    <distribution id="posterior" spec="util.CompoundDistribution">
        <distribution id="prior" spec="util.CompoundDistribution">
            {bits["tree_prior_block"]}
        </distribution>
        <distribution id="likelihood" spec="util.CompoundDistribution">
            <distribution id="treeLikelihood.alignment" spec="ThreadedTreeLikelihood" data="@alignment" tree="@Tree.t:alignment" useAmbiguities="false" branchRateModel="{bits["clock_ref"]}" siteModel="@SiteModel.s:alignment"/>
        </distribution>
    </distribution>

    <operator id="treeScaler.t:alignment" spec="ScaleOperator" scaleFactor="0.5" tree="@Tree.t:alignment" weight="3.0"/>
    <operator id="treeRootScaler.t:alignment" spec="ScaleOperator" rootOnly="true" scaleFactor="0.5" tree="@Tree.t:alignment" weight="3.0"/>
    <operator id="uniform.t:alignment" spec="Uniform" tree="@Tree.t:alignment" weight="30.0"/>
    <operator id="subtreeSlide.t:alignment" spec="SubtreeSlide" tree="@Tree.t:alignment" weight="15.0"/>
    <operator id="narrow.t:alignment" spec="Exchange" tree="@Tree.t:alignment" weight="15.0"/>
    <operator id="wide.t:alignment" spec="Exchange" tree="@Tree.t:alignment" isNarrow="false" weight="3.0"/>
    <operator id="wilsonBalding.t:alignment" spec="WilsonBalding" tree="@Tree.t:alignment" weight="3.0"/>
    {bits["ops"]}

    <logger id="screenlog" logEvery="{cfg.screen_every}">
        <log idref="posterior"/>
        <log idref="likelihood"/>
        <log idref="prior"/>
    </logger>

    <logger id="tracelog" fileName="analysis.log" logEvery="{cfg.log_every}" model="@posterior" sanitiseHeaders="true" sort="smart">
        <log idref="posterior"/>
        <log idref="likelihood"/>
        <log idref="prior"/>
        <log idref="Tree.t:alignment"/>
        {bits["logs"]}
    </logger>

    <logger id="treelog" fileName="analysis.trees" logEvery="{cfg.log_every}" mode="tree">
        <log idref="Tree.t:alignment"/>
    </logger>
</run>
</beast>
"""
    xml_path.write_text(xml, encoding="utf-8")


# ============================================================
# BEAST execution
# ============================================================
def build_beast_command(xml_path: Path, threads: int) -> List[str]:
    beast_bin = resolve_beast_bin()
    if not beast_bin:
        raise RuntimeError("BEAST executable not found. Set it in Settings or add it to PATH.")
    return [beast_bin, "-threads", str(threads), str(xml_path)]


def build_treeannotator_command(burnin_percent: int, trees_path: Path, out_tree: Path) -> Optional[List[str]]:
    tree_bin = resolve_treeannotator_bin()
    if not tree_bin:
        return None
    return [tree_bin, "-burnin", str(burnin_percent), str(trees_path), str(out_tree)]


def parse_trace_progress(log_path: Path, chain_length: int) -> Tuple[float, Optional[dict]]:
    if not log_path.exists():
        return 0.0, None

    header = None
    last = None
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if header is None and "\t" in line:
                    header = line.split("\t")
                    continue
                if header and "\t" in line:
                    parts = line.split("\t")
                    if len(parts) == len(header):
                        last = dict(zip(header, parts))

        if not last:
            return 0.0, None

        state = None
        for key in ("Sample", "sample", "state"):
            if key in last:
                try:
                    state = int(float(last[key]))
                    break
                except Exception:
                    pass

        if state is None:
            return 3.0, last

        progress = max(0.0, min(100.0, 100.0 * state / chain_length))
        return progress, last
    except Exception:
        return 0.0, None


# ============================================================
# Worker
# ============================================================
def execute_job(job_id: str, cfg: BeastConfig) -> None:
    job = load_job(job_id)
    paths = {k: Path(v) for k, v in job["paths"].items()}

    try:
        set_job(job_id, status="preparing", progress=2.0, message="Loading uploaded sequence file...")
        append_event(job_id, "Job started.")

        raw_records = load_alignment_records(paths["raw_input"])
        validate_taxa_names(raw_records)

        if cfg.run_alignment:
            append_event(job_id, "MUSCLE alignment requested.")
            tmp_in = paths["input_dir"] / Path("muscle_input.fasta")
            if isinstance(tmp_in, str):
                tmp_in = Path(tmp_in)
            write_fasta(raw_records, tmp_in)
            muscle_cmd = build_muscle_command(tmp_in, paths["aligned_input"])
            set_job(job_id, status="preparing", progress=8.0, message="Running MUSCLE alignment...")
            rc = run_subprocess(muscle_cmd, paths["input_dir"], paths["alignment_stdout"], paths["alignment_stderr"])
            if rc != 0:
                raise RuntimeError(f"MUSCLE failed with exit code {rc}\n{safe_read_text(paths['alignment_stderr'], 20000)}")
            append_event(job_id, "MUSCLE alignment completed.")
            records = load_alignment_records(paths["aligned_input"])
        else:
            if not is_rectangular_alignment(raw_records):
                raise ValueError("Input sequences are not equal length. Either upload an aligned FASTA/NEXUS or enable MUSCLE alignment.")
            records = raw_records
            write_fasta(records, paths["aligned_input"])
            append_event(job_id, "Input used as already aligned dataset.")

        validate_alignment(records)
        meta = parse_metadata_csv(paths["metadata_csv"])

        summary, taxa, mat = alignment_summary(records, meta)
        save_pairwise_matrix(taxa, mat, paths["pairwise_csv"], paths["pairwise_json"])

        upgma_nwk = upgma_tree_newick(taxa, mat)
        paths["upgma_nwk"].write_text(upgma_nwk, encoding="utf-8")
        paths["upgma_nexus"].write_text(newick_to_nexus(upgma_nwk, "upgma"), encoding="utf-8")

        paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["summary_txt"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

        set_job(
            job_id,
            summary=summary,
            pairwise_taxa=taxa,
            pairwise_matrix=mat,
            tree_newick=upgma_nwk,
            status="preparing",
            progress=14.0,
            message="Generating BEAST XML...",
        )
        append_event(job_id, "Detailed QC, pairwise distances, and UPGMA tree generated.")

        build_beast_xml(records, meta, cfg, paths["xml"])
        append_event(job_id, "BEAST XML created successfully.")
        set_job(job_id, status="ready", progress=20.0, message="XML ready. Starting BEAST...")

        beast_cmd = build_beast_command(paths["xml"], cfg.threads)
        set_job(job_id, status="running", progress=24.0, message="BEAST is running...")
        append_event(job_id, f"Launching BEAST with {cfg.threads} thread(s).")

        stop_flag = {"stop": False}

        def monitor():
            while not stop_flag["stop"]:
                progress, last = parse_trace_progress(paths["trace_log"], cfg.chain_length)
                manifest = build_manifest(load_job(job_id))
                set_job(job_id, manifest=manifest)
                if last:
                    msg = "BEAST running"
                    posterior = last.get("posterior")
                    likelihood = last.get("likelihood")
                    prior = last.get("prior")
                    if posterior is not None:
                        msg += f" | posterior={posterior}"
                    if likelihood is not None:
                        msg += f" | likelihood={likelihood}"
                    if prior is not None:
                        msg += f" | prior={prior}"
                    set_job(job_id, progress=max(24.0, progress), message=msg)
                time.sleep(2)

        mon = threading.Thread(target=monitor, daemon=True)
        mon.start()

        rc = run_subprocess(beast_cmd, paths["output_dir"], paths["runner_stdout"], paths["runner_stderr"])
        stop_flag["stop"] = True
        mon.join(timeout=2)

        if rc != 0:
            append_event(job_id, f"BEAST failed with exit code {rc}.")
            raise RuntimeError(f"BEAST failed with exit code {rc}\n{safe_read_text(paths['runner_stderr'], 20000)}")

        append_event(job_id, "BEAST completed successfully.")
        set_job(job_id, status="postprocessing", progress=95.0, message="Post-processing outputs...")

        if paths["trees"].exists():
            ta_cmd = build_treeannotator_command(cfg.treeannotator_burnin_percent, paths["trees"], paths["mcc_tree"])
            if ta_cmd:
                append_event(job_id, "Running TreeAnnotator...")
                rc2 = run_subprocess(ta_cmd, paths["output_dir"], paths["runner_stdout"], paths["runner_stderr"])
                if rc2 == 0:
                    append_event(job_id, "TreeAnnotator completed successfully.")
                else:
                    append_event(job_id, f"TreeAnnotator failed with exit code {rc2}.")
            else:
                append_event(job_id, "TreeAnnotator not configured; MCC tree skipped.")
        else:
            append_event(job_id, "Posterior trees file not found; MCC tree skipped.")

        manifest = build_manifest(load_job(job_id))
        set_job(job_id, status="finished", progress=100.0, message="Analysis complete.", manifest=manifest)
        append_event(job_id, "Job completed.")
    except Exception as e:
        append_event(job_id, f"ERROR: {str(e)}")
        manifest = build_manifest(load_job(job_id))
        set_job(
            job_id,
            status="failed",
            progress=100.0,
            message="Job failed.",
            error=str(e),
            manifest=manifest,
        )


# ============================================================
# Startup
# ============================================================
@app.on_event("startup")
async def startup_probe():
    runs_root = get_runs_root()
    health = {
        "time": iso_now(),
        "beast_bin": resolve_beast_bin(),
        "treeannotator_bin": resolve_treeannotator_bin(),
        "muscle_bin": resolve_muscle_bin(),
        "runs_root": str(runs_root),
        "java_opts": get_java_opts(),
    }
    (runs_root / "service_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")


# ============================================================
# API
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


@app.get("/api/health")
async def health():
    return {
        "app": APP_TITLE,
        "time": iso_now(),
        "beast_bin": resolve_beast_bin(),
        "treeannotator_bin": resolve_treeannotator_bin(),
        "muscle_bin": resolve_muscle_bin(),
        "runs_root": str(get_runs_root()),
        "java_opts": get_java_opts(),
        "settings_file": str(SETTINGS_FILE),
    }


@app.get("/api/settings")
async def api_get_settings():
    return load_settings()


@app.post("/api/settings")
async def api_save_settings(settings: AppSettingsModel):
    data = settings.model_dump()
    if data["runs_root"].strip():
        rr = Path(data["runs_root"]).resolve()
        rr.mkdir(parents=True, exist_ok=True)
        data["runs_root"] = str(rr)
    saved = save_settings(data)
    return {"ok": True, "settings": saved}


@app.get("/api/preflight")
async def preflight():
    beast_bin = resolve_beast_bin()
    tree_bin = resolve_treeannotator_bin()
    muscle_bin = resolve_muscle_bin()
    runs_root = get_runs_root()

    problems = []
    if not beast_bin:
        problems.append("BEAST executable not found.")
    if not runs_root.exists():
        problems.append("Runs root does not exist.")
    if not os.access(runs_root, os.W_OK):
        problems.append("Runs root is not writable.")

    return {
        "beast_bin": beast_bin,
        "treeannotator_bin": tree_bin,
        "muscle_bin": muscle_bin,
        "runs_root": str(runs_root),
        "runs_root_writable": os.access(runs_root, os.W_OK),
        "problems": problems,
        "ok": len(problems) == 0,
    }


@app.post("/api/jobs")
async def create_job(
    alignment: UploadFile = File(...),
    metadata_csv: Optional[UploadFile] = File(None),
    config_json: str = Form(...),
):
    try:
        cfg = BeastConfig(**json.loads(config_json))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid config: {e}")

    if not alignment.filename:
        raise HTTPException(status_code=400, detail="Sequence file is required.")

    ext = Path(alignment.filename).suffix.lower()
    if ext not in {".fa", ".fas", ".fasta", ".nex", ".nexus", ".txt"}:
        raise HTTPException(status_code=400, detail="Sequence input must be FASTA or NEXUS.")

    job_id = str(uuid.uuid4())
    paths = make_job_dirs(job_id, ext if ext else ".fasta")

    with Path(paths["raw_input"]).open("wb") as f:
        shutil.copyfileobj(alignment.file, f)

    if metadata_csv and metadata_csv.filename:
        with Path(paths["metadata_csv"]).open("wb") as f:
            shutil.copyfileobj(metadata_csv.file, f)
    else:
        Path(paths["metadata_csv"]).write_text("", encoding="utf-8")

    job = {
        "job_id": job_id,
        "job_name": cfg.job_name,
        "status": "queued",
        "progress": 0.0,
        "message": "Job accepted and queued.",
        "error": None,
        "created_at": now_ts(),
        "updated_at": now_ts(),
        "events": [],
        "summary": {},
        "manifest": [],
        "config": cfg.model_dump(),
        "capabilities": {
            "beast_bin": resolve_beast_bin(),
            "treeannotator_bin": resolve_treeannotator_bin(),
            "muscle_bin": resolve_muscle_bin(),
            "runs_root": str(get_runs_root()),
        },
        "pairwise_taxa": [],
        "pairwise_matrix": [],
        "tree_newick": "",
        "paths": paths,
    }

    with JOB_LOCK:
        JOBS[job_id] = job
        persist_job(job_id)

    append_event(job_id, "Job created.")
    append_event(job_id, f"Resolved BEAST binary: {resolve_beast_bin()}")
    append_event(job_id, f"Resolved TreeAnnotator binary: {resolve_treeannotator_bin()}")
    append_event(job_id, f"Resolved MUSCLE binary: {resolve_muscle_bin()}")
    append_event(job_id, f"Runs root: {get_runs_root()}")

    t = threading.Thread(target=execute_job, args=(job_id, cfg), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued", "message": "Job created successfully."}


@app.get("/api/jobs")
async def list_jobs():
    runs_root = get_runs_root()
    items = []
    for d in sorted(runs_root.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        state = d / "job_state.json"
        if state.exists():
            try:
                job = json.loads(state.read_text(encoding="utf-8"))
                items.append({
                    "job_id": job["job_id"],
                    "job_name": job["job_name"],
                    "status": job["status"],
                    "progress": job["progress"],
                    "updated_at": job["updated_at"],
                    "message": job["message"],
                })
            except Exception:
                pass
    return {"jobs": items[:50]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = load_job(job_id)
    job["manifest"] = build_manifest(job)
    persist_job(job_id)
    return JSONResponse(job)


@app.get("/api/jobs/{job_id}/files")
async def get_job_files(job_id: str):
    job = load_job(job_id)
    manifest = build_manifest(job)
    set_job(job_id, manifest=manifest)
    return {"files": manifest}


@app.get("/api/jobs/{job_id}/summary")
async def get_job_summary(job_id: str):
    job = load_job(job_id)
    return JSONResponse(job.get("summary", {}))


@app.get("/api/jobs/{job_id}/pairwise")
async def get_job_pairwise(job_id: str):
    job = load_job(job_id)
    return {
        "taxa": job.get("pairwise_taxa", []),
        "matrix": job.get("pairwise_matrix", []),
    }


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str):
    return {"events": load_job(job_id).get("events", [])}


@app.get("/api/jobs/{job_id}/xml")
async def get_job_xml(job_id: str):
    job = load_job(job_id)
    p = Path(job["paths"]["xml"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="XML not available yet")
    return PlainTextResponse(p.read_text(encoding="utf-8", errors="replace"))


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    job = load_job(job_id)
    parts = []
    for key in ["event_log", "alignment_stdout", "alignment_stderr", "runner_stdout", "runner_stderr", "trace_log"]:
        p = Path(job["paths"][key])
        if p.exists():
            parts.append(f"\n===== {p.name} =====\n")
            parts.append(safe_read_text(p, 120000))
    return PlainTextResponse("\n".join(parts) if parts else "No logs yet.")


@app.get("/api/jobs/{job_id}/download/{key}")
async def download_file(job_id: str, key: str):
    job = load_job(job_id)
    if key not in job["paths"]:
        raise HTTPException(status_code=404, detail="Unknown file key")
    p = Path(job["paths"][key])
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail=f"{key} file not found")
    return FileResponse(p, filename=p.name)


@app.post("/api/jobs/{job_id}/tree/save")
async def save_edited_tree(job_id: str, payload: dict):
    job = load_job(job_id)
    newick = str(payload.get("newick", "")).strip()
    validate_newick_text(newick)

    paths = {k: Path(v) for k, v in job["paths"].items()}
    paths["edited_tree_nwk"].write_text(newick, encoding="utf-8")
    paths["edited_tree_nexus"].write_text(newick_to_nexus(newick, "edited_tree"), encoding="utf-8")

    append_event(job_id, "Edited tree saved.")
    manifest = build_manifest(load_job(job_id))
    set_job(job_id, tree_newick=newick, manifest=manifest)

    return {"ok": True, "newick": newick}


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str):
    await websocket.accept()
    try:
        while True:
            job = load_job(job_id)
            manifest = build_manifest(job)
            await websocket.send_json({
                "job_id": job["job_id"],
                "job_name": job["job_name"],
                "status": job["status"],
                "progress": job["progress"],
                "message": job["message"],
                "error": job["error"],
                "summary": job["summary"],
                "manifest": manifest,
                "events_tail": job.get("events", [])[-15:],
                "pairwise_taxa": job.get("pairwise_taxa", []),
                "pairwise_matrix": job.get("pairwise_matrix", []),
                "tree_newick": job.get("tree_newick", ""),
            })
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return


# ============================================================
# Frontend
# ============================================================
HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BayesPhylo Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root{
  --bg1:#07101f;
  --bg2:#0d1730;
  --panel:#101b39;
  --text:#eef4ff;
  --muted:#9fb1d9;
  --line:rgba(255,255,255,.08);
  --blue:#7aa2ff;
  --cyan:#37e7c4;
  --green:#37d67a;
  --yellow:#f4c95d;
  --red:#ff7b7b;
  --shadow:0 18px 48px rgba(0,0,0,.35);
  --r:22px;
}
*{box-sizing:border-box}
body{
  margin:0;
  font-family:Inter,Segoe UI,Arial,sans-serif;
  color:var(--text);
  background:
    radial-gradient(circle at 15% 10%, rgba(122,162,255,.18), transparent 25%),
    radial-gradient(circle at 80% 0%, rgba(55,231,196,.10), transparent 22%),
    linear-gradient(180deg,var(--bg1),var(--bg2));
  min-height:100vh;
}
.container{width:min(1700px,96vw);margin:20px auto 34px}
.hero{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}
.card{
  background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02));
  border:1px solid var(--line);
  border-radius:var(--r);
  box-shadow:var(--shadow);
  backdrop-filter:blur(12px);
}
.hero-left{padding:28px;position:relative;overflow:hidden}
.hero-left:before{
  content:"";position:absolute;inset:-2px;
  background:linear-gradient(125deg, rgba(122,162,255,.16), transparent 40%, rgba(55,231,196,.12), transparent 70%);
  pointer-events:none;
}
h1{margin:0 0 10px;font-size:clamp(2.1rem,4vw,3.8rem);letter-spacing:-.04em;line-height:1.0}
.subtitle{color:var(--muted);font-size:1rem;line-height:1.7;max-width:900px}
.pills{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.pill{
  padding:8px 12px;border-radius:999px;border:1px solid var(--line);
  background:rgba(255,255,255,.05);color:#dce6ff;font-size:.9rem
}
.hero-right{padding:20px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
.stat{
  padding:16px;border-radius:18px;border:1px solid var(--line);
  background:linear-gradient(180deg,rgba(122,162,255,.12),rgba(255,255,255,.03))
}
.stat .label{color:var(--muted);font-size:.86rem}
.stat .value{font-size:.98rem;font-weight:800;margin-top:4px;word-break:break-word}
.main{display:grid;grid-template-columns:560px 1fr;gap:20px}
.panel{padding:22px}
h2{margin:0 0 8px;font-size:1.2rem}
.desc{color:var(--muted);font-size:.95rem;margin-bottom:16px}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.field{display:flex;flex-direction:column;gap:7px}
.field.full{grid-column:1/-1}
label{font-size:.9rem;font-weight:700;color:#dce6ff}
input[type=text],input[type=number],select,textarea{
  width:100%;padding:12px 14px;border-radius:14px;border:1px solid rgba(255,255,255,.1);
  background:#0b1430;color:var(--text);outline:none
}
textarea{min-height:120px;resize:vertical}
input[type=file]{
  width:100%;padding:12px;border-radius:14px;border:1px dashed rgba(255,255,255,.14);
  background:#0b1430;color:var(--muted)
}
input:focus,select:focus,textarea:focus{box-shadow:0 0 0 4px rgba(122,162,255,.13)}
.toggles{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}
.toggle{
  display:flex;align-items:center;gap:8px;padding:10px 12px;border-radius:14px;
  background:rgba(255,255,255,.04);border:1px solid var(--line)
}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
button{
  border:none;border-radius:16px;padding:13px 17px;background:linear-gradient(135deg,var(--blue),#a4bbff);
  color:#071226;font-weight:900;cursor:pointer
}
button.secondary{background:rgba(255,255,255,.06);color:var(--text);border:1px solid var(--line)}
.workspace{display:grid;grid-template-rows:auto auto 1fr;gap:16px}
.runbox{padding:18px;border-radius:18px;background:rgba(255,255,255,.04);border:1px solid var(--line)}
.runhead{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.badge{
  padding:8px 12px;border-radius:999px;border:1px solid var(--line);
  font-size:.84rem;font-weight:800;text-transform:capitalize
}
.queued{background:rgba(244,201,93,.12);color:#ffe69a}
.preparing,.ready,.postprocessing{background:rgba(122,162,255,.12);color:#d5e0ff}
.running{background:rgba(55,231,196,.12);color:#b9ffef}
.finished{background:rgba(55,214,122,.12);color:#bfffd4}
.failed{background:rgba(255,123,123,.13);color:#ffd0d0}
.progress{height:13px;border-radius:999px;background:#0a1228;border:1px solid var(--line);overflow:hidden;margin-top:12px}
.bar{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--cyan));transition:width .45s ease}
.meta{color:var(--muted);margin-top:8px;line-height:1.6;font-size:.94rem}
.cards8{display:grid;grid-template-columns:repeat(8,1fr);gap:12px}
.metric{padding:14px;border-radius:16px;border:1px solid var(--line);background:rgba(255,255,255,.04)}
.metric .k{font-size:.82rem;color:var(--muted)}
.metric .v{font-size:1.1rem;font-weight:800;margin-top:4px}
.tabs{display:flex;gap:8px;flex-wrap:wrap}
.tabbtn{
  padding:10px 12px;border-radius:12px;border:1px solid var(--line);
  background:rgba(255,255,255,.05);color:var(--text);cursor:pointer;font-weight:700
}
.tabbtn.active{background:linear-gradient(135deg,rgba(122,162,255,.24),rgba(55,231,196,.14))}
.tab{display:none}
.tab.active{display:block}
.panelgrid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.box{border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.04);padding:14px}
pre{
  margin:0;white-space:pre-wrap;word-break:break-word;background:#081020;border:1px solid var(--line);
  border-radius:14px;color:#dbe5ff;padding:14px;max-height:450px;overflow:auto;font-size:.86rem
}
.table{width:100%;border-collapse:collapse;font-size:.9rem}
.table th,.table td{padding:8px;border-bottom:1px solid rgba(255,255,255,.07);text-align:left;vertical-align:top}
.table th{color:#cbd8ff;position:sticky;top:0;background:#0b1430}
.filelink{
  display:inline-block;padding:8px 10px;border-radius:10px;background:rgba(255,255,255,.06);
  border:1px solid var(--line);color:var(--text);text-decoration:none;font-size:.88rem
}
.small{font-size:.85rem;color:var(--muted)}
.history-item{padding:10px 0;border-bottom:1px solid rgba(255,255,255,.07)}
canvas{width:100%;height:280px;background:#081020;border-radius:14px;border:1px solid var(--line)}
.treewrap{overflow:auto;border:1px solid var(--line);border-radius:14px;background:#081020;padding:8px}
svg text{fill:#eaf1ff;font-size:12px;font-family:Inter, sans-serif}
@media (max-width:1400px){
  .hero,.main,.cards8,.panelgrid,.form-grid{grid-template-columns:1fr}
  .hero-right{grid-template-columns:1fr}
}
</style>
</head>
<body>
<div class="container">

  <section class="hero">
    <div class="card hero-left">
      <h1>BayesPhylo Studio</h1>
      <div class="subtitle">
        End-to-end Bayesian phylogenetics workspace with optional MUSCLE alignment, genetic pairwise-distance analysis,
        browser tree viewer/editor, Newick/NEXUS export, and BEAST execution.
      </div>
      <div class="pills">
        <div class="pill">MUSCLE alignment</div>
        <div class="pill">Pairwise distance matrix</div>
        <div class="pill">UPGMA tree</div>
        <div class="pill">Tree viewer/editor</div>
        <div class="pill">Newick + NEXUS export</div>
        <div class="pill">Live monitoring</div>
      </div>
    </div>
    <div class="card hero-right">
      <div class="stat"><div class="label">BEAST binary</div><div class="value" id="health_beast">checking...</div></div>
      <div class="stat"><div class="label">TreeAnnotator</div><div class="value" id="health_treeannotator">checking...</div></div>
      <div class="stat"><div class="label">MUSCLE</div><div class="value" id="health_muscle">checking...</div></div>
      <div class="stat"><div class="label">Runs root</div><div class="value" id="health_runs_root">checking...</div></div>
    </div>
  </section>

  <section class="main">
    <div class="card panel">
      <h2>Application settings</h2>
      <div class="desc">Set local or deployment paths here.</div>

      <div class="form-grid">
        <div class="field full">
          <label>BEAST executable path</label>
          <input id="set_beast_bin" type="text" placeholder="Path to beast or beast.bat">
        </div>
        <div class="field full">
          <label>TreeAnnotator path</label>
          <input id="set_treeannotator_bin" type="text" placeholder="Path to treeannotator">
        </div>
        <div class="field full">
          <label>MUSCLE path</label>
          <input id="set_muscle_bin" type="text" placeholder="Path to muscle or muscle.exe">
        </div>
        <div class="field full">
          <label>Runs root</label>
          <input id="set_runs_root" type="text" placeholder="Directory for run outputs">
        </div>
        <div class="field full">
          <label>JAVA_OPTS</label>
          <input id="set_java_opts" type="text" placeholder="-Xms512m -Xmx4g">
        </div>
      </div>

      <div class="actions">
        <button type="button" id="saveSettings">Save settings</button>
        <button type="button" class="secondary" id="runPreflight">Run preflight</button>
      </div>

      <div style="margin-top:18px">
        <h2>Create analysis</h2>
        <div class="desc">Upload FASTA/NEXUS, optionally align with MUSCLE, then run QC, tree generation, and BEAST.</div>

        <form id="jobForm">
          <div class="form-grid">
            <div class="field full">
              <label>Job name</label>
              <input id="job_name" type="text" value="Time-calibrated PRRSV BEAST run">
            </div>

            <div class="field full">
              <label>Sequence file</label>
              <input id="alignment" type="file" accept=".fa,.fas,.fasta,.nex,.nexus,.txt" required>
            </div>

            <div class="field full">
              <label>Metadata CSV</label>
              <input id="metadata_csv" type="file" accept=".csv">
              <div class="small">Expected columns: <code>taxon,date</code>.</div>
            </div>

            <div class="field">
              <label>Chain length</label>
              <input id="chain_length" type="number" value="1000000" min="10000" step="1000">
            </div>
            <div class="field">
              <label>Log every</label>
              <input id="log_every" type="number" value="1000" min="100" step="100">
            </div>
            <div class="field">
              <label>Screen every</label>
              <input id="screen_every" type="number" value="1000" min="100" step="100">
            </div>
            <div class="field">
              <label>Threads</label>
              <input id="threads" type="number" value="2" min="1" max="64">
            </div>
            <div class="field">
              <label>Substitution model</label>
              <select id="substitution_model">
                <option value="HKY">HKY</option>
                <option value="GTR">GTR</option>
              </select>
            </div>
            <div class="field">
              <label>Clock model</label>
              <select id="clock_model">
                <option value="strict">Strict</option>
                <option value="relaxed_lognormal">Relaxed lognormal</option>
              </select>
            </div>
            <div class="field">
              <label>Tree prior</label>
              <select id="tree_prior">
                <option value="coalescent_constant">Coalescent constant</option>
                <option value="coalescent_skyline">Bayesian skyline</option>
                <option value="birth_death">Birth-death</option>
              </select>
            </div>
            <div class="field">
              <label>Gamma categories</label>
              <input id="gamma_categories" type="number" value="4" min="1" max="8">
            </div>
          </div>

          <div class="toggles">
            <label class="toggle"><input id="use_tip_dates" type="checkbox" checked> Use tip dates</label>
            <label class="toggle"><input id="estimate_base_freqs" type="checkbox" checked> Estimate base frequencies</label>
            <label class="toggle"><input id="sample_prior_only" type="checkbox"> Prior only</label>
            <label class="toggle"><input id="run_alignment" type="checkbox"> Run MUSCLE alignment</label>
          </div>

          <div class="actions">
            <button type="submit">Launch analysis</button>
            <button type="button" class="secondary" id="refreshHistory">Refresh history</button>
          </div>
        </form>
      </div>

      <div style="margin-top:18px">
        <h2 style="font-size:1rem">Recent jobs</h2>
        <div id="historyBox" class="small">No jobs loaded yet.</div>
      </div>
    </div>

    <div class="workspace">
      <div class="card panel runbox">
        <div class="runhead">
          <div>
            <div style="font-size:1.2rem;font-weight:900" id="jobTitle">No active job</div>
            <div class="small" id="jobId">Job ID will appear here</div>
          </div>
          <div id="statusBadge" class="badge queued">idle</div>
        </div>
        <div class="progress"><div class="bar" id="progressBar"></div></div>
        <div class="meta" id="jobMessage">Submit a job to begin.</div>
      </div>

      <div class="cards8">
        <div class="card panel metric"><div class="k">Taxa</div><div class="v" id="m_taxa">-</div></div>
        <div class="card panel metric"><div class="k">Length</div><div class="v" id="m_len">-</div></div>
        <div class="card panel metric"><div class="k">Mean GC%</div><div class="v" id="m_gc">-</div></div>
        <div class="card panel metric"><div class="k">Variable</div><div class="v" id="m_var">-</div></div>
        <div class="card panel metric"><div class="k">PI sites</div><div class="v" id="m_pi">-</div></div>
        <div class="card panel metric"><div class="k">Singleton</div><div class="v" id="m_singleton">-</div></div>
        <div class="card panel metric"><div class="k">Ambig frac</div><div class="v" id="m_ambig">-</div></div>
        <div class="card panel metric"><div class="k">Metadata cov</div><div class="v" id="m_meta_cov">-</div></div>
      </div>

      <div class="card panel">
        <div class="tabs">
          <button class="tabbtn active" data-tab="tab-overview">Overview</button>
          <button class="tabbtn" data-tab="tab-pairwise">Pairwise distances</button>
          <button class="tabbtn" data-tab="tab-tree">Tree viewer/editor</button>
          <button class="tabbtn" data-tab="tab-files">Files</button>
          <button class="tabbtn" data-tab="tab-logs">Logs</button>
          <button class="tabbtn" data-tab="tab-xml">XML</button>
          <button class="tabbtn" data-tab="tab-events">Events</button>
        </div>

        <div id="tab-overview" class="tab active" style="margin-top:14px">
          <div class="panelgrid">
            <div class="box">
              <div style="font-weight:800;margin-bottom:10px">Alignment statistics</div>
              <table class="table">
                <tr><th>Pairwise distance min</th><td id="pd_min">-</td></tr>
                <tr><th>Pairwise distance mean</th><td id="pd_mean">-</td></tr>
                <tr><th>Pairwise distance max</th><td id="pd_max">-</td></tr>
                <tr><th>Pairwise distance stdev</th><td id="pd_sd">-</td></tr>
                <tr><th>Constant sites</th><td id="constant_sites">-</td></tr>
                <tr><th>Sites with ambiguity/gap</th><td id="amb_sites">-</td></tr>
                <tr><th>Metadata matched</th><td id="meta_rows">-</td></tr>
                <tr><th>Date range</th><td id="date_range">-</td></tr>
              </table>
            </div>
            <div class="box">
              <div style="font-weight:800;margin-bottom:10px">Base composition</div>
              <canvas id="baseCanvas" width="700" height="280"></canvas>
            </div>
          </div>
        </div>

        <div id="tab-pairwise" class="tab" style="margin-top:14px">
          <div class="box">
            <div style="font-weight:800;margin-bottom:10px">Genetic pairwise distance matrix</div>
            <div id="pairwiseBox" class="small">No pairwise matrix yet.</div>
          </div>
        </div>

        <div id="tab-tree" class="tab" style="margin-top:14px">
          <div class="panelgrid">
            <div class="box">
              <div style="font-weight:800;margin-bottom:10px">Tree viewer</div>
              <div class="treewrap">
                <svg id="treeSvg" width="900" height="520"></svg>
              </div>
            </div>
            <div class="box">
              <div style="font-weight:800;margin-bottom:10px">Tree editor</div>
              <textarea id="newickEditor" placeholder="Newick will appear here"></textarea>
              <div class="actions">
                <button type="button" id="renderTree">Render tree</button>
                <button type="button" class="secondary" id="saveTree">Save edited tree</button>
              </div>
              <div class="small" style="margin-top:10px">
                You can edit the Newick text directly, re-render it, and save it as edited tree outputs.
              </div>
            </div>
          </div>
        </div>

        <div id="tab-files" class="tab" style="margin-top:14px">
          <div class="box">
            <div style="font-weight:800;margin-bottom:10px">Available files</div>
            <div id="filesBox" class="small">No files yet.</div>
          </div>
        </div>

        <div id="tab-logs" class="tab" style="margin-top:14px">
          <div class="box">
            <div style="font-weight:800;margin-bottom:10px">Runtime logs</div>
            <pre id="logsBox">No logs yet.</pre>
          </div>
        </div>

        <div id="tab-xml" class="tab" style="margin-top:14px">
          <div class="box">
            <div style="font-weight:800;margin-bottom:10px">Generated XML</div>
            <pre id="xmlBox">No XML yet.</pre>
          </div>
        </div>

        <div id="tab-events" class="tab" style="margin-top:14px">
          <div class="box">
            <div style="font-weight:800;margin-bottom:10px">Event stream</div>
            <pre id="eventsBox">No events yet.</pre>
          </div>
        </div>
      </div>
    </div>
  </section>
</div>

<script>
let activeJobId = null;
let ws = null;
let currentSummary = {};
let currentTreeNewick = "";
let currentPairwiseTaxa = [];
let currentPairwiseMatrix = [];

function qs(id){ return document.getElementById(id); }

function setBadge(status){
  const el = qs("statusBadge");
  el.className = "badge " + status;
  el.textContent = status;
}

function setProgress(v){
  qs("progressBar").style.width = `${Math.max(0, Math.min(100, v || 0))}%`;
}

function cfgFromForm(){
  return {
    job_name: qs("job_name").value.trim(),
    chain_length: Number(qs("chain_length").value),
    log_every: Number(qs("log_every").value),
    screen_every: Number(qs("screen_every").value),
    threads: Number(qs("threads").value),
    substitution_model: qs("substitution_model").value,
    clock_model: qs("clock_model").value,
    tree_prior: qs("tree_prior").value,
    gamma_categories: Number(qs("gamma_categories").value),
    use_tip_dates: qs("use_tip_dates").checked,
    estimate_base_freqs: qs("estimate_base_freqs").checked,
    sample_prior_only: qs("sample_prior_only").checked,
    treeannotator_burnin_percent: 10,
    run_alignment: qs("run_alignment").checked,
    assume_already_aligned: !qs("run_alignment").checked
  };
}

function settingsFromForm(){
  return {
    beast_bin: qs("set_beast_bin").value.trim(),
    treeannotator_bin: qs("set_treeannotator_bin").value.trim(),
    muscle_bin: qs("set_muscle_bin").value.trim(),
    runs_root: qs("set_runs_root").value.trim(),
    java_opts: qs("set_java_opts").value.trim()
  };
}

function updateSummary(s){
  currentSummary = s || {};
  qs("m_taxa").textContent = s.taxa_count ?? "-";
  qs("m_len").textContent = s.alignment_length ?? "-";
  qs("m_gc").textContent = s.gc_mean ?? "-";
  qs("m_var").textContent = s.variable_sites ?? "-";
  qs("m_pi").textContent = s.parsimony_informative_sites ?? "-";
  qs("m_singleton").textContent = s.singleton_sites ?? "-";
  qs("m_ambig").textContent = s.ambiguity_fraction ?? "-";
  qs("m_meta_cov").textContent = s.metadata_coverage_fraction ?? "-";

  qs("pd_min").textContent = s.pairwise_distance_min ?? "-";
  qs("pd_mean").textContent = s.pairwise_distance_mean ?? "-";
  qs("pd_max").textContent = s.pairwise_distance_max ?? "-";
  qs("pd_sd").textContent = s.pairwise_distance_stdev ?? "-";
  qs("constant_sites").textContent = s.constant_sites ?? "-";
  qs("amb_sites").textContent = s.sites_with_ambiguity_or_gap ?? "-";
  qs("meta_rows").textContent = s.metadata_rows_matched ?? "-";
  qs("date_range").textContent = (s.metadata_date_min && s.metadata_date_max)
    ? `${s.metadata_date_min} to ${s.metadata_date_max}` : "-";

  drawBaseComposition(s.base_counts || {});
}

function drawBaseComposition(counts){
  const canvas = qs("baseCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "#081020";
  ctx.fillRect(0,0,w,h);

  const keys = ["A","C","G","T","N","-","?"];
  const vals = keys.map(k => counts[k] || 0);
  const maxV = Math.max(1, ...vals);
  const pad = 45;
  const barW = 60;
  const gap = 22;
  const startX = 45;
  const colors = ["#7aa2ff","#37e7c4","#6ee7ff","#f4c95d","#ff9d66","#c084fc","#ff7b7b"];

  ctx.strokeStyle = "rgba(255,255,255,.10)";
  ctx.beginPath();
  ctx.moveTo(pad, h-pad);
  ctx.lineTo(w-pad, h-pad);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h-pad);
  ctx.stroke();

  keys.forEach((k, i) => {
    const x = startX + i * (barW + gap);
    const v = vals[i];
    const barH = ((h - pad*2) * v) / maxV;
    const y = h - pad - barH;
    ctx.fillStyle = colors[i];
    ctx.fillRect(x, y, barW, barH);
    ctx.fillStyle = "#e9f2ff";
    ctx.font = "14px Inter";
    ctx.fillText(k, x + 22, h - pad + 20);
    ctx.fillText(String(v), x + 4, y - 8);
  });
}

function updateFiles(manifest){
  const box = qs("filesBox");
  if(!manifest || manifest.length === 0){
    box.innerHTML = '<div class="small">No files available yet.</div>';
    return;
  }
  let html = '<table class="table"><tr><th>File</th><th>Size</th><th>Download</th></tr>';
  for(const f of manifest){
    html += `<tr>
      <td>${f.label}<div class="small">${f.filename}</div></td>
      <td>${f.size_human}</td>
      <td><a class="filelink" target="_blank" href="/api/jobs/${activeJobId}/download/${f.key}">Download</a></td>
    </tr>`;
  }
  html += '</table>';
  box.innerHTML = html;
}

function updatePairwise(taxa, matrix){
  currentPairwiseTaxa = taxa || [];
  currentPairwiseMatrix = matrix || [];
  const box = qs("pairwiseBox");
  if(!taxa || taxa.length === 0 || !matrix || matrix.length === 0){
    box.innerHTML = '<div class="small">No pairwise matrix yet.</div>';
    return;
  }

  let html = '<div style="max-height:520px;overflow:auto"><table class="table"><tr><th>taxon</th>';
  for(const t of taxa){
    html += `<th>${t}</th>`;
  }
  html += '</tr>';

  for(let i=0;i<taxa.length;i++){
    html += `<tr><th>${taxa[i]}</th>`;
    for(let j=0;j<taxa.length;j++){
      html += `<td>${Number(matrix[i][j]).toFixed(5)}</td>`;
    }
    html += '</tr>';
  }
  html += '</table></div>';
  box.innerHTML = html;
}

async function loadHealth(){
  const r = await fetch('/api/health');
  const d = await r.json();
  qs("health_beast").textContent = d.beast_bin || "not found";
  qs("health_treeannotator").textContent = d.treeannotator_bin || "not found";
  qs("health_muscle").textContent = d.muscle_bin || "not found";
  qs("health_runs_root").textContent = d.runs_root || "-";
}

async function loadSettings(){
  const r = await fetch('/api/settings');
  const d = await r.json();
  qs("set_beast_bin").value = d.beast_bin || "";
  qs("set_treeannotator_bin").value = d.treeannotator_bin || "";
  qs("set_muscle_bin").value = d.muscle_bin || "";
  qs("set_runs_root").value = d.runs_root || "";
  qs("set_java_opts").value = d.java_opts || "";
}

async function saveSettingsAction(){
  const r = await fetch('/api/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(settingsFromForm())
  });
  const d = await r.json();
  await loadHealth();
  return d;
}

async function runPreflightAction(){
  const r = await fetch('/api/preflight');
  const d = await r.json();
  alert(d.ok ? "Preflight OK" : d.problems.join("\n"));
}

async function refreshLogs(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/logs`);
    qs("logsBox").textContent = await r.text();
  }catch(e){}
}

async function refreshXml(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/xml`);
    qs("xmlBox").textContent = await r.text();
  }catch(e){}
}

async function refreshEvents(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/events`);
    const d = await r.json();
    const lines = (d.events || []).map(x => `[${x.time}] ${x.message}`);
    qs("eventsBox").textContent = lines.join("\n") || "No events yet.";
  }catch(e){}
}

async function refreshHistory(){
  const r = await fetch('/api/jobs');
  const d = await r.json();
  const jobs = d.jobs || [];
  if(jobs.length === 0){
    qs("historyBox").innerHTML = '<div class="small">No previous jobs found.</div>';
    return;
  }
  let html = '';
  for(const j of jobs){
    html += `<div class="history-item">
      <div style="font-weight:800">${j.job_name}</div>
      <div class="small">${j.job_id}</div>
      <div class="small">${j.status} | ${Math.round(j.progress || 0)}% | ${j.message || ''}</div>
      <div style="margin-top:6px"><a class="filelink" href="#" onclick="loadJob('${j.job_id}');return false;">Open</a></div>
    </div>`;
  }
  qs("historyBox").innerHTML = html;
}

/* -------- Newick parsing/rendering -------- */
function parseNewick(s){
  let index = 0;
  function skipWs(){ while(index < s.length && /\s/.test(s[index])) index++; }
  function parseName(){
    skipWs();
    let start = index;
    while(index < s.length && !/[,:();]/.test(s[index])) index++;
    return s.slice(start, index).trim();
  }
  function parseLength(){
    skipWs();
    if(s[index] !== ':') return null;
    index++;
    skipWs();
    let start = index;
    while(index < s.length && !/[(),;]/.test(s[index])) index++;
    const val = parseFloat(s.slice(start, index).trim());
    return isNaN(val) ? 0 : val;
  }
  function parseNode(){
    skipWs();
    let node = {name:"", length:0, children:[]};
    if(s[index] === '('){
      index++;
      node.children.push(parseNode());
      skipWs();
      while(s[index] === ','){
        index++;
        node.children.push(parseNode());
        skipWs();
      }
      if(s[index] !== ')') throw new Error("Invalid Newick: missing ')'");
      index++;
      node.name = parseName() || "";
      const len = parseLength();
      node.length = len == null ? 0 : len;
      return node;
    } else {
      node.name = parseName();
      const len = parseLength();
      node.length = len == null ? 0 : len;
      return node;
    }
  }
  const tree = parseNode();
  skipWs();
  if(s[index] === ';') return tree;
  throw new Error("Invalid Newick termination");
}

function collectLeaves(node, out=[]){
  if(!node.children || node.children.length === 0){
    out.push(node);
  } else {
    node.children.forEach(ch => collectLeaves(ch, out));
  }
  return out;
}

function setYPositions(node, yMap, counter){
  if(!node.children || node.children.length === 0){
    yMap.set(node, counter.value);
    counter.value += 1;
  } else {
    node.children.forEach(ch => setYPositions(ch, yMap, counter));
    const ys = node.children.map(ch => yMap.get(ch));
    yMap.set(node, (Math.min(...ys) + Math.max(...ys)) / 2);
  }
}

function setXPositions(node, parentX, xMap, scale){
  const x = parentX + (node.length || 0) * scale;
  xMap.set(node, x);
  (node.children || []).forEach(ch => setXPositions(ch, x, xMap, scale));
}

function maxPath(node, acc=0){
  const cur = acc + (node.length || 0);
  if(!node.children || node.children.length === 0) return cur;
  return Math.max(...node.children.map(ch => maxPath(ch, cur)));
}

function renderTree(newick){
  currentTreeNewick = newick;
  qs("newickEditor").value = newick;
  const svg = qs("treeSvg");
  while(svg.firstChild) svg.removeChild(svg.firstChild);

  let tree;
  try{
    tree = parseNewick(newick);
  }catch(err){
    alert(String(err));
    return;
  }

  const leaves = collectLeaves(tree);
  const yMap = new Map();
  setYPositions(tree, yMap, {value: 0});

  const maxLen = Math.max(1e-6, maxPath(tree, 0));
  const width = 860;
  const height = Math.max(500, leaves.length * 28 + 50);
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);

  const leftPad = 20;
  const rightPad = 240;
  const topPad = 20;
  const bottomPad = 20;
  const xScale = (width - leftPad - rightPad) / maxLen;
  const yScale = Math.max(24, (height - topPad - bottomPad) / Math.max(leaves.length,1));

  const xMap = new Map();
  setXPositions(tree, 0, xMap, xScale);

  function drawLine(x1,y1,x2,y2){
    const line = document.createElementNS("http://www.w3.org/2000/svg","line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("stroke", "#a9c0ff");
    line.setAttribute("stroke-width", "1.6");
    svg.appendChild(line);
  }

  function drawText(x,y,text){
    const t = document.createElementNS("http://www.w3.org/2000/svg","text");
    t.setAttribute("x", x);
    t.setAttribute("y", y);
    t.textContent = text;
    svg.appendChild(t);
  }

  function y(node){ return topPad + yMap.get(node) * yScale; }
  function x(node){ return leftPad + xMap.get(node); }

  function drawNode(node){
    const children = node.children || [];
    if(children.length > 0){
      const ys = children.map(ch => y(ch));
      drawLine(x(node), Math.min(...ys), x(node), Math.max(...ys));
      children.forEach(ch => {
        drawLine(x(node), y(ch), x(ch), y(ch));
        drawNode(ch);
      });
    } else {
      drawText(x(node) + 6, y(node) + 4, node.name || "unnamed");
    }
  }

  drawNode(tree);
}

async function saveEditedTree(){
  if(!activeJobId){
    alert("No active job.");
    return;
  }
  const newick = qs("newickEditor").value.trim();
  const r = await fetch(`/api/jobs/${activeJobId}/tree/save`, {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({newick})
  });
  const d = await r.json();
  if(!r.ok){
    alert(d.detail || "Failed to save tree.");
    return;
  }
  renderTree(d.newick);
  refreshHistory();
}

function updatePairwiseAndTree(taxa, matrix, newick){
  updatePairwise(taxa, matrix);
  if(newick){
    renderTree(newick);
  }
}

function updatePairwise(taxa, matrix){
  currentPairwiseTaxa = taxa || [];
  currentPairwiseMatrix = matrix || [];
  const box = qs("pairwiseBox");
  if(!taxa || taxa.length === 0 || !matrix || matrix.length === 0){
    box.innerHTML = '<div class="small">No pairwise matrix yet.</div>';
    return;
  }

  let html = '<div style="max-height:540px;overflow:auto"><table class="table"><tr><th>taxon</th>';
  for(const t of taxa){
    html += `<th>${t}</th>`;
  }
  html += '</tr>';

  for(let i=0;i<taxa.length;i++){
    html += `<tr><th>${taxa[i]}</th>`;
    for(let j=0;j<taxa.length;j++){
      html += `<td>${Number(matrix[i][j]).toFixed(5)}</td>`;
    }
    html += '</tr>';
  }
  html += '</table></div>';
  box.innerHTML = html;
}

async function loadSettings(){
  const r = await fetch('/api/settings');
  const d = await r.json();
  qs("set_beast_bin").value = d.beast_bin || "";
  qs("set_treeannotator_bin").value = d.treeannotator_bin || "";
  qs("set_muscle_bin").value = d.muscle_bin || "";
  qs("set_runs_root").value = d.runs_root || "";
  qs("set_java_opts").value = d.java_opts || "";
}

function connectWs(jobId){
  if(ws){
    try{ ws.close(); }catch(e){}
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${scheme}://${location.host}/ws/jobs/${jobId}`);
  ws.onmessage = async (ev) => {
    const d = JSON.parse(ev.data);
    qs("jobTitle").textContent = d.job_name || "Active job";
    qs("jobId").textContent = `Job ID: ${d.job_id}`;
    qs("jobMessage").textContent = d.error ? `${d.message} | ${d.error}` : (d.message || "");
    setBadge(d.status || "queued");
    setProgress(d.progress || 0);
    updateSummary(d.summary || {});
    updateFiles(d.manifest || []);
    updatePairwiseAndTree(d.pairwise_taxa || [], d.pairwise_matrix || [], d.tree_newick || "");
    if(d.events_tail){
      qs("eventsBox").textContent = d.events_tail.map(x => `[${x.time}] ${x.message}`).join("\n");
    }
    refreshLogs();
    refreshXml();
  };
}

async function loadJob(jobId){
  activeJobId = jobId;
  const r = await fetch(`/api/jobs/${jobId}`);
  const d = await r.json();
  qs("jobTitle").textContent = d.job_name;
  qs("jobId").textContent = `Job ID: ${d.job_id}`;
  qs("jobMessage").textContent = d.error ? `${d.message} | ${d.error}` : d.message;
  setBadge(d.status);
  setProgress(d.progress || 0);
  updateSummary(d.summary || {});
  updateFiles(d.manifest || []);
  updatePairwiseAndTree(d.pairwise_taxa || [], d.pairwise_matrix || [], d.tree_newick || "");
  connectWs(jobId);
  refreshLogs();
  refreshXml();
  refreshEvents();
}

document.querySelectorAll(".tabbtn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabbtn").forEach(x => x.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

qs("saveSettings").addEventListener("click", async () => {
  const r = await fetch('/api/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(settingsFromForm())
  });
  if(!r.ok){
    alert("Failed to save settings.");
    return;
  }
  await loadHealth();
  alert("Settings saved.");
});

qs("runPreflight").addEventListener("click", runPreflightAction);
qs("refreshHistory").addEventListener("click", refreshHistory);
qs("renderTree").addEventListener("click", () => {
  const txt = qs("newickEditor").value.trim();
  if(txt) renderTree(txt);
});
qs("saveTree").addEventListener("click", saveEditedTree);

qs("jobForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const alignment = qs("alignment").files[0];
  const metadata = qs("metadata_csv").files[0];

  if(!alignment){
    alert("Please upload a sequence file.");
    return;
  }

  qs("jobTitle").textContent = "Submitting job...";
  qs("jobId").textContent = "";
  qs("jobMessage").textContent = "Uploading input files...";
  setBadge("queued");
  setProgress(2);
  qs("logsBox").textContent = "Waiting for logs...";
  qs("xmlBox").textContent = "Waiting for XML...";
  qs("eventsBox").textContent = "Waiting for events...";
  qs("pairwiseBox").innerHTML = '<div class="small">Waiting for pairwise matrix...</div>';

  const fd = new FormData();
  fd.append("alignment", alignment);
  if(metadata) fd.append("metadata_csv", metadata);
  fd.append("config_json", JSON.stringify(cfgFromForm()));

  try{
    const r = await fetch("/api/jobs", {method:"POST", body:fd});
    if(!r.ok){
      throw new Error(await r.text());
    }
    const d = await r.json();
    activeJobId = d.job_id;
    await loadJob(activeJobId);
    refreshHistory();
  }catch(err){
    setBadge("failed");
    setProgress(100);
    qs("jobMessage").textContent = "Submission failed.";
    qs("logsBox").textContent = String(err);
  }
});

async function refreshLogs(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/logs`);
    qs("logsBox").textContent = await r.text();
  }catch(e){}
}

async function refreshXml(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/xml`);
    qs("xmlBox").textContent = await r.text();
  }catch(e){}
}

async function refreshEvents(){
  if(!activeJobId) return;
  try{
    const r = await fetch(`/api/jobs/${activeJobId}/events`);
    const d = await r.json();
    const lines = (d.events || []).map(x => `[${x.time}] ${x.message}`);
    qs("eventsBox").textContent = lines.join("\n") || "No events yet.";
  }catch(e){}
}

async function refreshHistory(){
  const r = await fetch('/api/jobs');
  const d = await r.json();
  const jobs = d.jobs || [];
  if(jobs.length === 0){
    qs("historyBox").innerHTML = '<div class="small">No previous jobs found.</div>';
    return;
  }
  let html = '';
  for(const j of jobs){
    html += `<div class="history-item">
      <div style="font-weight:800">${j.job_name}</div>
      <div class="small">${j.job_id}</div>
      <div class="small">${j.status} | ${Math.round(j.progress || 0)}% | ${j.message || ''}</div>
      <div style="margin-top:6px"><a class="filelink" href="#" onclick="loadJob('${j.job_id}');return false;">Open</a></div>
    </div>`;
  }
  qs("historyBox").innerHTML = html;
}

(async function init(){
  await loadSettings();
  await loadHealth();
  await refreshHistory();
  drawBaseComposition({});
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)