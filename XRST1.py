from __future__ import annotations

import io
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from Bio import SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# ---------- App setup ----------
app = FastAPI(title="PRRSV Predictive Evolution Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Antigenic sites in ORF5 (amino acid positions)
ORF5_ANTIGENIC_AA = [
    30, 31, 32, 33, 34, 35,
    56, 57, 58, 59, 60,
    79, 80, 81, 82, 83, 84,
    100, 101, 102, 103, 104,
    130, 131, 132
]
# Convert to nucleotide positions (0‑based)
ANTIGENIC_NT_POS = []
for aa in ORF5_ANTIGENIC_AA:
    base = (aa - 1) * 3
    ANTIGENIC_NT_POS.extend([base, base + 1, base + 2])

# ---------- Helper functions ----------
def _read_upload_as_bytes(upload: UploadFile) -> bytes:
    try:
        upload.file.seek(0)
    except Exception:
        pass
    return upload.file.read() or b""

def _bytes_to_text(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin-1", errors="replace")

def _read_csv_from_bytes(b: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b))

def _read_json_from_bytes(b: bytes) -> dict:
    return json.loads(_bytes_to_text(b))

def _entropy(seq: str, base: str = "ACGT") -> float:
    """Shannon entropy of a nucleotide sequence."""
    seq = (seq or "").upper()
    seq = "".join([c for c in seq if c in "ACGT"])
    if len(seq) == 0:
        return 0.0
    probs = np.array([seq.count(x) / len(seq) for x in base])
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())

def _kmer_freq(seq: str, k: int = 3) -> dict:
    seq = (seq or "").upper()
    seq = "".join([c for c in seq if c in "ACGT"])
    if len(seq) < k:
        return {}
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    c = Counter(kmers)
    total = sum(c.values()) or 1
    return {f"kmer_{k}_{kmer}": v / total for kmer, v in c.items()}

def save_uploaded_fasta(run_dir: str, fasta_bytes: bytes) -> str:
    """Save raw FASTA to run directory and return path."""
    fasta_path = os.path.join(run_dir, "sequences.fasta")
    with open(fasta_path, "wb") as f:
        f.write(fasta_bytes)
    return fasta_path

def load_sequences(run_id: str) -> list[SeqRecord]:
    """Load SeqRecords from the run's FASTA file."""
    fasta_path = os.path.join(OUTPUT_ROOT, run_id, "sequences.fasta")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"No sequences found for run {run_id}")
    return list(SeqIO.parse(fasta_path, "fasta"))

def align_sequences(records: list[SeqRecord]) -> MultipleSeqAlignment:
    """
    Simple alignment: assumes all sequences are already aligned (same length).
    In a real implementation you might use ClustalW etc.
    """
    lengths = {len(r.seq) for r in records}
    if len(lengths) != 1:
        raise ValueError("Sequences have different lengths. Alignment required.")
    return MultipleSeqAlignment(records)

# ---------- Feature extraction (enhanced) ----------
def wgs_to_features_from_bytes(fasta_bytes: bytes) -> pd.DataFrame:
    """Extract genomic features including per‑site entropy and dN/dS proxies."""
    fasta_text = _bytes_to_text(fasta_bytes)
    handle = io.StringIO(fasta_text)
    records = list(SeqIO.parse(handle, "fasta"))
    if not records:
        raise ValueError("No FASTA records.")

    sample_ids = [rec.id.strip() for rec in records]
    seqs = [str(rec.seq).upper() for rec in records]

    # Basic per‑sequence stats
    rows = []
    for sid, seq in zip(sample_ids, seqs):
        L = len(seq)
        gc = (seq.count("G") + seq.count("C")) / (L or 1)
        ent = _entropy(seq)
        km = _kmer_freq(seq, k=3)
        row = {
            "sample_id": sid,
            "genome_len": float(L),
            "gc_content": float(gc),
            "seq_entropy": float(ent),
        }
        row.update(km)
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)

    # If sequences are aligned (same length), compute site‑level features
    if len({len(s) for s in seqs}) == 1:
        seq_len = len(seqs[0])
        # Site‑wise Shannon entropy
        site_entropies = []
        for i in range(seq_len):
            col = [s[i] for s in seqs if s[i] in "ACGT"]
            if len(col) == 0:
                site_entropies.append(0.0)
            else:
                counts = Counter(col)
                probs = np.array(list(counts.values())) / len(col)
                site_entropies.append(float(-(probs * np.log2(probs)).sum()))
        df["mean_site_entropy"] = np.mean(site_entropies)
        df["max_site_entropy"] = np.max(site_entropies)

        # Antigenic divergence from consensus
        if ANTIGENIC_NT_POS:
            consensus = []
            for pos in ANTIGENIC_NT_POS:
                if pos < seq_len:
                    col = [s[pos] for s in seqs if s[pos] in "ACGT"]
                    if col:
                        consensus.append(Counter(col).most_common(1)[0][0])
                    else:
                        consensus.append("N")
                else:
                    consensus.append("N")
            ant_scores = []
            for s in seqs:
                mismatches = sum(1 for i, p in enumerate(ANTIGENIC_NT_POS)
                                 if p < len(s) and s[p] in "ACGT" and s[p] != consensus[i])
                ant_scores.append(mismatches / len(ANTIGENIC_NT_POS))
            df["antigenic_divergence"] = ant_scores
        else:
            df["antigenic_divergence"] = 0.0

        # Epistatic novelty (simple version: co‑occurrence of rare pairs)
        variable_positions = [i for i in range(seq_len) if len(set(s[i] for s in seqs)) > 1]
        if len(variable_positions) >= 10:
            # Use top 50 most variable sites
            top_var = sorted(variable_positions, key=lambda i: site_entropies[i], reverse=True)[:50]
            sub_mat = np.array([[s[i] for i in top_var] for s in seqs])
            n_sites = len(top_var)
            rare_threshold = max(1, int(0.05 * len(sample_ids)))
            novelty = []
            for i in range(len(sample_ids)):
                rare_pairs = 0
                total_pairs = 0
                for a in range(n_sites):
                    for b in range(a + 1, n_sites):
                        total_pairs += 1
                        pair = (sub_mat[i, a], sub_mat[i, b])
                        freq = sum(1 for j in range(len(sample_ids)) if (sub_mat[j, a], sub_mat[j, b]) == pair)
                        if freq <= rare_threshold:
                            rare_pairs += 1
                novelty.append(rare_pairs / (total_pairs or 1))
            df["epistatic_novelty"] = novelty
        else:
            df["epistatic_novelty"] = 0.0
    else:
        df["mean_site_entropy"] = 0.0
        df["max_site_entropy"] = 0.0
        df["antigenic_divergence"] = 0.0
        df["epistatic_novelty"] = 0.0

    # Remove duplicate sample_ids (keep first)
    if df["sample_id"].duplicated().any():
        df = df.groupby("sample_id").first().reset_index()

    return df

def epi_to_features_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
    """Parse epidemiology CSV into features, including geospatial if lat/lon present."""
    df = _read_csv_from_bytes(csv_bytes)
    if "sample_id" not in df.columns:
        raise ValueError("Epidemiology CSV must have 'sample_id' column.")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    # Binary flags
    if "vaccinated" in df.columns:
        df["vaccinated_bin"] = df["vaccinated"].astype(str).str.lower().isin(["yes", "1", "true", "y"]).astype(int)
    else:
        df["vaccinated_bin"] = 0

    if "homologous_or_heterologous" in df.columns:
        df["heterologous_bin"] = df["homologous_or_heterologous"].astype(str).str.lower().str.contains("hetero", na=False).astype(int)
    else:
        df["heterologous_bin"] = 0

    # Normalize numeric fields
    for c in ["age_at_vaccination", "age_at_challenge", "challenge_dose"]:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            mx = col.max() if col.max() > 0 else 1.0
            df[c + "_norm"] = col / mx

    # One‑hot vaccine type
    if "vaccine_type" in df.columns:
        dummies = pd.get_dummies(df["vaccine_type"].astype(str).fillna("NA"), prefix="vax")
        df = pd.concat([df, dummies], axis=1)

    # --- Geospatial processing (integrated) ---
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        if not df.empty:
            # Kernel density estimation (haversine)
            coords = df[["lat", "lon"]].values
            if len(coords) > 3:
                kde = KernelDensity(bandwidth=0.5, metric='haversine')
                coords_rad = np.radians(coords)
                kde.fit(coords_rad)
                log_dens = kde.score_samples(coords_rad)
                dens = np.exp(log_dens)
                dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
                df["spatial_density"] = dens
            else:
                df["spatial_density"] = 0.0

            # Scale coordinates
            df["lat_scaled"] = (df["lat"] - df["lat"].mean()) / (df["lat"].std() if df["lat"].std() > 0 else 1.0)
            df["lon_scaled"] = (df["lon"] - df["lon"].mean()) / (df["lon"].std() if df["lon"].std() > 0 else 1.0)
        else:
            # No valid coordinates; add placeholder columns
            df["lat_scaled"] = 0.0
            df["lon_scaled"] = 0.0
            df["spatial_density"] = 0.0
            df["lat"] = 0.0
            df["lon"] = 0.0
    else:
        # If lat/lon missing, add zero columns for compatibility
        df["lat_scaled"] = 0.0
        df["lon_scaled"] = 0.0
        df["spatial_density"] = 0.0
        df["lat"] = 0.0
        df["lon"] = 0.0

    # Keep relevant columns
    keep_cols = ["sample_id", "farm_id", "collection_date", "lat", "lon", "lat_scaled", "lon_scaled", "spatial_density"] + \
                [c for c in df.columns if c.endswith("_bin") or c.endswith("_norm") or c.startswith("vax_")]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy().fillna(0.0)

    # Deduplicate sample_id
    if df["sample_id"].duplicated().any():
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        first = df.groupby("sample_id").first().reset_index()
        avg = df.groupby("sample_id")[numeric_cols].mean().reset_index()
        df = first.merge(avg, on="sample_id", how="left", suffixes=("", "_avg"))
        # drop duplicate columns if any
        df = df.loc[:, ~df.columns.str.endswith("_avg")]

    return df

def merge_features(wgs_df: pd.DataFrame, epi_df: pd.DataFrame) -> pd.DataFrame:
    df = wgs_df.merge(epi_df, on="sample_id", how="inner")
    if df.empty:
        raise ValueError("No matching sample_id across files. Check that FASTA headers match CSV identifiers.")
    return df

def basic_qc(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_samples": int(df["sample_id"].nunique()),
        "missing_cells": int(df.isna().sum().sum()),
    }

def fit_pca(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, PCA, PCA]:
    X = df.select_dtypes(include=["number"]).drop(columns=["genome_len"], errors="ignore").fillna(0.0)
    if X.shape[0] < 3:
        raise ValueError("Need at least 3 samples for PCA.")
    Xs = StandardScaler().fit_transform(X.values)
    pca2 = PCA(n_components=2, random_state=42)
    emb2d = pca2.fit_transform(Xs)
    pca3 = PCA(n_components=3, random_state=42)
    emb3d = pca3.fit_transform(Xs)
    return emb2d, emb3d, pca2, pca3

def compute_risk_score(df: pd.DataFrame, emb3d: np.ndarray) -> np.ndarray:
    n = len(df)
    entropy = df.get("seq_entropy", pd.Series([0.0] * n)).to_numpy()
    antigenic = df.get("antigenic_divergence", pd.Series([0.0] * n)).to_numpy()
    epistatic = df.get("epistatic_novelty", pd.Series([0.0] * n)).to_numpy()
    spatial = df.get("spatial_density", pd.Series([0.0] * n)).to_numpy()
    vacc = df.get("vaccinated_bin", pd.Series([0.0] * n)).to_numpy()
    hetero = df.get("heterologous_bin", pd.Series([0.0] * n)).to_numpy()

    center = emb3d.mean(axis=0)
    novelty = np.linalg.norm(emb3d - center, axis=1)

    def norm01(x):
        x = np.asarray(x, dtype=float)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    score = (0.20 * norm01(entropy) +
             0.15 * norm01(antigenic) +
             0.15 * norm01(epistatic) +
             0.10 * norm01(spatial) +
             0.20 * norm01(novelty) +
             0.10 * (1 - norm01(vacc)) +
             0.10 * norm01(hetero))
    return score

def save_outputs(run_id: str, df: pd.DataFrame, emb2d: np.ndarray, emb3d: np.ndarray) -> str:
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_df = df.copy()
    out_df["pca_x"] = emb2d[:, 0]
    out_df["pca_y"] = emb2d[:, 1]
    out_df["pca3_x"] = emb3d[:, 0]
    out_df["pca3_y"] = emb3d[:, 1]
    out_df["pca3_z"] = emb3d[:, 2]
    out_df["risk_score"] = compute_risk_score(df, emb3d)
    csv_path = os.path.join(out_dir, "results.csv")
    out_df.to_csv(csv_path, index=False)
    features_path = os.path.join(out_dir, "features.csv")
    df.to_csv(features_path, index=False)
    return csv_path

def load_run_data(run_id: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_ROOT, run_id, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run {run_id} not found.")
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df

# ---------- Advanced analysis functions ----------
def compute_shannon_entropy(run_id: str) -> dict:
    """Per‑site and per‑sequence Shannon entropy."""
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "Shannon Entropy", "error": "No sequences found."}
    # Assume aligned
    seq_len = len(seqs[0])
    site_entropies = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            site_entropies.append(0.0)
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            site_entropies.append(float(-(probs * np.log2(probs)).sum()))
    # Summary
    df_site = pd.DataFrame({"position": list(range(1, seq_len + 1)), "entropy": site_entropies})
    # Plot
    fig = px.line(df_site, x="position", y="entropy", title="Site‑wise Shannon Entropy")
    # Text: top 10 variable sites
    top10 = df_site.nlargest(10, "entropy").to_html(index=False)
    return {
        "title": "Shannon Entropy",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": f"<h4>Top 10 most variable sites</h4>{top10}",
        "download_text": df_site.to_csv(index=False),
    }

def compute_dnds(run_id: str) -> dict:
    """
    Approximate dN/dS per sequence compared to consensus.
    Uses a simple Nei‑Gojobori method on ORF5 (assumed to be the provided sequences).
    """
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "dN/dS", "error": "No sequences."}
    # Ensure coding (length multiple of 3)
    seq_len = len(seqs[0])
    if seq_len % 3 != 0:
        return {"title": "dN/dS", "error": "Sequence length not a multiple of 3 – cannot compute codon‑based dN/dS."}
    # Build consensus nucleotide (most frequent per site)
    consensus = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if col:
            consensus.append(Counter(col).most_common(1)[0][0])
        else:
            consensus.append("N")
    cons_seq = "".join(consensus)

    results = []
    for sid, seq in zip([r.id for r in records], seqs):
        # Count synonymous and non‑synonymous differences per codon
        n_syn = 0
        n_nonsyn = 0
        n_sites_syn = 0  # total possible synonymous sites
        n_sites_nonsyn = 0
        for j in range(0, seq_len, 3):
            codon_seq = seq[j:j+3]
            codon_cons = cons_seq[j:j+3]
            if "N" in codon_seq or "N" in codon_cons:
                continue
            # Translate
            try:
                aa_seq = Seq(codon_seq).translate()
                aa_cons = Seq(codon_cons).translate()
            except:
                continue
            if aa_seq == aa_cons:
                continue  # no difference
            # Count differences per codon – crude: if amino acids differ, count 1 nonsyn, else 0 syn
            # For a proper method we would need to enumerate pathways.
            # Here we simply mark whole codon as nonsyn if amino acids differ.
            if aa_seq != aa_cons:
                n_nonsyn += 1
                n_sites_nonsyn += 1
            else:
                n_syn += 1
                n_sites_syn += 1
        # Avoid division by zero
        pN = n_nonsyn / (n_sites_nonsyn if n_sites_nonsyn else 1)
        pS = n_syn / (n_sites_syn if n_sites_syn else 1)
        # Jukes‑Cantor correction
        dN = -3/4 * np.log(1 - 4/3 * pN) if pN < 0.75 else np.nan
        dS = -3/4 * np.log(1 - 4/3 * pS) if pS < 0.75 else np.nan
        dnds = dN / dS if dS and dS > 0 else np.nan
        results.append({
            "sample_id": sid,
            "pN": pN,
            "pS": pS,
            "dN": dN,
            "dS": dS,
            "dN/dS": dnds
        })
    df = pd.DataFrame(results)
    # Plot
    fig = px.bar(df, x="sample_id", y="dN/dS", title="Per‑sequence dN/dS ratio (vs consensus)")
    text = df.to_html(index=False)
    return {
        "title": "dN/dS Analysis",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": text,
        "download_text": df.to_csv(index=False)
    }

def analyze_epistatic_network(run_id: str) -> dict:
    """Mutual information based network of variable sites."""
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if len(seqs) < 5:
        return {"title": "Epistatic Network", "error": "Need at least 5 sequences."}
    seq_len = len(seqs[0])
    # Find variable positions
    var_pos = [i for i in range(seq_len) if len(set(s[i] for s in seqs)) > 1]
    if len(var_pos) < 2:
        return {"title": "Epistatic Network", "error": "Less than 2 variable positions."}
    # Subset to top 50 most variable to keep manageable
    site_entropy = []
    for i in var_pos:
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            ent = 0
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            ent = -sum(p * np.log2(p) for p in probs)
        site_entropy.append(ent)
    top_idx = np.argsort(site_entropy)[-50:]  # top 50
    selected_pos = [var_pos[i] for i in top_idx]
    # Build matrix of characters (simplify to 0/1 for each nucleotide? Use one‑hot? For MI we need discrete states)
    # We'll use the nucleotide as symbol.
    data = []
    for s in seqs:
        row = [s[i] for i in selected_pos]
        data.append(row)
    data = np.array(data)
    # Compute mutual information between pairs
    mi_matrix = np.zeros((len(selected_pos), len(selected_pos)))
    for i in range(len(selected_pos)):
        for j in range(i+1, len(selected_pos)):
            # discrete mutual information
            xi = data[:, i]
            xj = data[:, j]
            # build contingency table
            states_i = set(xi)
            states_j = set(xj)
            contingency = {}
            for si in states_i:
                for sj in states_j:
                    contingency[(si, sj)] = 0
            for a, b in zip(xi, xj):
                if a in "ACGT" and b in "ACGT":
                    contingency[(a, b)] += 1
            total = sum(contingency.values())
            if total == 0:
                mi = 0
            else:
                mi = 0
                for (si, sj), count in contingency.items():
                    pij = count / total
                    pi = sum(contingency[(si, sk)] for sk in states_j) / total
                    pj = sum(contingency[(sk, sj)] for sk in states_i) / total
                    if pij > 0 and pi > 0 and pj > 0:
                        mi += pij * np.log2(pij / (pi * pj))
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    # Build graph (threshold)
    threshold = np.percentile(mi_matrix[mi_matrix > 0], 90) if np.any(mi_matrix > 0) else 0.1
    G = nx.Graph()
    for i, pos_i in enumerate(selected_pos):
        G.add_node(pos_i)
    for i in range(len(selected_pos)):
        for j in range(i+1, len(selected_pos)):
            if mi_matrix[i, j] > threshold:
                G.add_edge(selected_pos[i], selected_pos[j], weight=mi_matrix[i, j])
    # Plot with plotly (use networkx layout)
    if G.number_of_nodes() == 0:
        return {"title": "Epistatic Network", "error": "No edges above threshold."}
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    for u, v, w in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                      mode='lines', line=dict(width=w['weight']*5, color='#888'),
                                      hoverinfo='none'))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=[f"Pos {n}" for n in G.nodes()],
                            textposition="top center",
                            marker=dict(size=10, color='lightblue'))
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title="Epistatic Network (MI > 90th percentile)",
                                     showlegend=False,
                                     hovermode='closest'))
    return {
        "title": "Epistatic Network",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": f"<p>Network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.</p>",
        "download_text": pd.DataFrame([(u, v, w['weight']) for u, v, w in G.edges(data=True)],
                                      columns=["pos1", "pos2", "MI"]).to_csv(index=False)
    }

def analyze_individual_trajectories(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "seq_entropy" not in df.columns:
        return {"title": "Individual Trajectories", "error": "seq_entropy missing."}
    df_sorted = df.sort_values("seq_entropy").reset_index(drop=True)
    df_sorted["pseudo_time"] = np.arange(len(df_sorted)) / len(df_sorted)
    color_col = "vaccinated_bin" if "vaccinated_bin" in df.columns else None
    fig = px.scatter(df_sorted, x="pseudo_time", y="seq_entropy", color=color_col,
                     hover_data=["sample_id"], title="Individual Trajectories (pseudo‑time = entropy order)")
    return {
        "title": "Individual Trajectories",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Trajectories ordered by increasing sequence entropy.</p>"
    }

def analyze_farm_coupling(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "farm_id" not in df.columns:
        return {"title": "Farm-Level Coupling", "error": "farm_id column missing."}
    farm_stats = df.groupby("farm_id").agg(
        n_samples=("sample_id", "count"),
        mean_entropy=("seq_entropy", "mean"),
        mean_risk=("risk_score", "mean") if "risk_score" in df.columns else ("seq_entropy", lambda x: 0)
    ).reset_index()
    fig = px.bar(farm_stats, x="farm_id", y="mean_entropy", color="n_samples",
                 title="Farm‑Level Mean Entropy")
    return {
        "title": "Farm‑Level Coupling",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": farm_stats.to_html(index=False),
        "download_text": farm_stats.to_csv(index=False)
    }

def analyze_spatiotemporal_diffusion(run_id: str) -> dict:
    df = load_run_data(run_id)
    if not {"collection_date", "lat", "lon"}.issubset(df.columns):
        return {"title": "Spatio‑Temporal Diffusion", "error": "Requires collection_date, lat, lon."}
    df = df.dropna(subset=["collection_date", "lat", "lon"])
    if df.empty:
        return {"title": "Spatio‑Temporal Diffusion", "error": "No valid dates/locations."}
    df["date_str"] = pd.to_datetime(df["collection_date"]).dt.strftime("%Y-%m-%d")
    fig = px.scatter_geo(df, lat="lat", lon="lon", animation_frame="date_str",
                         color="risk_score" if "risk_score" in df.columns else None,
                         hover_name="sample_id", title="Spatio‑Temporal Diffusion")
    return {
        "title": "Spatio‑Temporal Diffusion",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": f"<p>{len(df)} samples with dates.</p>"
    }

def analyze_vaccine_escape(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "antigenic_divergence" not in df.columns or "vaccinated_bin" not in df.columns:
        return {"title": "Vaccine Escape Prediction", "error": "Missing antigenic_divergence or vaccinated_bin."}
    df["escape_score"] = df["antigenic_divergence"] * (1 - df["vaccinated_bin"])
    top_escape = df.nlargest(10, "escape_score")[["sample_id", "antigenic_divergence", "vaccinated_bin", "escape_score"]]
    fig = px.scatter(df, x="antigenic_divergence", y="risk_score" if "risk_score" in df.columns else "seq_entropy",
                     color="vaccinated_bin", hover_data=["sample_id"],
                     title="Vaccine Escape Candidates")
    return {
        "title": "Vaccine Escape Prediction",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": top_escape.to_html(index=False),
        "download_text": top_escape.to_csv(index=False)
    }

def analyze_early_warning(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Early Warning Signals", "error": "Requires collection_date."}
    df = df.dropna(subset=["collection_date"]).sort_values("collection_date")
    if "risk_score" not in df.columns:
        df["risk_score"] = compute_risk_score(df, np.zeros((len(df), 3)))
    window = max(3, len(df) // 10)
    if len(df) < window:
        return {"title": "Early Warning Signals", "error": "Insufficient data."}
    df["roll_var"] = df["risk_score"].rolling(window, min_periods=2).var()
    df["roll_acf"] = df["risk_score"].rolling(window, min_periods=2).apply(lambda x: x.autocorr() if len(x) > 2 else np.nan)
    fig = px.line(df, x="collection_date", y=["risk_score", "roll_var", "roll_acf"],
                  title="Early Warning Signals")
    return {
        "title": "Early Warning Signals",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Rolling variance and autocorrelation of risk score.</p>"
    }

def analyze_geospatial_clustering(run_id: str) -> dict:
    df = load_run_data(run_id)
    if not {"lat", "lon"}.issubset(df.columns):
        return {"title": "Geospatial Clustering", "error": "lat/lon missing."}
    coords = df[["lat", "lon"]].values
    if len(coords) < 3:
        return {"title": "Geospatial Clustering", "error": "Not enough points."}
    db = DBSCAN(eps=0.5, min_samples=2, metric='haversine')
    coords_rad = np.radians(coords)
    labels = db.fit_predict(coords_rad)
    df["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    fig = px.scatter_mapbox(df, lat="lat", lon="lon", color=labels.astype(str),
                            hover_name="sample_id", title=f"Geospatial Clusters (DBSCAN, {n_clusters} clusters)",
                            mapbox_style="open-street-map", zoom=5)
    cluster_counts = pd.Series(labels).value_counts().to_frame().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    return {
        "title": "Geospatial Clustering",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": cluster_counts.to_html(index=False),
        "download_text": cluster_counts.to_csv(index=False)
    }

def analyze_geospatial_regression(run_id: str) -> dict:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    df = load_run_data(run_id)
    if not {"lat", "lon"}.issubset(df.columns):
        return {"title": "Geospatial Regression", "error": "lat/lon missing."}
    if "risk_score" not in df.columns:
        df["risk_score"] = compute_risk_score(df, np.zeros((len(df), 3)))
    X = df[["lat", "lon"]].values
    y = df["risk_score"].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    reg = LinearRegression().fit(Xp, y)
    # create grid
    lat_range = np.linspace(df["lat"].min(), df["lat"].max(), 30)
    lon_range = np.linspace(df["lon"].min(), df["lon"].max(), 30)
    LATS, LONS = np.meshgrid(lat_range, lon_range)
    grid = np.column_stack([LATS.ravel(), LONS.ravel()])
    grid_poly = poly.transform(grid)
    risk_pred = reg.predict(grid_poly).reshape(LATS.shape)
    fig = go.Figure(data=[
        go.Contour(z=risk_pred, x=lon_range, y=lat_range, colorscale='Viridis', name="Predicted Risk"),
        go.Scatter(x=df["lon"], y=df["lat"], mode='markers',
                   marker=dict(color=df["risk_score"], colorscale='Viridis', showscale=False),
                   text=df["sample_id"], name="Samples")
    ])
    fig.update_layout(title="Geospatial Regression (risk ~ lat+lon)")
    # summary text
    coef_df = pd.DataFrame({"feature": poly.get_feature_names_out(["lat", "lon"]), "coefficient": reg.coef_})
    return {
        "title": "Geospatial Regression",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": coef_df.to_html(index=False),
        "download_text": coef_df.to_csv(index=False)
    }

def analyze_manifold_curvature(run_id: str) -> dict:
    df = load_run_data(run_id)
    res_path = os.path.join(OUTPUT_ROOT, run_id, "results.csv")
    if not os.path.exists(res_path):
        return {"title": "Manifold Curvature", "error": "Embeddings not found. Run main analysis first."}
    df_res = pd.read_csv(res_path)
    df = df.merge(df_res[["sample_id", "pca3_x", "pca3_y", "pca3_z"]], on="sample_id", how="left")
    if "pca3_x" not in df.columns:
        return {"title": "Manifold Curvature", "error": "PCA embeddings missing."}
    from sklearn.neighbors import NearestNeighbors
    coords = df[["pca3_x", "pca3_y", "pca3_z"]].values
    if len(coords) < 5:
        return {"title": "Manifold Curvature", "error": "Need at least 5 points."}
    nbrs = NearestNeighbors(n_neighbors=min(5, len(coords))).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    curvature = []
    for i, neigh in enumerate(indices):
        if len(neigh) < 3:
            curvature.append(0)
            continue
        local = coords[neigh]
        local_centered = local - local.mean(axis=0)
        U, S, Vt = np.linalg.svd(local_centered)
        if len(S) >= 3:
            curvature.append(S[2] / (S[0] + 1e-9))
        else:
            curvature.append(0)
    df["curvature"] = curvature
    fig = px.scatter_3d(df, x="pca3_x", y="pca3_y", z="pca3_z", color=curvature,
                        hover_name="sample_id", title="Manifold Curvature")
    return {
        "title": "Manifold Curvature",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": f"<p>Mean curvature: {np.mean(curvature):.4f}</p>"
    }

def analyze_immune_escape_pathways(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "antigenic_divergence" not in df.columns:
        return {"title": "Immune Escape Pathways", "error": "antigenic_divergence missing."}
    top = df.nlargest(10, "antigenic_divergence")[["sample_id", "antigenic_divergence", "risk_score"]]
    fig = px.bar(top, x="sample_id", y="antigenic_divergence", color="risk_score",
                 title="Top 10 Antigenic Divergence")
    return {
        "title": "Immune Escape Pathways",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": top.to_html(index=False),
        "download_text": top.to_csv(index=False)
    }

def analyze_recombination_hotspots(run_id: str) -> dict:
    # Placeholder: could use site entropy as proxy
    df = load_run_data(run_id)
    if "max_site_entropy" in df.columns:
        text = f"<p>Max site entropy: {df['max_site_entropy'].mean():.3f}</p>"
    else:
        text = "<p>Recombination hotspot detection requires full alignment.</p>"
    return {
        "title": "Recombination Hotspot Detection",
        "text": text
    }

def analyze_temporal_trend(run_id: str) -> dict:
    df = load_run_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Temporal Trend", "error": "collection_date missing."}
    df = df.dropna(subset=["collection_date"]).sort_values("collection_date")
    if "risk_score" not in df.columns:
        df["risk_score"] = compute_risk_score(df, np.zeros((len(df), 3)))
    fig = px.line(df, x="collection_date", y="risk_score", title="Risk Score Over Time")
    return {
        "title": "Temporal Trend",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Time series of risk score.</p>"
    }

# ---------- New Analysis Functions ----------
def analyze_mutation_heatmap(run_id: str) -> dict:
    """Heatmap of nucleotide frequencies per site."""
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "Mutation Heatmap", "error": "No sequences."}
    seq_len = len(seqs[0])
    # Build frequency matrix (nucleotides: A, C, G, T)
    freq_matrix = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if not col:
            freq_matrix.append([0, 0, 0, 0])
        else:
            a = col.count('A') / len(col)
            c = col.count('C') / len(col)
            g = col.count('G') / len(col)
            t = col.count('T') / len(col)
            freq_matrix.append([a, c, g, t])
    freq_matrix = np.array(freq_matrix).T  # shape (4, seq_len)
    # Plot heatmap
    fig = ff.create_annotated_heatmap(
        z=freq_matrix,
        x=[f"Pos {i+1}" for i in range(seq_len)],
        y=['A', 'C', 'G', 'T'],
        colorscale='Viridis',
        showscale=True
    )
    fig.update_layout(title="Nucleotide Frequency per Site")
    return {
        "title": "Mutation Heatmap",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Heatmap shows frequency of each nucleotide at each position.</p>"
    }

def analyze_correlation_matrix(run_id: str) -> dict:
    """Correlation matrix of numeric features."""
    df = load_run_data(run_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return {"title": "Correlation Matrix", "error": "Not enough numeric columns."}
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    return {
        "title": "Correlation Matrix",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Pearson correlation between numeric features.</p>",
        "download_text": corr.to_csv()
    }

def analyze_tsne(run_id: str) -> dict:
    """t-SNE projection of samples."""
    df = load_run_data(run_id)
    res_path = os.path.join(OUTPUT_ROOT, run_id, "results.csv")
    if not os.path.exists(res_path):
        return {"title": "t-SNE", "error": "Results not found. Run main analysis first."}
    df_res = pd.read_csv(res_path)
    df = df.merge(df_res[["sample_id", "risk_score"]], on="sample_id", how="left")
    # Use the same feature set as PCA
    X = df.select_dtypes(include=["number"]).drop(columns=["genome_len", "risk_score"], errors="ignore").fillna(0.0)
    if X.shape[0] < 5:
        return {"title": "t-SNE", "error": "Need at least 5 samples."}
    Xs = StandardScaler().fit_transform(X.values)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    emb = tsne.fit_transform(Xs)
    df_plot = pd.DataFrame({"tsne_x": emb[:, 0], "tsne_y": emb[:, 1], "risk_score": df["risk_score"]})
    fig = px.scatter(df_plot, x="tsne_x", y="tsne_y", color="risk_score", title="t-SNE Projection")
    return {
        "title": "t-SNE",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>t-Distributed Stochastic Neighbor Embedding.</p>"
    }

def analyze_risk_factors(run_id: str) -> dict:
    """Feature importance for risk score using Random Forest."""
    df = load_run_data(run_id)
    if "risk_score" not in df.columns:
        return {"title": "Risk Factor Analysis", "error": "risk_score not found. Run main analysis first."}
    # Select numeric features (excluding target and identifiers)
    exclude = ["sample_id", "farm_id", "collection_date", "risk_score", "pca_x", "pca_y", "pca3_x", "pca3_y", "pca3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 2:
        return {"title": "Risk Factor Analysis", "error": "Not enough features."}
    X = df[feature_cols].fillna(0.0)
    y = df["risk_score"]
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    fig = px.bar(imp_df, x="importance", y="feature", orientation='h', title="Feature Importance for Risk Score")
    return {
        "title": "Risk Factor Analysis",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": imp_df.to_html(index=False),
        "download_text": imp_df.to_csv(index=False)
    }

def analyze_temporal_clustering(run_id: str) -> dict:
    """Cluster samples by collection date and visualize."""
    df = load_run_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Temporal Clustering", "error": "collection_date missing."}
    df = df.dropna(subset=["collection_date"]).copy()
    df["date"] = pd.to_datetime(df["collection_date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    counts = df["year_month"].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, title="Sample Count per Month")
    return {
        "title": "Temporal Clustering",
        "plot": fig.to_html(full_html=False, include_plotlyjs=False),
        "text": "<p>Distribution of samples over time.</p>",
        "download_text": counts.reset_index().to_csv(index=False)
    }

# Map analysis names to functions (including new ones)
ANALYSES = {
    "shannon_entropy": compute_shannon_entropy,
    "dnds": compute_dnds,
    "epistatic_network": analyze_epistatic_network,
    "individual_trajectories": analyze_individual_trajectories,
    "farm_coupling": analyze_farm_coupling,
    "spatiotemporal_diffusion": analyze_spatiotemporal_diffusion,
    "vaccine_escape": analyze_vaccine_escape,
    "early_warning": analyze_early_warning,
    "geospatial_clustering": analyze_geospatial_clustering,
    "geospatial_regression": analyze_geospatial_regression,
    "manifold_curvature": analyze_manifold_curvature,
    "immune_escape_pathways": analyze_immune_escape_pathways,
    "recombination_hotspots": analyze_recombination_hotspots,
    "temporal_trend": analyze_temporal_trend,
    # New analyses
    "mutation_heatmap": analyze_mutation_heatmap,
    "correlation_matrix": analyze_correlation_matrix,
    "tsne": analyze_tsne,
    "risk_factors": analyze_risk_factors,
    "temporal_clustering": analyze_temporal_clustering,
}

# ---------- API Endpoints ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/download/{run_id}")
def download(run_id: str):
    csv_path = os.path.join(OUTPUT_ROOT, run_id, "results.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="results.csv not found")
    return FileResponse(csv_path, media_type="text/csv", filename=f"prrsv_results_{run_id}.csv")

@app.post("/run")
async def run_api(
    wgs_fasta: UploadFile = File(...),
    epi_csv: UploadFile = File(...),
):
    run_id = str(uuid.uuid4())[:8]
    try:
        wgs_b = _read_upload_as_bytes(wgs_fasta)
        epi_b = _read_upload_as_bytes(epi_csv)

        # Save raw FASTA for later analyses
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)
        save_uploaded_fasta(run_dir, wgs_b)

        wgs_df = wgs_to_features_from_bytes(wgs_b)
        epi_df = epi_to_features_from_bytes(epi_b)

        merged = merge_features(wgs_df, epi_df)
        qc = basic_qc(merged)

        emb2d, emb3d, pca2, pca3 = fit_pca(merged)
        risk = compute_risk_score(merged, emb3d)

        csv_path = save_outputs(run_id, merged, emb2d, emb3d)

        return JSONResponse({
            "status": "success",
            "run_id": run_id,
            "qc": qc,
            "n_samples": int(merged.shape[0]),
            "download_url": f"/download/{run_id}",
            "preview": merged.head(10).to_dict(orient="records"),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/run_ui", response_class=HTMLResponse)
async def run_ui(
    wgs_fasta: UploadFile = File(...),
    epi_csv: UploadFile = File(...),
):
    run_id = str(uuid.uuid4())[:8]
    try:
        wgs_b = _read_upload_as_bytes(wgs_fasta)
        epi_b = _read_upload_as_bytes(epi_csv)

        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)
        save_uploaded_fasta(run_dir, wgs_b)

        wgs_df = wgs_to_features_from_bytes(wgs_b)
        epi_df = epi_to_features_from_bytes(epi_b)

        merged = merge_features(wgs_df, epi_df)
        qc = basic_qc(merged)

        emb2d, emb3d, pca2, pca3 = fit_pca(merged)
        risk = compute_risk_score(merged, emb3d)

        save_outputs(run_id, merged, emb2d, emb3d)

        out = merged.copy()
        out["pca_x"] = emb2d[:, 0]
        out["pca_y"] = emb2d[:, 1]
        out["risk_score"] = risk
        fig1 = px.scatter(
            out,
            x="pca_x",
            y="pca_y",
            color="risk_score",
            hover_data=[c for c in out.columns if c not in ["pca_x", "pca_y"]],
            title="PCA embedding (2D) — color = risk score",
        )
        fig1_html = fig1.to_html(full_html=False, include_plotlyjs=False)

        # Additional plot: Risk score distribution
        fig2 = px.histogram(out, x="risk_score", nbins=20, title="Risk Score Distribution")
        fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)

        preview_df = out.head(12)
        return render_results_page(run_id, qc, preview_df, fig1_html, fig2_html)

    except Exception as e:
        return HTMLResponse(
            f"<h2>Run failed</h2><p><b>Error:</b> {str(e)}</p><p><a href='/'>Go back</a></p>",
            status_code=400,
        )

@app.get("/analyze/{run_id}/{analysis_name}")
async def analyze(run_id: str, analysis_name: str):
    if analysis_name not in ANALYSES:
        raise HTTPException(status_code=404, detail="Analysis not found")
    try:
        result = ANALYSES[analysis_name](run_id)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- HTML templates ----------
def render_results_page(run_id: str, qc: dict, preview_df: pd.DataFrame, fig1_html: str, fig2_html: str) -> str:
    qc_html = "<ul>" + "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in qc.items()]) + "</ul>"
    preview_html = preview_df.to_html(index=False, escape=True)
    download_url = f"/download/{run_id}"

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Results — {run_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; max-width: 1400px; background: #f9f9f9; }}
    .container {{ display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }}
    .card {{ background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 20px; }}
    h1, h2, h3 {{ color: #2c3e50; }}
    .button {{ display: inline-block; padding: 10px 18px; border-radius: 30px; background: #2c3e50; color: white; text-decoration: none; font-weight: 600; border: none; cursor: pointer; transition: 0.2s; }}
    .button:hover {{ background: #1e2b37; }}
    .analysis-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin: 20px 0; }}
    .analysis-btn {{ background: #ecf0f1; border: none; padding: 12px; border-radius: 30px; cursor: pointer; font-weight: 500; transition: 0.2s; }}
    .analysis-btn:hover {{ background: #d5dbdb; }}
    #analysis-output {{ background: white; border-radius: 16px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
    #loading {{ display: none; text-align: center; padding: 40px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f2f2f2; }}
  </style>
</head>
<body>
  <h1>PRRSV Predictive Evolution — Run {run_id}</h1>
  <div style="display: flex; gap: 20px;">
    <a class="button" href="{download_url}">Download results.csv</a>
    <a class="button" href="/" style="background: #7f8c8d;">New Run</a>
  </div>

  <div class="container">
    <div>
      <div class="card">
        <h3>QC Summary</h3>
        {qc_html}
      </div>
      <div class="card">
        <h3>Preview</h3>
        {preview_html}
      </div>
    </div>
    <div>
      <div class="card">
        <h3>PCA Embedding</h3>
        {fig1_html}
      </div>
      <div class="card">
        <h3>Risk Score Distribution</h3>
        {fig2_html}
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Advanced Analyses</h2>
    <div class="analysis-grid">
      <button class="analysis-btn" data-analysis="shannon_entropy">Shannon Entropy</button>
      <button class="analysis-btn" data-analysis="dnds">dN/dS</button>
      <button class="analysis-btn" data-analysis="epistatic_network">Epistatic Network</button>
      <button class="analysis-btn" data-analysis="individual_trajectories">Individual Trajectories</button>
      <button class="analysis-btn" data-analysis="farm_coupling">Farm Coupling</button>
      <button class="analysis-btn" data-analysis="spatiotemporal_diffusion">Spatio‑Temporal</button>
      <button class="analysis-btn" data-analysis="vaccine_escape">Vaccine Escape</button>
      <button class="analysis-btn" data-analysis="early_warning">Early Warning</button>
      <button class="analysis-btn" data-analysis="geospatial_clustering">Geospatial Clustering</button>
      <button class="analysis-btn" data-analysis="geospatial_regression">Geospatial Regression</button>
      <button class="analysis-btn" data-analysis="manifold_curvature">Manifold Curvature</button>
      <button class="analysis-btn" data-analysis="immune_escape_pathways">Immune Escape Pathways</button>
      <button class="analysis-btn" data-analysis="recombination_hotspots">Recombination Hotspots</button>
      <button class="analysis-btn" data-analysis="temporal_trend">Temporal Trend</button>
      <button class="analysis-btn" data-analysis="mutation_heatmap">Mutation Heatmap</button>
      <button class="analysis-btn" data-analysis="correlation_matrix">Correlation Matrix</button>
      <button class="analysis-btn" data-analysis="tsne">t-SNE</button>
      <button class="analysis-btn" data-analysis="risk_factors">Risk Factor Analysis</button>
      <button class="analysis-btn" data-analysis="temporal_clustering">Temporal Clustering</button>
    </div>
    <div id="loading">⏳ Running analysis...</div>
    <div id="analysis-output"></div>
  </div>

  <script>
    const runId = "{run_id}";
    const buttons = document.querySelectorAll('.analysis-btn');
    const outputDiv = document.getElementById('analysis-output');
    const loadingDiv = document.getElementById('loading');

    buttons.forEach(btn => {{
      btn.addEventListener('click', async () => {{
        const analysis = btn.dataset.analysis;
        outputDiv.innerHTML = '';
        loadingDiv.style.display = 'block';
        try {{
          const response = await fetch(`/analyze/${{runId}}/${{analysis}}`);
          const data = await response.json();
          loadingDiv.style.display = 'none';
          let html = `<h3>${{data.title || analysis}}</h3>`;
          if (data.error) {{
            html += `<p style="color:red;">Error: ${{data.error}}</p>`;
          }} else {{
            if (data.plot) html += data.plot;
            if (data.text) html += data.text;
            if (data.download_text) {{
              const blob = new Blob([data.download_text], {{type: 'text/csv'}});
              const url = URL.createObjectURL(blob);
              html += `<p><a href="${{url}}" download="${{analysis}}_results.csv">Download CSV</a></p>`;
            }}
          }}
          outputDiv.innerHTML = html;
        }} catch (error) {{
          loadingDiv.style.display = 'none';
          outputDiv.innerHTML = `<p style="color:red;">Error: ${{error}}</p>`;
        }}
      }});
    }});
  </script>
  <footer style="text-align: center; color: #7f8c8d; padding: 20px; margin-top: 30px;">
    Developed by Nahiduzzaman, URA, Department of Microbiology and Hygiene, BAU
  </footer>
</body>
</html>
    """

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PRRSV Predictive Evolution Platform</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f0f4f8; color: #2c3e50; }
    .hero { background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 60px 20px; text-align: center; }
    .hero h1 { font-size: 3em; margin: 0; }
    .hero p { font-size: 1.3em; opacity: 0.9; }
    .container { max-width: 1200px; margin: 40px auto; padding: 0 20px; }
    .card { background: white; border-radius: 24px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); margin-bottom: 30px; }
    .upload-area { border: 2px dashed #ccc; border-radius: 20px; padding: 30px; text-align: center; }
    .file-input { margin: 15px 0; }
    .button { background: #2a5298; color: white; border: none; padding: 14px 28px; border-radius: 40px; font-size: 1.1em; font-weight: 600; cursor: pointer; transition: 0.2s; }
    .button:hover { background: #1e3c72; }
    .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 40px; }
    .feature { background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
    .feature h3 { margin-top: 0; color: #2a5298; }
    footer { text-align: center; color: #7f8c8d; padding: 20px; margin-top: 30px; }
  </style>
</head>
<body>
  <div class="hero">
    <h1>PRRSV Predictive Evolution</h1>
    <p>Topology‑Aware Manifold Learning & Quantum‑Inspired Dynamics</p>
  </div>
  <div class="container">
    <div class="card">
      <h2>Start a New Analysis</h2>
      <p>Upload your data files (FASTA and epidemiology CSV). The CSV must contain <code>sample_id</code>, and may include <code>lat</code>, <code>lon</code> for geospatial analyses.</p>
      <div class="upload-area">
        <form id="upload-form" enctype="multipart/form-data" action="/run_ui" method="post">
          <div class="file-input">
            <label><strong>1. WGS / ORF5 FASTA</strong> <small>(.fasta, .fa)</small></label><br>
            <input type="file" name="wgs_fasta" accept=".fasta,.fa,.fna" required>
          </div>
          <div class="file-input">
            <label><strong>2. Epidemiology CSV</strong> <small>(must include sample_id, optionally lat/lon)</small></label><br>
            <input type="file" name="epi_csv" accept=".csv" required>
          </div>
          <button class="button" type="submit">Run Pipeline</button>
        </form>
      </div>
    </div>

    <h2 style="text-align: center;">Analytical Modules</h2>
    <div class="features">
      <div class="feature"><h3>Shannon Entropy</h3><p>Per‑site and per‑sequence diversity metrics.</p></div>
      <div class="feature"><h3>dN/dS</h3><p>Selection pressure on ORF5.</p></div>
      <div class="feature"><h3>Epistatic Networks</h3><p>Co‑evolution of genomic sites.</p></div>
      <div class="feature"><h3>Individual Trajectories</h3><p>Evolutionary paths under immune pressure.</p></div>
      <div class="feature"><h3>Farm Coupling</h3><p>Herd‑level evolutionary dynamics.</p></div>
      <div class="feature"><h3>Spatio‑Temporal Diffusion</h3><p>Geographic spread and risk hotspots.</p></div>
      <div class="feature"><h3>Vaccine Escape</h3><p>Prediction of immune‑escape variants.</p></div>
      <div class="feature"><h3>Early Warning</h3><p>Detect critical transitions.</p></div>
      <div class="feature"><h3>Manifold Curvature</h3><p>Topological constraints on evolution.</p></div>
      <div class="feature"><h3>Immune Escape Pathways</h3><p>Candidate escape mutations.</p></div>
    </div>
  </div>
  <footer>
    Developed by Nahiduzzaman, URA, Department of Microbiology and Hygiene, BAU
  </footer>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)