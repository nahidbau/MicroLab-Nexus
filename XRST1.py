from __future__ import annotations

import io
import json
import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Optional, List, Dict, Any
import base64

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Bio import SeqIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, entropy, chi2_contingency, gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors
import warnings

# UMAP import – handle any failure gracefully
try:
    from umap import UMAP

    UMAP_AVAILABLE = True
except Exception as e:
    UMAP_AVAILABLE = False


    class UMAP:
        def __init__(self, *args, **kwargs):
            raise ImportError("UMAP not available. Install umap-learn and numba.")


    print(f"Warning: UMAP import failed: {e}. Falling back to PCA.")

warnings.filterwarnings('ignore')

# Optional: for persistent homology
try:
    import gudhi as gd

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

# Optional: for VAE / neural ODE / LSTM
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
ORF5_ANTIGENIC_AA = [30, 31, 32, 33, 34, 35, 56, 57, 58, 59, 60, 79, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104, 130,
                     131, 132]
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


def _entropy(seq: str, base: str = "ACGT") -> float:
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
    fasta_path = os.path.join(run_dir, "sequences.fasta")
    with open(fasta_path, "wb") as f:
        f.write(fasta_bytes)
    return fasta_path


def load_sequences(run_id: str) -> list[SeqRecord]:
    fasta_path = os.path.join(OUTPUT_ROOT, run_id, "sequences.fasta")
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"No sequences found for run {run_id}")
    return list(SeqIO.parse(fasta_path, "fasta"))


def align_sequences(records: list[SeqRecord]) -> MultipleSeqAlignment:
    lengths = {len(r.seq) for r in records}
    if len(lengths) != 1:
        raise ValueError("Sequences have different lengths. Alignment required.")
    return MultipleSeqAlignment(records)


def load_run_data(run_id: str) -> pd.DataFrame:
    path = os.path.join(OUTPUT_ROOT, run_id, "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run {run_id} not found.")
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def load_merged_data(run_id: str) -> pd.DataFrame:
    """Load features.csv and merge with results.csv if exists, to get UMAP and risk_score."""
    df = load_run_data(run_id)
    res_path = os.path.join(OUTPUT_ROOT, run_id, "results.csv")
    if os.path.exists(res_path):
        df_res = pd.read_csv(res_path)
        # Keep only the columns we need from results (avoid duplicates)
        res_cols = ["sample_id", "umap_x", "umap_y", "umap3_x", "umap3_y", "umap3_z", "risk_score"]
        res_cols = [c for c in res_cols if c in df_res.columns]
        df = df.merge(df_res[res_cols], on="sample_id", how="left")
    return df


# ---------- Feature extraction ----------
def extract_features_from_alignment(records: list[SeqRecord]) -> pd.DataFrame:
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    if not seqs:
        return pd.DataFrame()
    L = len(seqs[0])
    n_samples = len(seqs)

    # Site entropy
    site_entropy = []
    for i in range(L):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            site_entropy.append(0.0)
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            site_entropy.append(entropy(probs, base=2))
    site_entropy = np.array(site_entropy)

    # Consensus
    consensus = []
    for i in range(L):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if col:
            consensus.append(Counter(col).most_common(1)[0][0])
        else:
            consensus.append("N")
    cons_seq = "".join(consensus)

    rows = []
    for sid, seq in zip(ids, seqs):
        gc = (seq.count("G") + seq.count("C")) / L if L > 0 else 0
        ent = _entropy(seq)
        dist_to_cons = sum(1 for a, b in zip(seq, cons_seq) if a in "ACGT" and b in "ACGT" and a != b) / L
        ant_mismatch = 0
        ant_total = 0
        for pos in ANTIGENIC_NT_POS:
            if pos < L and seq[pos] in "ACGT" and cons_seq[pos] in "ACGT":
                ant_total += 1
                if seq[pos] != cons_seq[pos]:
                    ant_mismatch += 1
        ant_div = ant_mismatch / ant_total if ant_total > 0 else 0.0
        km = _kmer_freq(seq, k=3)
        row = {
            "sample_id": sid,
            "gc_content": gc,
            "seq_entropy": ent,
            "dist_to_consensus": dist_to_cons,
            "antigenic_divergence": ant_div,
        }
        row.update(km)
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    df["mean_site_entropy"] = site_entropy.mean()
    df["max_site_entropy"] = site_entropy.max()
    df["n_variable_sites"] = np.sum(site_entropy > 0)

    # Epistatic novelty
    if n_samples >= 5 and L >= 10:
        top_var_idx = np.argsort(site_entropy)[-50:]
        sub_seqs = np.array([[seq[i] for i in top_var_idx] for seq in seqs])
        n_sites = len(top_var_idx)
        rare_threshold = max(2, int(0.05 * n_samples))
        novelty = []
        for i in range(n_samples):
            rare_pairs = 0
            total_pairs = 0
            for a in range(n_sites):
                for b in range(a + 1, n_sites):
                    total_pairs += 1
                    pair = (sub_seqs[i, a], sub_seqs[i, b])
                    freq = sum(1 for j in range(n_samples) if (sub_seqs[j, a], sub_seqs[j, b]) == pair)
                    if freq <= rare_threshold:
                        rare_pairs += 1
            novelty.append(rare_pairs / total_pairs if total_pairs > 0 else 0)
        df["epistatic_novelty"] = novelty
    else:
        df["epistatic_novelty"] = 0.0
    return df


def epi_to_features_from_bytes(csv_bytes: bytes) -> pd.DataFrame:
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
        df["heterologous_bin"] = df["homologous_or_heterologous"].astype(str).str.lower().str.contains("hetero",
                                                                                                       na=False).astype(
            int)
    else:
        df["heterologous_bin"] = 0

    # Normalize numeric
    for c in ["age_at_vaccination", "age_at_challenge", "challenge_dose"]:
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            mx = col.max() if col.max() > 0 else 1.0
            df[c + "_norm"] = col / mx

    # One-hot vaccine type
    if "vaccine_type" in df.columns:
        dummies = pd.get_dummies(df["vaccine_type"].astype(str).fillna("NA"), prefix="vax")
        df = pd.concat([df, dummies], axis=1)

    # Geospatial
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df = df.dropna(subset=["lat", "lon"])
        if not df.empty and len(df) > 3:
            coords = df[["lat", "lon"]].values
            kde = KernelDensity(bandwidth=0.5, metric='haversine')
            coords_rad = np.radians(coords)
            kde.fit(coords_rad)
            log_dens = kde.score_samples(coords_rad)
            dens = np.exp(log_dens)
            dens = (dens - dens.min()) / (dens.max() - dens.min() + 1e-9)
            df["spatial_density"] = dens
            df["lat_scaled"] = (df["lat"] - df["lat"].mean()) / (df["lat"].std() if df["lat"].std() > 0 else 1.0)
            df["lon_scaled"] = (df["lon"] - df["lon"].mean()) / (df["lon"].std() if df["lon"].std() > 0 else 1.0)
        else:
            df["lat_scaled"] = df["lon_scaled"] = df["spatial_density"] = 0.0
    else:
        df["lat_scaled"] = df["lon_scaled"] = df["spatial_density"] = 0.0
        df["lat"] = df["lon"] = 0.0

    keep_cols = ["sample_id", "farm_id", "collection_date", "lat", "lon", "lat_scaled", "lon_scaled",
                 "spatial_density"] + \
                [c for c in df.columns if c.endswith("_bin") or c.endswith("_norm") or c.startswith("vax_")]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy().fillna(0.0)

    # Deduplicate sample_id
    if df["sample_id"].duplicated().any():
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        first = df.groupby("sample_id").first().reset_index()
        avg = df.groupby("sample_id")[numeric_cols].mean().reset_index()
        df = first.merge(avg, on="sample_id", how="left", suffixes=("", "_avg"))
        df = df.loc[:, ~df.columns.str.endswith("_avg")]
    return df


def merge_features(genomic_df: pd.DataFrame, epi_df: pd.DataFrame) -> pd.DataFrame:
    df = genomic_df.merge(epi_df, on="sample_id", how="inner")
    if df.empty:
        raise ValueError("No matching sample_id across files.")
    return df


def basic_qc(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_samples": int(df["sample_id"].nunique()),
        "missing_cells": int(df.isna().sum().sum()),
    }


# ---------- Topology‑aware embedding ----------
def topology_aware_embedding(sequences: List[str], n_components=3):
    unique_chars = set(''.join(sequences))
    char_to_int = {ch: i for i, ch in enumerate(sorted(unique_chars))}
    X_encoded = []
    for s in sequences:
        vec = [char_to_int.get(c, 0) for c in s]
        X_encoded.append(vec)
    X_encoded = np.array(X_encoded)
    if UMAP_AVAILABLE:
        reducer = UMAP(n_components=n_components, metric='hamming', random_state=42)
        embedding = reducer.fit_transform(X_encoded)
        return embedding, reducer
    else:
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(X_encoded)
        return embedding, reducer


# ---------- Quantum-inspired dynamics (VAE) ----------
if TORCH_AVAILABLE:
    class VAE(nn.Module):
        def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.mu_layer(h), self.logvar_layer(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar


    class NeuralODE(nn.Module):
        def __init__(self, latent_dim, hidden_dim=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim)
            )

        def forward(self, t, z):
            return self.net(z)


    def train_vae(features, latent_dim=8, epochs=50):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = VAE(input_dim=X.shape[1], latent_dim=latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                recon, mu, logvar = model(x)
                recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.1 * kl_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return model, scaler


    def predict_evolution(vae_model, scaler, start_point, n_steps=10, ode=None):
        device = next(vae_model.parameters()).device
        z_mu, _ = vae_model.encode(torch.tensor(scaler.transform([start_point]), dtype=torch.float32).to(device))
        z = z_mu.detach().cpu().numpy().flatten()
        trajectory = [z]
        if ode is not None:
            dt = 0.1
            z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            for _ in range(n_steps):
                dz = ode.forward(0, z_t)
                z_t = z_t + dt * dz
                trajectory.append(z_t.detach().cpu().numpy().flatten())
        else:
            for _ in range(n_steps):
                z = z + np.random.randn(*z.shape) * 0.1
                trajectory.append(z)
        decoded = []
        for z_pt in trajectory:
            z_t = torch.tensor(z_pt, dtype=torch.float32, device=device).unsqueeze(0)
            recon = vae_model.decoder(z_t)
            decoded.append(scaler.inverse_transform(recon.detach().cpu().numpy()).flatten())
        return np.array(decoded)


# ---------- Risk scoring ----------
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


def save_outputs(run_id: str, df: pd.DataFrame, emb2d: np.ndarray, emb3d: np.ndarray, manifold_model=None) -> str:
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_df = df.copy()
    out_df["umap_x"] = emb2d[:, 0]
    out_df["umap_y"] = emb2d[:, 1]
    out_df["umap3_x"] = emb3d[:, 0]
    out_df["umap3_y"] = emb3d[:, 1]
    out_df["umap3_z"] = emb3d[:, 2]
    out_df["risk_score"] = compute_risk_score(df, emb3d)
    csv_path = os.path.join(out_dir, "results.csv")
    out_df.to_csv(csv_path, index=False)
    features_path = os.path.join(out_dir, "features.csv")
    df.to_csv(features_path, index=False)
    if manifold_model is not None:
        import joblib
        joblib.dump(manifold_model, os.path.join(out_dir, "umap_model.pkl"))
    return csv_path


# ---------- ANALYSIS FUNCTIONS (each returns dict with plot and text) ----------
def analysis_shannon_entropy(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "Shannon Entropy", "error": "No sequences found."}
    seq_len = len(seqs[0])
    site_entropies = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            site_entropies.append(0.0)
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            site_entropies.append(entropy(probs, base=2))
    df_site = pd.DataFrame({"position": list(range(1, seq_len + 1)), "entropy": site_entropies})
    plot_type = plot_params.get("plot_type", "line") if plot_params else "line"
    if plot_type == "line":
        fig = px.line(df_site, x="position", y="entropy", title="Site‑wise Shannon Entropy")
    elif plot_type == "bar":
        fig = px.bar(df_site, x="position", y="entropy", title="Site‑wise Shannon Entropy")
    elif plot_type == "heatmap":
        z = df_site["entropy"].values.reshape(1, -1)
        fig = ff.create_annotated_heatmap(z, x=list(df_site["position"]), y=["Entropy"], colorscale='Viridis')
    elif plot_type == "violin":
        fig = px.violin(df_site, y="entropy", title="Distribution of Site Entropy")
    elif plot_type == "box":
        fig = px.box(df_site, y="entropy", title="Boxplot of Site Entropy")
    elif plot_type == "area":
        fig = px.area(df_site, x="position", y="entropy", title="Site Entropy Area Chart")
    elif plot_type == "density_heatmap":
        z = np.tile(site_entropies, (10, 1))
        fig = px.imshow(z, aspect="auto", title="Site Entropy Density Heatmap",
                        labels=dict(x="Position", y="Replicate"))
    else:
        fig = px.line(df_site, x="position", y="entropy")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    top10 = df_site.nlargest(10, "entropy").to_html(index=False)
    return {
        "title": "Shannon Entropy",
        "figure": figure_json,
        "text": f"<h4>Top 10 most variable sites</h4>{top10}",
        "download_text": df_site.to_csv(index=False),
    }


def analysis_dnds(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "dN/dS", "error": "No sequences."}
    seq_len = len(seqs[0])
    if seq_len % 3 != 0:
        return {"title": "dN/dS", "error": "Sequence length not a multiple of 3."}
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
        n_syn = n_nonsyn = 0
        n_sites_syn = n_sites_nonsyn = 0
        for j in range(0, seq_len, 3):
            codon_seq = seq[j:j + 3]
            codon_cons = cons_seq[j:j + 3]
            if "N" in codon_seq or "N" in codon_cons:
                continue
            try:
                aa_seq = Seq(codon_seq).translate()
                aa_cons = Seq(codon_cons).translate()
            except:
                continue
            if aa_seq == aa_cons:
                n_syn += 1
                n_sites_syn += 1
            else:
                n_nonsyn += 1
                n_sites_nonsyn += 1
        pN = n_nonsyn / (n_sites_nonsyn if n_sites_nonsyn else 1)
        pS = n_syn / (n_sites_syn if n_sites_syn else 1)
        dN = -3 / 4 * np.log(1 - 4 / 3 * pN) if pN < 0.75 else np.nan
        dS = -3 / 4 * np.log(1 - 4 / 3 * pS) if pS < 0.75 else np.nan
        dnds = dN / dS if dS and dS > 0 else np.nan
        results.append({"sample_id": sid, "dN/dS": dnds, "pN": pN, "pS": pS})
    df = pd.DataFrame(results)
    plot_type = plot_params.get("plot_type", "bar") if plot_params else "bar"
    if plot_type == "bar":
        fig = px.bar(df, x="sample_id", y="dN/dS", title="Per‑sequence dN/dS ratio (vs consensus)")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="sample_id", y="dN/dS", title="dN/dS per sample")
    elif plot_type == "histogram":
        fig = px.histogram(df, x="dN/dS", title="Distribution of dN/dS")
    elif plot_type == "box":
        fig = px.box(df, y="dN/dS", title="Boxplot of dN/dS")
    elif plot_type == "violin":
        fig = px.violin(df, y="dN/dS", title="Violin plot of dN/dS")
    else:
        fig = px.bar(df, x="sample_id", y="dN/dS")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "dN/dS Analysis", "figure": figure_json, "text": df.to_html(index=False),
            "download_text": df.to_csv(index=False)}


def analysis_epistatic_network(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if len(seqs) < 5:
        return {"title": "Epistatic Network", "error": "Need at least 5 sequences."}
    seq_len = len(seqs[0])
    var_pos = [i for i in range(seq_len) if len(set(s[i] for s in seqs)) > 1]
    if len(var_pos) < 2:
        return {"title": "Epistatic Network", "error": "Less than 2 variable positions."}
    site_entropy = []
    for i in var_pos:
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            ent = 0
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            ent = entropy(probs, base=2)
        site_entropy.append(ent)
    top_idx = np.argsort(site_entropy)[-50:]
    selected_pos = [var_pos[i] for i in top_idx]
    mi_matrix = np.zeros((len(selected_pos), len(selected_pos)))
    for i, p_i in enumerate(selected_pos):
        for j, p_j in enumerate(selected_pos):
            if i >= j: continue
            vec_i = [s[p_i] for s in seqs]
            vec_j = [s[p_j] for s in seqs]
            mi = mutual_info_score(vec_i, vec_j)
            mi_matrix[i, j] = mi_matrix[j, i] = mi
    threshold = np.percentile(mi_matrix[mi_matrix > 0], 90) if np.any(mi_matrix > 0) else 0.1
    G = nx.Graph()
    for pos in selected_pos:
        G.add_node(pos)
    for i in range(len(selected_pos)):
        for j in range(i + 1, len(selected_pos)):
            if mi_matrix[i, j] > threshold:
                G.add_edge(selected_pos[i], selected_pos[j], weight=mi_matrix[i, j])
    if G.number_of_nodes() == 0:
        return {"title": "Epistatic Network", "error": "No edges above threshold."}
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    for u, v, w in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     mode='lines', line=dict(width=w['weight'] * 5, color='#888'),
                                     hoverinfo='none'))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=[f"Pos {n}" for n in G.nodes()],
                            textposition="top center",
                            marker=dict(size=10, color='lightblue'))
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title="Epistatic Network (MI > 90th percentile)",
                                     showlegend=False, hovermode='closest'))
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Epistatic Network", "figure": figure_json,
            "text": f"<p>Network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.</p>",
            "download_text": pd.DataFrame([(u, v, w['weight']) for u, v, w in G.edges(data=True)],
                                          columns=["pos1", "pos2", "MI"]).to_csv(index=False)}


def analysis_individual_trajectories(run_id: str, plot_params: dict = None) -> dict:
    # ------------------------------------------------------------------
    # 1. Load and validate data
    # ------------------------------------------------------------------
    try:
        df = load_merged_data(run_id)
    except FileNotFoundError:
        return {
            "title": "Individual Trajectories",
            "error": f"Run '{run_id}' not found. Please run the main pipeline first."
        }
    except Exception as e:
        return {
            "title": "Individual Trajectories",
            "error": f"Error loading data: {str(e)}"
        }

    if df.empty:
        return {"title": "Individual Trajectories", "error": "No data available for this run."}

    required_cols = ["sample_id", "seq_entropy"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return {
            "title": "Individual Trajectories",
            "error": f"Missing required columns: {', '.join(missing)}"
        }

    # ------------------------------------------------------------------
    # 2. Prepare dataframe: sort by entropy and create pseudo‑time
    # ------------------------------------------------------------------
    try:
        df_sorted = df.sort_values("seq_entropy").reset_index(drop=True)
        n = len(df_sorted)
        if n == 0:
            return {"title": "Individual Trajectories", "error": "No samples after sorting."}
        df_sorted["pseudo_time"] = np.linspace(0, 1, n)  # safe even if n == 1
    except Exception as e:
        return {"title": "Individual Trajectories", "error": f"Error preparing data: {str(e)}"}

    # ------------------------------------------------------------------
    # 3. Determine colour column (prefer risk_score, then vaccinated_bin)
    # ------------------------------------------------------------------
    color_col = None
    color_continuous_scale = None
    if "risk_score" in df_sorted.columns and df_sorted["risk_score"].notna().any():
        color_col = "risk_score"
        color_continuous_scale = "Viridis"
    elif "vaccinated_bin" in df_sorted.columns:
        color_col = "vaccinated_bin"
        # no continuous scale for categorical

    # ------------------------------------------------------------------
    # 4. Build hover data (include only columns that exist)
    # ------------------------------------------------------------------
    hover_data = ["sample_id", "seq_entropy"]
    for col in ["risk_score", "vaccinated_bin", "antigenic_divergence", "gc_content"]:
        if col in df_sorted.columns:
            hover_data.append(col)

    plot_type = plot_params.get("plot_type", "scatter") if plot_params else "scatter"

    # ------------------------------------------------------------------
    # 5. Create the plot
    # ------------------------------------------------------------------
    try:
        fig = None

        # ---- 3D trajectory (requires UMAP columns) ----
        if plot_type == "3d_scatter" and all(c in df_sorted.columns for c in ['umap3_x', 'umap3_y', 'umap3_z']):
            # Ensure no NaNs in coordinates
            coords_ok = df_sorted[['umap3_x', 'umap3_y', 'umap3_z']].notna().all(axis=1)
            if not coords_ok.any():
                return {"title": "Individual Trajectories", "error": "All UMAP 3D coordinates are NaN."}
            if coords_ok.sum() < 2:
                # Only one valid point – just show a marker, no line
                fig = px.scatter_3d(
                    df_sorted[coords_ok],
                    x='umap3_x', y='umap3_y', z='umap3_z',
                    color=color_col,
                    color_continuous_scale=color_continuous_scale,
                    hover_data=hover_data,
                    title="Individual Trajectory (single point)"
                )
            else:
                # Build custom figure with line + markers
                fig = go.Figure()

                # Line connecting points in pseudo‑time order (only if >1 point)
                line_df = df_sorted[coords_ok].sort_values("pseudo_time")
                fig.add_trace(go.Scatter3d(
                    x=line_df['umap3_x'],
                    y=line_df['umap3_y'],
                    z=line_df['umap3_z'],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.6)', width=3),
                    hoverinfo='none',
                    showlegend=False
                ))

                # Markers (coloured)
                marker_kwargs = {
                    'x': df_sorted['umap3_x'],
                    'y': df_sorted['umap3_y'],
                    'z': df_sorted['umap3_z'],
                    'mode': 'markers',
                    'text': df_sorted['sample_id'],
                    'hoverinfo': 'text',
                    'showlegend': False
                }
                if color_col:
                    marker_kwargs['marker'] = {
                        'color': df_sorted[color_col],
                        'colorscale': 'Viridis' if color_col == 'risk_score' else None,
                        'showscale': (color_col == 'risk_score'),
                        'size': 5
                    }
                else:
                    marker_kwargs['marker'] = {'color': 'blue', 'size': 5}
                fig.add_trace(go.Scatter3d(**marker_kwargs))

                fig.update_layout(title="Individual Trajectories on Manifold (3D)")

        # ---- 2D scatter (pseudo‑time vs entropy) ----
        elif plot_type == "scatter":
            fig = px.scatter(
                df_sorted,
                x='pseudo_time', y='seq_entropy',
                color=color_col,
                color_continuous_scale=color_continuous_scale,
                hover_data=hover_data,
                title="Individual Trajectories (pseudo‑time = entropy order)"
            )

        elif plot_type == "line":
            fig = px.line(
                df_sorted,
                x='pseudo_time', y='seq_entropy',
                color=color_col,
                hover_data=hover_data,
                title="Individual Trajectories (line)"
            )

        # ---- Fallback (simple scatter) ----
        else:
            fig = px.scatter(
                df_sorted,
                x='pseudo_time', y='seq_entropy',
                hover_data=hover_data,
                title="Individual Trajectories"
            )

        if fig is None:
            return {"title": "Individual Trajectories", "error": "Plot creation failed (no figure)."}

        # ------------------------------------------------------------------
        # 6. Apply user‑specified layout parameters (width, height, etc.)
        # ------------------------------------------------------------------
        if plot_params:
            apply_plot_params(fig, plot_params)

        figure_json = fig.to_dict()
        return {
            "title": "Individual Trajectories",
            "figure": figure_json,
            "text": f"<p>Trajectories ordered by increasing sequence entropy. {n} samples.</p>"
        }

    except Exception as e:
        # Catch any unexpected error during plot generation
        return {
            "title": "Individual Trajectories",
            "error": f"Error during plot generation: {str(e)}"
        }


def analysis_farm_coupling(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "farm_id" not in df.columns:
        return {"title": "Farm‑Level Coupling", "error": "farm_id column missing."}
    farm_stats = df.groupby("farm_id").agg(
        n_samples=("sample_id", "count"),
        mean_entropy=("seq_entropy", "mean"),
        mean_risk=("risk_score", "mean") if "risk_score" in df.columns else ("seq_entropy", lambda x: 0)
    ).reset_index()
    plot_type = plot_params.get("plot_type", "bar") if plot_params else "bar"
    if plot_type == "bar":
        fig = px.bar(farm_stats, x="farm_id", y="mean_entropy", color="n_samples",
                     title="Farm‑Level Mean Entropy")
    elif plot_type == "scatter":
        fig = px.scatter(farm_stats, x="farm_id", y="mean_entropy", size="n_samples",
                         title="Farm‑Level Mean Entropy")
    elif plot_type == "heatmap":
        pivot = farm_stats.set_index("farm_id")[["mean_entropy", "n_samples"]]
        fig = px.imshow(pivot.T, text_auto=True, aspect="auto", title="Farm Statistics Heatmap")
    else:
        fig = px.bar(farm_stats, x="farm_id", y="mean_entropy")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Farm‑Level Coupling", "figure": figure_json,
            "text": farm_stats.to_html(index=False), "download_text": farm_stats.to_csv(index=False)}


def analysis_spatiotemporal_diffusion(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if not {"collection_date", "lat", "lon"}.issubset(df.columns):
        return {"title": "Spatio‑Temporal Diffusion", "error": "Requires collection_date, lat, lon."}
    df = df.dropna(subset=["collection_date", "lat", "lon"])
    if df.empty:
        return {"title": "Spatio‑Temporal Diffusion", "error": "No valid dates/locations."}
    df["date_str"] = pd.to_datetime(df["collection_date"]).dt.strftime("%Y-%m-%d")
    mapbox_token = plot_params.get("mapbox_token", "") if plot_params else ""
    if mapbox_token:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", animation_frame="date_str",
                                color="risk_score" if "risk_score" in df.columns else None,
                                hover_name="sample_id", title="Spatio‑Temporal Diffusion",
                                mapbox_style="satellite-streets", zoom=5)
        fig.update_layout(mapbox_accesstoken=mapbox_token)
    else:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", animation_frame="date_str",
                                color="risk_score" if "risk_score" in df.columns else None,
                                hover_name="sample_id", title="Spatio‑Temporal Diffusion",
                                mapbox_style="open-street-map", zoom=5)
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Spatio‑Temporal Diffusion", "figure": figure_json,
            "text": f"<p>{len(df)} samples with dates.</p>"}


def analysis_geospatial_clustering(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
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
    mapbox_token = plot_params.get("mapbox_token", "") if plot_params else ""
    if mapbox_token:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color=labels.astype(str),
                                hover_name="sample_id", title=f"Geospatial Clusters (DBSCAN, {n_clusters} clusters)",
                                mapbox_style="satellite-streets", zoom=5)
        fig.update_layout(mapbox_accesstoken=mapbox_token)
    else:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color=labels.astype(str),
                                hover_name="sample_id", title=f"Geospatial Clusters (DBSCAN, {n_clusters} clusters)",
                                mapbox_style="open-street-map", zoom=5)
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    cluster_counts = pd.Series(labels).value_counts().to_frame().reset_index()
    cluster_counts.columns = ["cluster", "count"]
    return {"title": "Geospatial Clustering", "figure": figure_json,
            "text": cluster_counts.to_html(index=False), "download_text": cluster_counts.to_csv(index=False)}


def analysis_geospatial_regression(run_id: str, plot_params: dict = None) -> dict:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    df = load_merged_data(run_id)
    if not {"lat", "lon"}.issubset(df.columns):
        return {"title": "Geospatial Regression", "error": "lat/lon missing."}
    if "risk_score" not in df.columns:
        return {"title": "Geospatial Regression", "error": "risk_score not found. Run main analysis first."}
    X = df[["lat", "lon"]].values
    y = df["risk_score"].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xp = poly.fit_transform(X)
    reg = LinearRegression().fit(Xp, y)
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
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    coef_df = pd.DataFrame({"feature": poly.get_feature_names_out(["lat", "lon"]), "coefficient": reg.coef_})
    return {"title": "Geospatial Regression", "figure": figure_json,
            "text": coef_df.to_html(index=False), "download_text": coef_df.to_csv(index=False)}


def analysis_manifold_curvature(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if not {"umap3_x", "umap3_y", "umap3_z"}.issubset(df.columns):
        return {"title": "Manifold Curvature", "error": "UMAP 3D embeddings not found. Run main analysis first."}
    coords = df[["umap3_x", "umap3_y", "umap3_z"]].values
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
    plot_type = plot_params.get("plot_type", "3d_scatter") if plot_params else "3d_scatter"
    if plot_type == "3d_scatter":
        fig = px.scatter_3d(df, x="umap3_x", y="umap3_y", z="umap3_z", color=curvature,
                            hover_name="sample_id", title="Manifold Curvature")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="umap3_x", y="umap3_y", color=curvature,
                         hover_name="sample_id", title="Manifold Curvature (2D projection)")
    elif plot_type == "histogram":
        fig = px.histogram(df, x="curvature", title="Distribution of Curvature")
    else:
        fig = px.scatter_3d(df, x="umap3_x", y="umap3_y", z="umap3_z", color=curvature)
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Manifold Curvature", "figure": figure_json,
            "text": f"<p>Mean curvature: {np.mean(curvature):.4f}</p>"}


def analysis_immune_escape_pathways(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "antigenic_divergence" not in df.columns:
        return {"title": "Immune Escape Pathways", "error": "antigenic_divergence missing."}
    if "risk_score" not in df.columns:
        df["risk_score"] = 0  # fallback
    top = df.nlargest(10, "antigenic_divergence")[["sample_id", "antigenic_divergence", "risk_score"]]
    plot_type = plot_params.get("plot_type", "bar") if plot_params else "bar"
    if plot_type == "bar":
        fig = px.bar(top, x="sample_id", y="antigenic_divergence", color="risk_score",
                     title="Top 10 Antigenic Divergence")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="antigenic_divergence", y="risk_score", hover_data=["sample_id"],
                         title="Antigenic Divergence vs Risk")
    elif plot_type == "3d_scatter" and all(c in df.columns for c in ['umap3_x', 'umap3_y', 'umap3_z']):
        fig = px.scatter_3d(df, x='umap3_x', y='umap3_y', z='umap3_z', color='antigenic_divergence',
                            hover_name='sample_id', title="Antigenic Divergence on Manifold")
    else:
        fig = px.bar(top, x="sample_id", y="antigenic_divergence")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Immune Escape Pathways", "figure": figure_json,
            "text": top.to_html(index=False), "download_text": top.to_csv(index=False)}


def analysis_recombination_hotspots(run_id: str, plot_params: dict = None) -> dict:
    # Placeholder – could be implemented with more sophisticated methods
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if len(seqs) < 10:
        return {"title": "Recombination Hotspot Detection", "error": "Need at least 10 sequences."}
    seq_len = len(seqs[0])
    # Simple entropy-based proxy
    site_entropy = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if len(col) == 0:
            site_entropy.append(0.0)
        else:
            counts = Counter(col)
            probs = np.array(list(counts.values())) / len(col)
            site_entropy.append(entropy(probs, base=2))
    df_site = pd.DataFrame({"position": list(range(1, seq_len + 1)), "entropy": site_entropy})
    # Highlight positions with entropy > 90th percentile as potential hotspots
    threshold = np.percentile(site_entropy, 90)
    hotspots = df_site[df_site["entropy"] > threshold]
    fig = px.scatter(df_site, x="position", y="entropy", title="Site Entropy (potential recombination hotspots)")
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"90th percentile ({threshold:.2f})")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Recombination Hotspot Detection (Entropy-based)", "figure": figure_json,
            "text": f"<p>{len(hotspots)} sites above 90th percentile entropy.</p>{hotspots.to_html(index=False)}",
            "download_text": hotspots.to_csv(index=False)}


def analysis_temporal_trend(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Temporal Trend", "error": "collection_date missing."}
    df = df.dropna(subset=["collection_date"]).sort_values("collection_date")
    if "risk_score" not in df.columns:
        return {"title": "Temporal Trend", "error": "risk_score not found. Run main analysis first."}
    plot_type = plot_params.get("plot_type", "line") if plot_params else "line"
    if plot_type == "line":
        fig = px.line(df, x="collection_date", y="risk_score", title="Risk Score Over Time")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="collection_date", y="risk_score", title="Risk Score Over Time")
    elif plot_type == "bar":
        df_month = df.set_index("collection_date").resample('M')['risk_score'].mean().reset_index()
        fig = px.bar(df_month, x="collection_date", y="risk_score", title="Monthly Mean Risk Score")
    elif plot_type == "area":
        fig = px.area(df, x="collection_date", y="risk_score", title="Risk Score Area Chart")
    else:
        fig = px.line(df, x="collection_date", y="risk_score")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Temporal Trend", "figure": figure_json, "text": "<p>Time series of risk score.</p>"}


def analysis_mutation_heatmap(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "Mutation Heatmap", "error": "No sequences."}
    seq_len = len(seqs[0])
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
    freq_matrix = np.array(freq_matrix).T
    fig = ff.create_annotated_heatmap(
        z=freq_matrix,
        x=[f"Pos {i + 1}" for i in range(seq_len)],
        y=['A', 'C', 'G', 'T'],
        colorscale='Viridis', showscale=True
    )
    fig.update_layout(title="Nucleotide Frequency per Site")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Mutation Heatmap", "figure": figure_json,
            "text": "<p>Heatmap shows frequency of each nucleotide at each position.</p>"}


def analysis_correlation_matrix(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return {"title": "Correlation Matrix", "error": "Not enough numeric columns."}
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Correlation Matrix", "figure": figure_json,
            "text": "<p>Pearson correlation between numeric features.</p>",
            "download_text": corr.to_csv()}


def analysis_tsne(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "risk_score" not in df.columns:
        return {"title": "t-SNE", "error": "risk_score not found. Run main analysis first."}
    exclude = ["sample_id", "farm_id", "collection_date", "risk_score", "umap_x", "umap_y", "umap3_x", "umap3_y",
               "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 2:
        return {"title": "t-SNE", "error": "Not enough features."}
    X = df[feature_cols].fillna(0.0).values
    if X.shape[0] < 5:
        return {"title": "t-SNE", "error": "Need at least 5 samples."}
    Xs = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    emb = tsne.fit_transform(Xs)
    df_plot = pd.DataFrame({"tsne_x": emb[:, 0], "tsne_y": emb[:, 1], "risk_score": df["risk_score"]})
    fig = px.scatter(df_plot, x="tsne_x", y="tsne_y", color="risk_score", title="t-SNE Projection")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "t-SNE", "figure": figure_json, "text": "<p>t-Distributed Stochastic Neighbor Embedding.</p>"}


def analysis_risk_factors(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "risk_score" not in df.columns:
        return {"title": "Risk Factor Analysis", "error": "risk_score not found. Run main analysis first."}
    exclude = ["sample_id", "farm_id", "collection_date", "risk_score", "umap_x", "umap_y", "umap3_x", "umap3_y",
               "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 2:
        return {"title": "Risk Factor Analysis", "error": "Not enough features."}
    X = df[feature_cols].fillna(0.0)
    y = df["risk_score"]
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance",
                                                                                            ascending=False)
    fig = px.bar(imp_df, x="importance", y="feature", orientation='h', title="Feature Importance for Risk Score")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Risk Factor Analysis", "figure": figure_json,
            "text": imp_df.to_html(index=False), "download_text": imp_df.to_csv(index=False)}


def analysis_temporal_clustering(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Temporal Clustering", "error": "collection_date missing."}
    df = df.dropna(subset=["collection_date"]).copy()
    df["date"] = pd.to_datetime(df["collection_date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    counts = df["year_month"].value_counts().sort_index()
    fig = px.bar(x=counts.index, y=counts.values, title="Sample Count per Month")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Temporal Clustering", "figure": figure_json,
            "text": "<p>Distribution of samples over time.</p>",
            "download_text": counts.reset_index().to_csv(index=False)}


def analysis_quantum_dynamics(run_id: str, plot_params: dict = None) -> dict:
    if not TORCH_AVAILABLE:
        return {"title": "Quantum‑Inspired Dynamics", "error": "PyTorch not installed."}
    df = load_merged_data(run_id)
    exclude = ["sample_id", "farm_id", "collection_date", "umap_x", "umap_y", "umap3_x", "umap3_y", "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 3:
        return {"title": "Quantum‑Inspired Dynamics", "error": "Not enough numeric features."}
    X = df[feature_cols].fillna(0.0).values
    vae, scaler = train_vae(X, latent_dim=8, epochs=50)
    start = X[0]
    trajectory = predict_evolution(vae, scaler, start, n_steps=20)
    traj_df = pd.DataFrame(trajectory, columns=feature_cols)
    traj_df["step"] = np.arange(len(traj_df))
    if len(feature_cols) >= 3:
        fig = px.scatter_3d(traj_df, x=feature_cols[0], y=feature_cols[1], z=feature_cols[2],
                            text="step", title="Simulated Evolutionary Trajectory (VAE + ODE)")
    else:
        fig = px.line(traj_df, x="step", y=feature_cols[0], title="Simulated Trajectory")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Quantum‑Inspired Dynamics", "figure": figure_json,
            "text": "<p>Simulated evolutionary path using VAE and neural ODE.</p>"}


def analysis_phylogenetic_tree(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    if len(seqs) < 3:
        return {"title": "Phylogenetic Tree", "error": "Need at least 3 sequences."}

    def encode_seq(s):
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return np.array([mapping.get(c, 0) for c in s])

    X = np.array([encode_seq(s) for s in seqs])
    dist_mat = pairwise_distances(X, metric='hamming')
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(dist_mat, method='average')
    fig = ff.create_dendrogram(Z, labels=ids, orientation='bottom')
    fig.update_layout(title="Neighbor-Joining Tree (UPGMA)")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Phylogenetic Tree", "figure": figure_json,
            "text": "<p>UPGMA dendrogram based on Hamming distance.</p>"}


def analysis_selection_pressure_fel(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    seq_len = len(seqs[0])
    if seq_len % 3 != 0:
        return {"title": "Site-wise Selection Pressure", "error": "Sequence length not multiple of 3."}
    n_codons = seq_len // 3
    consensus = []
    for i in range(seq_len):
        col = [s[i] for s in seqs if s[i] in "ACGT"]
        if col:
            consensus.append(Counter(col).most_common(1)[0][0])
        else:
            consensus.append("N")
    cons_seq = "".join(consensus)
    results = []
    for codon_idx in range(n_codons):
        start = codon_idx * 3
        codon_cons = cons_seq[start:start + 3]
        if "N" in codon_cons:
            continue
        aa_cons = Seq(codon_cons).translate()
        n_syn = n_nonsyn = 0
        for seq in seqs:
            codon_seq = seq[start:start + 3]
            if "N" in codon_seq:
                continue
            aa_seq = Seq(codon_seq).translate()
            if aa_seq == aa_cons:
                n_syn += 1
            else:
                n_nonsyn += 1
        total = n_syn + n_nonsyn
        if total == 0:
            continue
        pN = n_nonsyn / total
        pS = n_syn / total
        dnds_site = pN / (pS + 1e-9)
        results.append({"codon": codon_idx + 1, "pN": pN, "pS": pS, "dN/dS": dnds_site})
    df = pd.DataFrame(results)
    if df.empty:
        return {"title": "Site-wise Selection Pressure", "error": "No variable codon sites."}
    plot_type = plot_params.get("plot_type", "bar") if plot_params else "bar"
    if plot_type == "bar":
        fig = px.bar(df, x="codon", y="dN/dS", title="Site-wise dN/dS (FEL approximation)")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="codon", y="dN/dS", title="Site-wise dN/dS")
    elif plot_type == "line":
        fig = px.line(df, x="codon", y="dN/dS", title="Site-wise dN/dS")
    elif plot_type == "heatmap":
        z = df["dN/dS"].values.reshape(1, -1)
        fig = ff.create_annotated_heatmap(z, x=list(df["codon"]), y=["dN/dS"], colorscale='Viridis')
    else:
        fig = px.bar(df, x="codon", y="dN/dS")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Site-wise Selection Pressure", "figure": figure_json,
            "text": df.to_html(index=False), "download_text": df.to_csv(index=False)}


def analysis_transmission_network(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    if len(seqs) < 2:
        return {"title": "Transmission Network", "error": "Need at least 2 sequences."}

    def encode_seq(s):
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return np.array([mapping.get(c, 0) for c in s])

    X = np.array([encode_seq(s) for s in seqs])
    dist_mat = pairwise_distances(X, metric='hamming')
    threshold = plot_params.get("dist_threshold", 0.05) if plot_params else 0.05
    G = nx.Graph()
    for i, sid in enumerate(ids):
        G.add_node(sid)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if dist_mat[i, j] < threshold:
                G.add_edge(ids[i], ids[j], weight=1 - dist_mat[i, j])
    if G.number_of_edges() == 0:
        return {"title": "Transmission Network", "error": "No edges below threshold."}
    pos = nx.spring_layout(G, seed=42)
    edge_trace = []
    for u, v, w in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     mode='lines', line=dict(width=w['weight'] * 5, color='#888'),
                                     hoverinfo='none'))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=[n for n in G.nodes()],
                            textposition="top center",
                            marker=dict(size=10, color='lightblue'))
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(title=f"Transmission Network (dist < {threshold})",
                                     showlegend=False, hovermode='closest'))
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Transmission Network", "figure": figure_json,
            "text": f"<p>Network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.</p>"}


def analysis_antigenic_variation(run_id: str, plot_params: dict = None) -> dict:
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    if not seqs:
        return {"title": "Antigenic Variation", "error": "No sequences."}
    seq_len = len(seqs[0])
    antigenic_positions = [p for p in ANTIGENIC_NT_POS if p < seq_len]
    if not antigenic_positions:
        return {"title": "Antigenic Variation", "error": "No antigenic positions in range."}
    freq_data = []
    for pos in antigenic_positions:
        col = [s[pos] for s in seqs if s[pos] in "ACGT"]
        if not col:
            continue
        counts = Counter(col)
        total = len(col)
        for nt in "ACGT":
            freq_data.append({"position": pos + 1, "nucleotide": nt, "frequency": counts.get(nt, 0) / total})
    df_freq = pd.DataFrame(freq_data)
    if df_freq.empty:
        return {"title": "Antigenic Variation", "error": "No variable antigenic sites."}
    pivot = df_freq.pivot(index="nucleotide", columns="position", values="frequency").fillna(0)
    fig = ff.create_annotated_heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                                      colorscale='Viridis', showscale=True)
    fig.update_layout(title="Nucleotide Frequencies at Antigenic Sites")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Antigenic Variation", "figure": figure_json,
            "text": "<p>Heatmap of nucleotide frequencies at antigenic positions.</p>",
            "download_text": df_freq.to_csv(index=False)}


def analysis_machine_learning_prediction(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    target = None
    if "outcome" in df.columns:
        target = "outcome"
    elif "vaccine_efficacy" in df.columns:
        target = "vaccine_efficacy"
    elif "risk_score" in df.columns:
        target = "risk_score"
    else:
        return {"title": "ML Prediction", "error": "No suitable target column found (outcome, vaccine_efficacy, risk_score)."}
    exclude = ["sample_id", "farm_id", "collection_date", target, "umap_x", "umap_y", "umap3_x", "umap3_y", "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 2:
        return {"title": "ML Prediction", "error": "Not enough features."}
    X = df[feature_cols].fillna(0.0)
    y = df[target].fillna(0.0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    df_plot = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    fig = px.scatter(df_plot, x="actual", y="predicted", trendline="ols", title=f"Predicted vs Actual ({target})")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values("importance",
                                                                                                        ascending=False)
    return {"title": "Machine Learning Prediction", "figure": figure_json,
            "text": imp_df.to_html(index=False), "download_text": imp_df.to_csv(index=False)}


def analysis_pca(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    exclude = ["sample_id", "farm_id", "collection_date", "risk_score", "umap_x", "umap_y", "umap3_x", "umap3_y",
               "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 2:
        return {"title": "PCA", "error": "Not enough numeric features."}
    X = df[feature_cols].fillna(0.0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(5, X.shape[1]))
    components = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_
    plot_type = plot_params.get("plot_type", "scatter") if plot_params else "scatter"
    if plot_type == "scatter":
        df_pca = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1],
                               "risk_score": df["risk_score"] if "risk_score" in df.columns else 0,
                               "sample_id": df["sample_id"]})
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="risk_score", hover_name="sample_id",
                         title="PCA (first two components)")
    elif plot_type == "3d_scatter" and components.shape[1] >= 3:
        df_pca = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1], "PC3": components[:, 2],
                               "risk_score": df["risk_score"] if "risk_score" in df.columns else 0,
                               "sample_id": df["sample_id"]})
        fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color="risk_score", hover_name="sample_id",
                            title="PCA (first three components)")
    elif plot_type == "biplot":
        df_pca = pd.DataFrame(components[:, :2], columns=["PC1", "PC2"])
        df_pca["sample_id"] = df["sample_id"]
        loadings = pca.components_[:2, :].T
        fig = px.scatter(df_pca, x="PC1", y="PC2", hover_name="sample_id", title="PCA Biplot")
        for i, feature in enumerate(feature_cols):
            fig.add_annotation(x=loadings[i, 0] * max(components[:, 0]),
                               y=loadings[i, 1] * max(components[:, 1]),
                               text=feature, showarrow=True, arrowhead=2)
    elif plot_type == "scree":
        fig = px.bar(x=list(range(1, len(explained_var) + 1)), y=explained_var,
                     labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
                     title="Scree Plot")
    elif plot_type == "heatmap":
        loadings_df = pd.DataFrame(pca.components_[:min(5, pca.n_components_)],
                                   columns=feature_cols,
                                   index=[f"PC{i + 1}" for i in range(min(5, pca.n_components_))])
        fig = px.imshow(loadings_df, text_auto=True, aspect="auto", title="PCA Loadings")
    else:
        fig = px.scatter(x=components[:, 0], y=components[:, 1])
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    comp_df = pd.DataFrame(components, columns=[f"PC{i + 1}" for i in range(components.shape[1])])
    return {"title": "Principal Component Analysis", "figure": figure_json,
            "text": f"<p>Explained variance: {explained_var}</p>",
            "download_text": comp_df.to_csv(index=False)}


def analysis_vaccine_escape(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "antigenic_divergence" not in df.columns or "vaccinated_bin" not in df.columns:
        return {"title": "Vaccine Escape Prediction", "error": "Missing antigenic_divergence or vaccinated_bin."}
    df["escape_score"] = df["antigenic_divergence"] * (1 - df["vaccinated_bin"])
    top_escape = df.nlargest(10, "escape_score")[
        ["sample_id", "antigenic_divergence", "vaccinated_bin", "escape_score"]]
    plot_type = plot_params.get("plot_type", "scatter") if plot_params else "scatter"
    if plot_type == "scatter":
        y_col = "risk_score" if "risk_score" in df.columns else "seq_entropy"
        fig = px.scatter(df, x="antigenic_divergence", y=y_col,
                         color="vaccinated_bin", hover_data=["sample_id"], title="Vaccine Escape Candidates")
    elif plot_type == "bar":
        fig = px.bar(top_escape, x="sample_id", y="escape_score", title="Top Vaccine Escape Candidates")
    elif plot_type == "3d_scatter" and all(c in df.columns for c in ['umap3_x', 'umap3_y', 'umap3_z']):
        fig = px.scatter_3d(df, x='umap3_x', y='umap3_y', z='umap3_z', color='escape_score',
                            hover_name='sample_id', title="3D Escape Score on Manifold")
    elif plot_type == "box":
        fig = px.box(df, x="vaccinated_bin", y="antigenic_divergence", title="Antigenic Divergence by Vaccination")
    elif plot_type == "violin":
        fig = px.violin(df, x="vaccinated_bin", y="antigenic_divergence", title="Antigenic Divergence by Vaccination")
    else:
        fig = px.scatter(df, x="antigenic_divergence", y="seq_entropy")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Vaccine Escape Prediction", "figure": figure_json,
            "text": top_escape.to_html(index=False), "download_text": top_escape.to_csv(index=False)}


def analysis_early_warning(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if "collection_date" not in df.columns:
        return {"title": "Early Warning Signals", "error": "Requires collection_date."}
    df = df.dropna(subset=["collection_date"]).sort_values("collection_date")
    if "risk_score" not in df.columns:
        return {"title": "Early Warning Signals", "error": "risk_score not found. Run main analysis first."}
    window = max(3, len(df) // 10)
    if len(df) < window:
        return {"title": "Early Warning Signals", "error": "Insufficient data."}
    df["roll_var"] = df["risk_score"].rolling(window, min_periods=2).var()
    df["roll_acf"] = df["risk_score"].rolling(window, min_periods=2).apply(
        lambda x: x.autocorr() if len(x) > 2 else np.nan)
    plot_type = plot_params.get("plot_type", "line") if plot_params else "line"
    if plot_type == "line":
        fig = px.line(df, x="collection_date", y=["risk_score", "roll_var", "roll_acf"],
                      title="Early Warning Signals")
    elif plot_type == "scatter":
        fig = px.scatter(df, x="collection_date", y="risk_score", title="Risk Score Over Time")
    elif plot_type == "area":
        fig = px.area(df, x="collection_date", y="roll_var", title="Rolling Variance")
    else:
        fig = px.line(df, x="collection_date", y="risk_score")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict() if fig else None
    return {"title": "Early Warning Signals", "figure": figure_json,
            "text": "<p>Rolling variance and autocorrelation of risk score.</p>"}


def analysis_raincloud(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    var = plot_params.get("variable", "risk_score") if plot_params else "risk_score"
    if var not in df.columns:
        var = "seq_entropy" if "seq_entropy" in df.columns else None
    if var is None:
        return {"title": "Raincloud Plot", "error": "No suitable variable found."}
    fig = go.Figure()
    fig.add_trace(go.Violin(y=df[var], name=var, box_visible=False, meanline_visible=True,
                            points=False, side='negative', line_color='lightblue'))
    fig.add_trace(go.Box(y=df[var], name=var, boxmean='sd', boxpoints=False,
                         line_color='black', fillcolor='rgba(0,0,0,0)'))
    fig.add_trace(go.Scatter(x=np.random.normal(1, 0.04, size=len(df)), y=df[var],
                             mode='markers', name='points',
                             marker=dict(color='darkblue', size=4)))
    fig.update_layout(title=f"Raincloud Plot of {var}")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Raincloud Plot", "figure": figure_json,
            "text": f"<p>Raincloud plot for variable: {var}</p>"}


def analysis_venn(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    group_col = plot_params.get("group_col", "vaccinated_bin") if plot_params else "vaccinated_bin"
    if group_col not in df.columns:
        return {"title": "Venn Diagram", "error": f"Group column '{group_col}' not found."}
    groups = df[group_col].unique()
    if len(groups) != 2:
        return {"title": "Venn Diagram", "error": "Venn diagram requires exactly two groups."}
    group1_ids = df[df[group_col] == groups[0]]["sample_id"].tolist()
    group2_ids = df[df[group_col] == groups[1]]["sample_id"].tolist()
    seq_len = len(seqs[0])
    ant_pos = [p for p in ANTIGENIC_NT_POS if p < seq_len]

    def get_mutations(sample_ids):
        mutations = set()
        for sid in sample_ids:
            if sid not in ids:
                continue
            idx = ids.index(sid)
            seq = seqs[idx]
            for pos in ant_pos:
                if seq[pos] in "ACGT":
                    mutations.add((pos + 1, seq[pos]))
        return mutations

    set1 = get_mutations(group1_ids)
    set2 = get_mutations(group2_ids)
    try:
        fig = ff.create_venn2(sets=(set1, set2), set_labels=(str(groups[0]), str(groups[1])))
    except Exception as e:
        return {"title": "Venn Diagram", "error": f"Could not create Venn: {e}"}
    fig.update_layout(title="Mutation Set Venn Diagram (Antigenic Sites)")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Venn Diagram", "figure": figure_json,
            "text": f"<p>Set1 size: {len(set1)}, Set2 size: {len(set2)}, Intersection: {len(set1 & set2)}</p>"}


def analysis_tmrca(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    records = load_sequences(run_id)
    seqs = [str(r.seq).upper() for r in records]
    ids = [r.id for r in records]
    if "collection_date" not in df.columns:
        return {"title": "TMRCA Estimation", "error": "collection_date required for molecular clock."}
    date_df = df[["sample_id", "collection_date"]].dropna()
    if date_df.empty:
        return {"title": "TMRCA Estimation", "error": "No valid collection dates."}
    date_df["date_num"] = pd.to_datetime(date_df["collection_date"]).apply(
        lambda x: x.year + (x.dayofyear - 1) / 365.25)
    common_ids = set(ids).intersection(date_df["sample_id"])
    if len(common_ids) < 3:
        return {"title": "TMRCA Estimation", "error": "Need at least 3 samples with dates."}
    seqs_filtered = []
    ids_filtered = []
    dates = []
    for sid in common_ids:
        idx = ids.index(sid)
        seqs_filtered.append(seqs[idx])
        ids_filtered.append(sid)
        dates.append(date_df[date_df["sample_id"] == sid]["date_num"].iloc[0])
    dates = np.array(dates)

    def encode_seq(s):
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return np.array([mapping.get(c, 0) for c in s])

    X = np.array([encode_seq(s) for s in seqs_filtered])
    dist_mat = pairwise_distances(X, metric='hamming')
    from Bio.Phylo.TreeConstruction import DistanceMatrix
    names = ids_filtered
    matrix = []
    for i in range(len(names)):
        row = []
        for j in range(i):
            row.append(dist_mat[i, j])
        matrix.append(row)
    dm = DistanceMatrix(names, matrix)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)
    tree.root_at_midpoint()
    tips = tree.get_terminals()
    tip_dates = []
    tip_dist = []
    for tip in tips:
        name = tip.name
        if name in ids_filtered:
            tip_dates.append(dates[ids_filtered.index(name)])
            tip_dist.append(tree.distance(tip, tree.root))
    tip_dates = np.array(tip_dates)
    tip_dist = np.array(tip_dist)
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(tip_dates, tip_dist)
    tmrca = -intercept / slope if slope != 0 else np.nan
    fig = px.scatter(x=tip_dates, y=tip_dist, labels={'x': 'Date', 'y': 'Root-to-tip distance'},
                     title="Root-to-tip regression")
    x_range = np.linspace(min(tip_dates), max(tip_dates), 100)
    y_range = slope * x_range + intercept
    fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='Regression'))
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "TMRCA Estimation", "figure": figure_json,
            "text": f"<p>TMRCA: {tmrca:.3f} (year)<br>Slope (subs/site/year): {slope:.6f}<br>R²: {r_value ** 2:.3f}</p>"}


def analysis_spatial_interpolation(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    if not {"lat", "lon", "risk_score"}.issubset(df.columns):
        return {"title": "Spatial Interpolation", "error": "lat, lon, risk_score required."}
    df = df.dropna(subset=["lat", "lon", "risk_score"])
    if df.empty:
        return {"title": "Spatial Interpolation", "error": "No valid points."}
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    grid_lat = np.linspace(lat_min, lat_max, 50)
    grid_lon = np.linspace(lon_min, lon_max, 50)
    LATS, LONS = np.meshgrid(grid_lat, grid_lon)
    grid_points = np.column_stack([LATS.ravel(), LONS.ravel()])
    from scipy.spatial.distance import cdist
    coords = df[["lat", "lon"]].values
    values = df["risk_score"].values
    dist = cdist(grid_points, coords, metric='euclidean')
    weights = 1.0 / (dist + 1e-9)
    interpolated = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
    interpolated = interpolated.reshape(LATS.shape)
    mapbox_token = plot_params.get("mapbox_token", "") if plot_params else ""
    if mapbox_token:
        fig = go.Figure(go.Scattermapbox(
            lat=df["lat"], lon=df["lon"], mode='markers',
            marker=dict(color=df["risk_score"], colorscale='Viridis', size=8),
            text=df["sample_id"], name='Samples'
        ))
        fig.add_trace(go.Contour(
            z=interpolated, x=grid_lon, y=grid_lat,
            colorscale='Viridis', opacity=0.5, name='Interpolated Risk'
        ))
        fig.update_layout(mapbox_style="satellite-streets", mapbox_accesstoken=mapbox_token,
                          mapbox_center={"lat": df["lat"].mean(), "lon": df["lon"].mean()},
                          mapbox_zoom=5, title="Spatial Interpolation of Risk Score")
    else:
        fig = go.Figure(data=[
            go.Contour(z=interpolated, x=grid_lon, y=grid_lat, colorscale='Viridis'),
            go.Scatter(x=df["lon"], y=df["lat"], mode='markers',
                       marker=dict(color=df["risk_score"], colorscale='Viridis', showscale=False),
                       text=df["sample_id"])
        ])
        fig.update_layout(title="Spatial Interpolation of Risk Score")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Spatial Interpolation", "figure": figure_json,
            "text": "<p>Inverse distance weighting interpolation of risk score.</p>"}


def analysis_hotspot(run_id: str, plot_params: dict = None) -> dict:
    try:
        from esda.getisord import G_Local
        import libpysal
    except ImportError:
        return {"title": "Hotspot Analysis",
                "error": "Requires esda and libpysal. Install with: pip install esda libpysal"}
    df = load_merged_data(run_id)
    if not {"lat", "lon", "risk_score"}.issubset(df.columns):
        return {"title": "Hotspot Analysis", "error": "lat, lon, risk_score required."}
    df = df.dropna(subset=["lat", "lon", "risk_score"])
    if len(df) < 5:
        return {"title": "Hotspot Analysis", "error": "Need at least 5 points."}
    coords = df[["lat", "lon"]].values
    from libpysal.weights import KNN
    w = KNN.from_array(coords, k=5)
    w.transform = 'r'
    y = df["risk_score"].values
    g_local = G_Local(y, w, transform='r', star=True)
    df["Gi_star"] = g_local.Zs
    df["p_value"] = g_local.p_sim
    conditions = [
        (df["Gi_star"] > 1.96) & (df["p_value"] < 0.05),
        (df["Gi_star"] < -1.96) & (df["p_value"] < 0.05)
    ]
    choices = ['Hotspot', 'Coldspot']
    df["hotspot"] = np.select(conditions, choices, default='Not significant')
    mapbox_token = plot_params.get("mapbox_token", "") if plot_params else ""
    if mapbox_token:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="hotspot",
                                hover_name="sample_id", title="Hotspot Analysis (Gi*)",
                                mapbox_style="satellite-streets", zoom=5)
        fig.update_layout(mapbox_accesstoken=mapbox_token)
    else:
        fig = px.scatter_mapbox(df, lat="lat", lon="lon", color="hotspot",
                                hover_name="sample_id", title="Hotspot Analysis (Gi*)",
                                mapbox_style="open-street-map", zoom=5)
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Hotspot Analysis", "figure": figure_json,
            "text": df[["sample_id", "hotspot", "Gi_star", "p_value"]].to_html(index=False),
            "download_text": df[["sample_id", "hotspot", "Gi_star", "p_value"]].to_csv(index=False)}


def analysis_bubble_chart(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    x = plot_params.get("x", "seq_entropy") if plot_params else "seq_entropy"
    y = plot_params.get("y", "antigenic_divergence") if plot_params else "antigenic_divergence"
    size = plot_params.get("size", "risk_score") if plot_params else "risk_score"
    color = plot_params.get("color", "vaccinated_bin") if plot_params else "vaccinated_bin"
    if x not in df.columns or y not in df.columns:
        return {"title": "Bubble Chart", "error": "Required columns missing."}
    if size not in df.columns:
        size = None
    if color not in df.columns:
        color = None
    fig = px.scatter(df, x=x, y=y, size=size, color=color, hover_name="sample_id",
                     title="Bubble Chart")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Bubble Chart", "figure": figure_json, "text": "<p>Multi-dimensional bubble chart.</p>"}


def analysis_parallel_coordinates(run_id: str, plot_params: dict = None) -> dict:
    df = load_merged_data(run_id)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        return {"title": "Parallel Coordinates", "error": "Need at least 3 numeric columns."}
    selected = plot_params.get("dimensions", numeric_cols[:6]) if plot_params else numeric_cols[:6]
    if isinstance(selected, str):
        selected = [s.strip() for s in selected.split(',')]
    selected = [c for c in selected if c in df.columns]
    if len(selected) < 2:
        return {"title": "Parallel Coordinates", "error": "Not enough valid dimensions."}
    color_col = plot_params.get("color", "risk_score") if plot_params else "risk_score"
    if color_col not in df.columns:
        color_col = selected[0]
    fig = px.parallel_coordinates(df, dimensions=selected, color=color_col,
                                  title="Parallel Coordinates Plot")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Parallel Coordinates", "figure": figure_json,
            "text": "<p>High-dimensional feature inspection.</p>"}


# New analysis: Quantum Forecast (extends quantum_dynamics with multiple trajectories)
def analysis_quantum_forecast(run_id: str, plot_params: dict = None) -> dict:
    if not TORCH_AVAILABLE:
        return {"title": "Quantum Forecast", "error": "PyTorch not installed."}
    df = load_merged_data(run_id)
    exclude = ["sample_id", "farm_id", "collection_date", "umap_x", "umap_y", "umap3_x", "umap3_y", "umap3_z"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    if len(feature_cols) < 3:
        return {"title": "Quantum Forecast", "error": "Not enough numeric features."}
    X = df[feature_cols].fillna(0.0).values
    vae, scaler = train_vae(X, latent_dim=8, epochs=50)
    # Generate multiple trajectories from different starting points
    n_forecast = plot_params.get("n_forecast", 5) if plot_params else 5
    n_steps = plot_params.get("n_steps", 20) if plot_params else 20
    trajectories = []
    for i in range(min(n_forecast, len(X))):
        traj = predict_evolution(vae, scaler, X[i], n_steps=n_steps)
        trajectories.append(traj)
    # Plot first three features in 3D
    if len(feature_cols) >= 3:
        fig = go.Figure()
        for j, traj in enumerate(trajectories):
            df_traj = pd.DataFrame(traj, columns=feature_cols)
            fig.add_trace(go.Scatter3d(x=df_traj[feature_cols[0]], y=df_traj[feature_cols[1]], z=df_traj[feature_cols[2]],
                                       mode='lines+markers', name=f'Start {j}'))
        fig.update_layout(title="Quantum Forecast: Multiple Evolutionary Trajectories")
    else:
        fig = go.Figure()
        for j, traj in enumerate(trajectories):
            df_traj = pd.DataFrame(traj, columns=feature_cols)
            fig.add_trace(go.Scatter(x=list(range(n_steps+1)), y=df_traj[feature_cols[0]],
                                     mode='lines+markers', name=f'Start {j}'))
        fig.update_layout(title="Quantum Forecast: Feature Evolution", xaxis_title="Step", yaxis_title=feature_cols[0])
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    return {"title": "Quantum Forecast", "figure": figure_json,
            "text": f"<p>Generated {len(trajectories)} trajectories of {n_steps} steps each.</p>"}


# New analysis: ML Forecast (LSTM for risk score prediction)
def analysis_ml_forecast(run_id: str, plot_params: dict = None) -> dict:
    if not TORCH_AVAILABLE:
        return {"title": "ML Forecast", "error": "PyTorch not installed."}
    df = load_merged_data(run_id)
    if "collection_date" not in df.columns or "risk_score" not in df.columns:
        return {"title": "ML Forecast", "error": "Requires collection_date and risk_score."}
    df = df.dropna(subset=["collection_date", "risk_score"]).sort_values("collection_date")
    if len(df) < 10:
        return {"title": "ML Forecast", "error": "Need at least 10 time points."}
    # Prepare time series
    dates = pd.to_datetime(df["collection_date"])
    values = df["risk_score"].values
    # Simple LSTM model
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.linear(out[:, -1, :])
            return out
    # Create sequences
    seq_len = plot_params.get("seq_len", 5) if plot_params else 5
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    if len(X) < 2:
        return {"title": "ML Forecast", "error": "Not enough sequences."}
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)
    # Create dataset
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # Model
    model = LSTMPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    # Train
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    # Predict on test
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy().flatten()
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred))), y=y_pred, mode='lines+markers', name='Predicted'))
    fig.update_layout(title="LSTM Forecast of Risk Score (Test Set)", xaxis_title="Time Step", yaxis_title="Risk Score")
    if plot_params:
        apply_plot_params(fig, plot_params)
    figure_json = fig.to_dict()
    # Also forecast future steps
    # Use last sequence to predict next
    last_seq = values[-seq_len:].reshape(1, seq_len, 1)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).to(device)
    future_preds = []
    current_seq = last_seq_t.clone()
    for _ in range(10):
        with torch.no_grad():
            pred = model(current_seq)
        future_preds.append(pred.item())
        # Update sequence: remove first, append pred
        new_seq = torch.cat([current_seq[:, 1:, :], pred.unsqueeze(2)], dim=1)
        current_seq = new_seq
    # Add future to plot
    fig.add_trace(go.Scatter(x=list(range(len(values), len(values)+10)), y=future_preds,
                             mode='lines+markers', name='Future Forecast', line=dict(dash='dash')))
    # Update figure
    figure_json = fig.to_dict()
    return {"title": "ML Forecast (LSTM)", "figure": figure_json,
            "text": f"<p>Trained on {len(X_train)} sequences, tested on {len(X_test)}. Future 10 steps forecasted.</p>"}


# Common plot parameter applier
def apply_plot_params(fig: go.Figure, params: dict):
    if not params:
        return
    width = params.get("width")
    height = params.get("height")
    if width and height:
        fig.update_layout(width=int(width), height=int(height))
    bgcolor = params.get("bgcolor")
    if bgcolor:
        fig.update_layout(plot_bgcolor=bgcolor, paper_bgcolor=bgcolor)
    show_legend = params.get("show_legend")
    if show_legend is not None:
        fig.update_layout(showlegend=show_legend)
    title = params.get("title")
    if title is not None:
        fig.update_layout(title=title if title else "")


# Map analysis names to functions
ANALYSES = {
    "shannon_entropy": analysis_shannon_entropy,
    "dnds": analysis_dnds,
    "epistatic_network": analysis_epistatic_network,
    "individual_trajectories": analysis_individual_trajectories,
    "farm_coupling": analysis_farm_coupling,
    "spatiotemporal_diffusion": analysis_spatiotemporal_diffusion,
    "vaccine_escape": analysis_vaccine_escape,
    "early_warning": analysis_early_warning,
    "geospatial_clustering": analysis_geospatial_clustering,
    "geospatial_regression": analysis_geospatial_regression,
    "manifold_curvature": analysis_manifold_curvature,
    "immune_escape_pathways": analysis_immune_escape_pathways,
    "recombination_hotspots": analysis_recombination_hotspots,
    "temporal_trend": analysis_temporal_trend,
    "mutation_heatmap": analysis_mutation_heatmap,
    "correlation_matrix": analysis_correlation_matrix,
    "tsne": analysis_tsne,
    "risk_factors": analysis_risk_factors,
    "temporal_clustering": analysis_temporal_clustering,
    "quantum_dynamics": analysis_quantum_dynamics,
    "phylogenetic_tree": analysis_phylogenetic_tree,
    "selection_pressure_fel": analysis_selection_pressure_fel,
    "transmission_network": analysis_transmission_network,
    "antigenic_variation": analysis_antigenic_variation,
    "ml_prediction": analysis_machine_learning_prediction,
    "pca": analysis_pca,
    "raincloud": analysis_raincloud,
    "venn": analysis_venn,
    "tmrca": analysis_tmrca,
    "spatial_interpolation": analysis_spatial_interpolation,
    "hotspot": analysis_hotspot,
    "bubble_chart": analysis_bubble_chart,
    "parallel_coordinates": analysis_parallel_coordinates,
    "quantum_forecast": analysis_quantum_forecast,
    "ml_forecast": analysis_ml_forecast,
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
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)
        fasta_path = save_uploaded_fasta(run_dir, wgs_b)
        records = list(SeqIO.parse(io.StringIO(_bytes_to_text(wgs_b)), "fasta"))
        if not records:
            raise ValueError("No FASTA records.")
        lengths = {len(r.seq) for r in records}
        if len(lengths) != 1:
            raise ValueError("Sequences have different lengths. Please provide aligned FASTA.")
        genomic_df = extract_features_from_alignment(records)
        epi_df = epi_to_features_from_bytes(epi_b)
        merged = merge_features(genomic_df, epi_df)
        qc = basic_qc(merged)
        seqs = [str(r.seq).upper() for r in records]
        emb3d, manifold_model = topology_aware_embedding(seqs, n_components=3)
        emb2d = emb3d[:, :2]
        csv_path = save_outputs(run_id, merged, emb2d, emb3d, manifold_model)
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
        fasta_path = save_uploaded_fasta(run_dir, wgs_b)
        records = list(SeqIO.parse(io.StringIO(_bytes_to_text(wgs_b)), "fasta"))
        if not records:
            raise ValueError("No FASTA records.")
        lengths = {len(r.seq) for r in records}
        if len(lengths) != 1:
            raise ValueError("Sequences have different lengths. Please provide aligned FASTA.")
        genomic_df = extract_features_from_alignment(records)
        epi_df = epi_to_features_from_bytes(epi_b)
        merged = merge_features(genomic_df, epi_df)
        qc = basic_qc(merged)
        seqs = [str(r.seq).upper() for r in records]
        emb3d, manifold_model = topology_aware_embedding(seqs, n_components=3)
        emb2d = emb3d[:, :2]
        csv_path = save_outputs(run_id, merged, emb2d, emb3d, manifold_model)
        # Load results for preview
        out = pd.read_csv(csv_path)
        fig1 = px.scatter(out, x="umap_x", y="umap_y", color="risk_score",
                          hover_data=[c for c in out.columns if c not in ["umap_x", "umap_y"]],
                          title="UMAP embedding (2D) — color = risk score")
        fig2 = px.histogram(out, x="risk_score", nbins=20, title="Risk Score Distribution")
        preview_df = out.head(12)
        return render_results_page(run_id, qc, preview_df, fig1, fig2)
    except Exception as e:
        return HTMLResponse(f"<h2>Run failed</h2><p><b>Error:</b> {str(e)}</p><p><a href='/'>Go back</a></p>",
                            status_code=400)


@app.post("/analyze/{run_id}/{analysis_name}")
async def analyze(
        run_id: str,
        analysis_name: str,
        request: Request,
):
    if analysis_name not in ANALYSES:
        raise HTTPException(status_code=404, detail="Analysis not found")
    try:
        body = await request.json() if request.headers.get("content-type") == "application/json" else {}
        plot_params = body.get("plot_params", {})
        result = ANALYSES[analysis_name](run_id, plot_params)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------- HTML rendering ----------
def render_results_page(run_id: str, qc: dict, preview_df: pd.DataFrame, fig1: go.Figure, fig2: go.Figure) -> str:
    qc_html = "<ul>" + "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in qc.items()]) + "</ul>"
    preview_html = preview_df.to_html(index=False, escape=True)
    download_url = f"/download/{run_id}"
    fig1_json = fig1.to_json()
    fig2_json = fig2.to_json()
    analysis_options = "\n".join(
        [f'<option value="{name}">{name.replace("_", " ").title()}</option>' for name in ANALYSES.keys()])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRRSV Evolution – Run {run_id}</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}
        .animate-float {{
            animation: float 4s ease-in-out infinite;
        }}
        .glass {{
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
    </style>
</head>
<body class="bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 font-sans min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-blue-500 animate-float">
                PRRSV Evolution · {run_id}
            </h1>
            <div class="space-x-4">
                <a href="{download_url}" class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-full transition-all shadow-lg inline-block">Download CSV</a>
                <a href="/" class="bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-6 rounded-full transition-all shadow-lg inline-block">New Run</a>
            </div>
        </div>

        <!-- QC + Preview Grid -->
        <div class="grid md:grid-cols-3 gap-6 mb-8">
            <div class="glass rounded-2xl p-6">
                <h2 class="text-xl font-semibold mb-4 text-teal-300">QC Summary</h2>
                {qc_html}
            </div>
            <div class="md:col-span-2 glass rounded-2xl p-6 overflow-x-auto">
                <h2 class="text-xl font-semibold mb-4 text-teal-300">Preview (first 12 samples)</h2>
                {preview_html}
            </div>
        </div>

        <!-- UMAP + Histogram -->
        <div class="grid md:grid-cols-2 gap-6 mb-8">
            <div class="glass rounded-2xl p-4 h-96" id="umap-plot"></div>
            <div class="glass rounded-2xl p-4 h-96" id="hist-plot"></div>
        </div>

        <!-- Advanced Analysis Panel -->
        <div class="glass rounded-2xl p-6">
            <h2 class="text-2xl font-bold mb-6 text-teal-300">Advanced Analysis</h2>
            <div class="flex flex-wrap gap-4 mb-6">
                <select id="analysis-select" class="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 flex-grow">
                    {analysis_options}
                </select>
                <button id="run-analysis" class="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-8 rounded-full transition-all">Run</button>
            </div>

            <!-- Parameter fields -->
            <div id="param-fields" class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 p-4 bg-gray-800 rounded-lg">
                <!-- JS will populate -->
            </div>

            <!-- Loading & Output -->
            <div id="loading" class="text-center py-8 hidden">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-teal-400"></div>
                <p class="mt-2 text-teal-300">Running analysis...</p>
            </div>
            <div id="analysis-output" class="mt-6"></div>
        </div>
    </div>

    <script>
        const runId = "{run_id}";
        const analyses = {list(ANALYSES.keys())};

        // Render initial plots
        const umapPlot = document.getElementById('umap-plot');
        const histPlot = document.getElementById('hist-plot');
        Plotly.react(umapPlot, JSON.parse('{fig1_json}').data, JSON.parse('{fig1_json}').layout);
        Plotly.react(histPlot, JSON.parse('{fig2_json}').data, JSON.parse('{fig2_json}').layout);

        // Parameter templates for each analysis (extend as needed)
        const paramTemplates = {{
            'shannon_entropy': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="line">Line</option><option value="bar">Bar</option><option value="heatmap">Heatmap</option><option value="violin">Violin</option><option value="box">Box</option><option value="area">Area</option><option value="density_heatmap">Density Heatmap</option></select></label>`,
            'dnds': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="bar">Bar</option><option value="scatter">Scatter</option><option value="histogram">Histogram</option><option value="box">Box</option><option value="violin">Violin</option></select></label>`,
            'epistatic_network': `<p class="text-gray-400">No parameters</p>`,
            'individual_trajectories': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="scatter">Scatter</option><option value="line">Line</option><option value="3d_scatter">3D Scatter</option></select></label>`,
            'farm_coupling': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="bar">Bar</option><option value="scatter">Scatter</option><option value="heatmap">Heatmap</option></select></label>`,
            'spatiotemporal_diffusion': `<label class="block"><span class="text-sm">Mapbox token (optional):</span> <input type="text" name="mapbox_token" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'geospatial_clustering': `<label class="block"><span class="text-sm">Mapbox token (optional):</span> <input type="text" name="mapbox_token" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'geospatial_regression': `<p class="text-gray-400">No parameters</p>`,
            'manifold_curvature': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="3d_scatter">3D Scatter</option><option value="scatter">2D Scatter</option><option value="histogram">Histogram</option></select></label>`,
            'immune_escape_pathways': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="bar">Bar</option><option value="scatter">Scatter</option><option value="3d_scatter">3D Scatter</option></select></label>`,
            'recombination_hotspots': `<p class="text-gray-400">No parameters</p>`,
            'temporal_trend': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="line">Line</option><option value="scatter">Scatter</option><option value="bar">Bar</option><option value="area">Area</option></select></label>`,
            'mutation_heatmap': `<p class="text-gray-400">No parameters</p>`,
            'correlation_matrix': `<p class="text-gray-400">No parameters</p>`,
            'tsne': `<p class="text-gray-400">No parameters</p>`,
            'risk_factors': `<p class="text-gray-400">No parameters</p>`,
            'temporal_clustering': `<p class="text-gray-400">No parameters</p>`,
            'quantum_dynamics': `<p class="text-gray-400">No parameters</p>`,
            'phylogenetic_tree': `<p class="text-gray-400">No parameters</p>`,
            'selection_pressure_fel': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="bar">Bar</option><option value="scatter">Scatter</option><option value="line">Line</option><option value="heatmap">Heatmap</option></select></label>`,
            'transmission_network': `<label class="block"><span class="text-sm">Distance threshold:</span> <input type="number" name="dist_threshold" value="0.05" step="0.01" min="0" max="1" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'antigenic_variation': `<p class="text-gray-400">No parameters</p>`,
            'ml_prediction': `<p class="text-gray-400">No parameters</p>`,
            'pca': `<label class="block"><span class="text-sm">Plot type:</span> <select name="plot_type" class="bg-gray-700 rounded px-2 py-1 w-full"><option value="scatter">Scatter</option><option value="3d_scatter">3D Scatter</option><option value="biplot">Biplot</option><option value="scree">Scree</option><option value="heatmap">Heatmap</option></select></label>`,
            'raincloud': `<label class="block"><span class="text-sm">Variable:</span> <input type="text" name="variable" value="risk_score" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'venn': `<label class="block"><span class="text-sm">Group column:</span> <input type="text" name="group_col" value="vaccinated_bin" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'tmrca': `<p class="text-gray-400">No parameters</p>`,
            'spatial_interpolation': `<label class="block"><span class="text-sm">Mapbox token (optional):</span> <input type="text" name="mapbox_token" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'hotspot': `<label class="block"><span class="text-sm">Mapbox token (optional):</span> <input type="text" name="mapbox_token" class="bg-gray-700 rounded px-2 py-1 w-full"></label>`,
            'bubble_chart': `
                <label class="block"><span class="text-sm">X:</span> <input type="text" name="x" value="seq_entropy" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
                <label class="block"><span class="text-sm">Y:</span> <input type="text" name="y" value="antigenic_divergence" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
                <label class="block"><span class="text-sm">Size:</span> <input type="text" name="size" value="risk_score" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
                <label class="block"><span class="text-sm">Color:</span> <input type="text" name="color" value="vaccinated_bin" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
            `,
            'parallel_coordinates': `
                <label class="block"><span class="text-sm">Dimensions (comma-separated):</span> <input type="text" name="dimensions" value="seq_entropy,antigenic_divergence,gc_content" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
                <label class="block"><span class="text-sm">Color:</span> <input type="text" name="color" value="risk_score" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
            `,
            'quantum_forecast': `
                <label class="block"><span class="text-sm">Number of trajectories:</span> <input type="number" name="n_forecast" value="5" min="1" max="20" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
                <label class="block"><span class="text-sm">Steps per trajectory:</span> <input type="number" name="n_steps" value="20" min="5" max="100" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
            `,
            'ml_forecast': `
                <label class="block"><span class="text-sm">Sequence length:</span> <input type="number" name="seq_len" value="5" min="2" max="20" class="bg-gray-700 rounded px-2 py-1 w-full"></label>
            `
        }};

        function updateParamFields() {{
            const analysis = document.getElementById('analysis-select').value;
            const container = document.getElementById('param-fields');
            container.innerHTML = (paramTemplates[analysis] || '<p class="text-gray-400">No parameters</p>') + 
                '<div class="col-span-3 grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-600">' +
                '<label class="block"><span class="text-sm">Width:</span> <input type="number" name="width" value="800" class="bg-gray-700 rounded px-2 py-1 w-full"></label>' +
                '<label class="block"><span class="text-sm">Height:</span> <input type="number" name="height" value="600" class="bg-gray-700 rounded px-2 py-1 w-full"></label>' +
                '<label class="block"><span class="text-sm">BG color:</span> <input type="color" name="bgcolor" value="#ffffff" class="bg-gray-700 rounded px-2 py-1 w-full"></label>' +
                '<label class="flex items-center gap-2"><input type="checkbox" name="show_legend" checked class="bg-gray-700"> <span class="text-sm">Show legend</span></label>' +
                '<label class="block col-span-2"><span class="text-sm">Title (optional):</span> <input type="text" name="title" class="bg-gray-700 rounded px-2 py-1 w-full"></label>' +
                '</div>';
        }}

        document.getElementById('analysis-select').addEventListener('change', updateParamFields);
        updateParamFields();

        document.getElementById('run-analysis').addEventListener('click', async () => {{
            const analysis = document.getElementById('analysis-select').value;
            const inputs = document.querySelectorAll('#param-fields input, #param-fields select');
            const plotParams = {{}};
            inputs.forEach(input => {{
                if (input.name) {{
                    if (input.type === 'checkbox') plotParams[input.name] = input.checked;
                    else if (input.type === 'number') plotParams[input.name] = parseFloat(input.value);
                    else plotParams[input.name] = input.value;
                }}
            }});
            const outputDiv = document.getElementById('analysis-output');
            const loadingDiv = document.getElementById('loading');
            outputDiv.innerHTML = '';
            loadingDiv.classList.remove('hidden');

            try {{
                const response = await fetch(`/analyze/${{runId}}/${{analysis}}`, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{plot_params: plotParams}})
                }});
                const data = await response.json();
                loadingDiv.classList.add('hidden');
                if (data.error) {{
                    outputDiv.innerHTML = `<div class="bg-red-900 border border-red-700 rounded-lg p-4 text-red-200">Error: ${{data.error}}</div>`;
                }} else {{
                    let html = `<h3 class="text-xl font-bold text-teal-300 mb-4">${{data.title || analysis}}</h3>`;
                    if (data.figure) {{
                        const plotDiv = document.createElement('div');
                        plotDiv.className = 'w-full h-96 mb-4';
                        outputDiv.appendChild(plotDiv);
                        Plotly.react(plotDiv, data.figure.data, data.figure.layout);
                    }}
                    if (data.text) {{
                        const textDiv = document.createElement('div');
                        textDiv.className = 'prose prose-invert max-w-none bg-gray-800 rounded-lg p-4 overflow-x-auto';
                        textDiv.innerHTML = data.text;
                        outputDiv.appendChild(textDiv);
                    }}
                    if (data.download_text) {{
                        const blob = new Blob([data.download_text], {{type: 'text/csv'}});
                        const url = URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = `${{analysis}}_results.csv`;
                        link.className = 'inline-block mt-4 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-full';
                        link.textContent = 'Download CSV';
                        outputDiv.appendChild(link);
                    }}
                }}
            }} catch (error) {{
                loadingDiv.classList.add('hidden');
                outputDiv.innerHTML = `<div class="bg-red-900 border border-red-700 rounded-lg p-4 text-red-200">Error: ${{error}}</div>`;
            }}
        }});
    </script>
</body>
</html>"""


# ---------- New Landing Page (removed feature grid, added about/help/instructions) ----------
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRRSV Predictive Evolution</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }
        .animate-float {
            animation: float 6s ease-in-out infinite;
        }
        .glass {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .bg-gradient- animated {
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 text-white font-sans min-h-screen">
    <!-- Animated 3D background (Plotly) -->
    <div id="bg-plot" class="fixed top-0 left-0 w-full h-full -z-10 opacity-30"></div>

    <div class="relative z-10 container mx-auto px-4 py-8">
        <!-- Hero -->
        <div class="text-center mb-12 animate-float">
            <h1 class="text-6xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-yellow-300 to-pink-300 drop-shadow-2xl">
                PRRSV Predictive Evolution
            </h1>
            <p class="text-2xl text-gray-200 max-w-3xl mx-auto">
                Topology‑Aware Manifold Learning · Quantum‑Inspired Dynamics · Geospatial Intelligence
            </p>
        </div>

        <!-- Main Upload Card -->
        <div class="glass rounded-3xl p-8 max-w-2xl mx-auto shadow-2xl mb-12">
            <h2 class="text-3xl font-semibold mb-6 text-center">Start a New Analysis</h2>
            <form id="upload-form" enctype="multipart/form-data" action="/run_ui" method="post" class="space-y-6">
                <div>
                    <label class="block text-lg mb-2">1. WGS / ORF5 FASTA <span class="text-sm text-gray-300">(aligned)</span></label>
                    <input type="file" name="wgs_fasta" accept=".fasta,.fa,.fna" required class="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-pink-600 file:text-white hover:file:bg-pink-700 transition" />
                </div>
                <div>
                    <label class="block text-lg mb-2">2. Epidemiology CSV <span class="text-sm text-gray-300">(sample_id, optional lat/lon, collection_date, farm_id, vaccinated, etc.)</span></label>
                    <input type="file" name="epi_csv" accept=".csv" required class="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-pink-600 file:text-white hover:file:bg-pink-700 transition" />
                </div>
                <button type="submit" class="w-full bg-gradient-to-r from-yellow-400 to-pink-500 text-gray-900 font-bold py-4 px-6 rounded-full text-xl shadow-lg hover:scale-105 transition-transform">
                    Launch Pipeline
                </button>
            </form>
        </div>

        <!-- About, Help, Instructions (Accordion style) -->
        <div class="max-w-4xl mx-auto space-y-4">
            <!-- About -->
            <details class="glass rounded-xl p-4 open:bg-white/20 transition">
                <summary class="text-2xl font-semibold cursor-pointer list-none">📘 About</summary>
                <div class="mt-4 text-gray-200 space-y-2">
                    <p>This platform integrates genomic and epidemiological data to predict evolutionary trajectories of PRRSV. It combines manifold learning (UMAP), quantum-inspired variational autoencoders, and geospatial analysis to identify high-risk variants and transmission hotspots.</p>
                    <p>Developed by Nahiduzzaman, URA, Department of Microbiology and Hygiene, BAU.</p>
                </div>
            </details>

            <!-- Help -->
            <details class="glass rounded-xl p-4 open:bg-white/20 transition">
                <summary class="text-2xl font-semibold cursor-pointer list-none">❓ Help</summary>
                <div class="mt-4 text-gray-200 space-y-2">
                    <p>For any issues, please contact the developer. Common solutions:</p>
                    <ul class="list-disc list-inside">
                        <li>Ensure FASTA sequences are aligned (same length).</li>
                        <li>CSV must contain a 'sample_id' column matching FASTA headers.</li>
                        <li>For geospatial features, include 'lat' and 'lon' columns.</li>
                        <li>To use satellite imagery, obtain a free Mapbox token from <a href="https://mapbox.com" target="_blank" class="text-yellow-300 underline">mapbox.com</a> and enter it in analysis parameters.</li>
                    </ul>
                </div>
            </details>

            <!-- Instructions / File Structure -->
            <details class="glass rounded-xl p-4 open:bg-white/20 transition">
                <summary class="text-2xl font-semibold cursor-pointer list-none">📁 File Structure & Instructions</summary>
                <div class="mt-4 text-gray-200 space-y-4">
                    <h3 class="text-xl font-semibold">FASTA file</h3>
                    <p>Aligned nucleotide sequences (ORF5 or full genome). Example:</p>
                    <pre class="bg-gray-800 p-2 rounded text-sm overflow-x-auto">
>sample_1
ATG... (same length for all)
>sample_2
ATG...</pre>

                    <h3 class="text-xl font-semibold">Epidemiology CSV</h3>
                    <p>Required column: <code>sample_id</code> (must match FASTA headers). Optional columns:</p>
                    <ul class="list-disc list-inside">
                        <li><code>farm_id</code> – farm identifier</li>
                        <li><code>collection_date</code> – date (YYYY-MM-DD)</li>
                        <li><code>lat</code>, <code>lon</code> – decimal coordinates</li>
                        <li><code>vaccinated</code> – yes/no/1/0</li>
                        <li><code>homologous_or_heterologous</code> – homologous/heterologous</li>
                        <li><code>vaccine_type</code> – for one-hot encoding</li>
                        <li><code>age_at_vaccination</code>, <code>age_at_challenge</code>, <code>challenge_dose</code> – numeric</li>
                    </ul>
                    <p>Example:</p>
                    <pre class="bg-gray-800 p-2 rounded text-sm overflow-x-auto">
sample_id,farm_id,collection_date,lat,lon,vaccinated,homologous_or_heterologous
s1,FARM1,2023-01-15,35.2,-80.1,yes,homologous
s2,FARM1,2023-02-20,35.3,-80.2,no,</pre>
                </div>
            </details>

            <!-- Link to Satellite Imagery -->
            <div class="glass rounded-xl p-4 text-center">
                <p class="text-lg">🛰️ For satellite imagery in geospatial analyses, get a free Mapbox token and enter it in the analysis panel.</p>
                <a href="https://account.mapbox.com/access-tokens/" target="_blank" class="inline-block mt-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-full transition">Get Mapbox Token</a>
            </div>
        </div>

        <footer class="text-center mt-16 text-gray-300 text-sm">
            © 2025 Nahiduzzaman, URA, Department of Microbiology and Hygiene, BAU
        </footer>
    </div>

    <script>
        // Animated 3D background (rotating torus)
        const n = 100;
        const theta = Array.from({length: n}, (_,i) => i * 2*Math.PI/n);
        const phi = Array.from({length: n}, (_,i) => i * Math.PI/n);
        const x = [], y = [], z = [];
        for (let i=0; i<n; i++) {
            for (let j=0; j<n; j++) {
                const R = 2;
                const r = 1;
                const u = theta[i];
                const v = phi[j];
                x.push((R + r*Math.cos(v)) * Math.cos(u));
                y.push((R + r*Math.cos(v)) * Math.sin(u));
                z.push(r * Math.sin(v));
            }
        }
        const trace = {
            x: x, y: y, z: z,
            mode: 'markers',
            type: 'scatter3d',
            marker: {
                size: 2,
                color: z,
                colorscale: 'Viridis',
                opacity: 0.8
            }
        };
        const layout = {
            scene: {
                camera: { eye: {x: 1.5, y: 1.5, z: 1.5} },
                xaxis: { visible: false },
                yaxis: { visible: false },
                zaxis: { visible: false }
            },
            margin: { l:0, r:0, t:0, b:0 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        Plotly.newPlot('bg-plot', [trace], layout, {displayModeBar: false, responsive: true});
        // Rotate camera
        let angle = 0;
        setInterval(() => {
            angle += 0.02;
            Plotly.relayout('bg-plot', {
                'scene.camera': { eye: {x: 3*Math.sin(angle), y: 3*Math.cos(angle), z: 1.5} }
            });
        }, 100);
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)