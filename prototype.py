from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import random
import math
import json

app = FastAPI(
    title="PRRSV Evolutionary Intelligence Platform",
    version="11.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ------------------------------------------------------------
# ENORMOUS DUMMY DATA – analysis-focused (EXPANDED)
# ------------------------------------------------------------

# Jobs (30 entries) – unchanged
job_titles = [
    "GNN training · Epistatic network",
    "Manifold embedding · diffusion maps",
    "Immune escape prediction · quantum kernel",
    "Recombination hotspot detection",
    "Fitness landscape VAE",
    "Topological data analysis",
    "Mutation trajectory LSTM",
    "Antigenic mapping (AlphaFold2)",
    "Epistatic network update",
    "Recombination analysis (RDP5)",
    "Manifold refinement",
    "Immune escape batch prediction",
    "Phylogenetic inference (IQ-TREE)",
    "Molecular clock (BEAST)",
    "Outbreak source tracking"
]
job_models = [
    "Graph Neural Network", "Diffusion Map", "Quantum SVM", "RDP5",
    "Variational Autoencoder", "Persistent Homology", "LSTM", "AlphaFold2",
    "GCN", "RDP5", "UMAP", "Quantum SVM", "IQ-TREE", "BEAST", "Maximum Likelihood"
]
JOBS = []
for i in range(1, 31):
    idx = i % len(job_titles)
    JOBS.append({
        "id": f"JOB-{5000+i}",
        "title": job_titles[idx],
        "status": random.choice(["Running","Completed","Queued","Paused"]),
        "progress": random.randint(0,100),
        "started": f"{random.randint(8,15):02d}:{random.randint(0,59):02d}",
        "eta": random.choice(["5 min","12 min","30 min","Done","--"]),
        "model": job_models[idx]
    })

# Results (80 samples with mutation lists) – expanded to 80
RESULTS = []
lineages = ["L1","L2","L3","L4","L5","L6","L7","L8","L9"]
mutations_pool = ["N32S","E89K","G152R","A213T","S315I","M401L","T45A","V112I","P61S","L124F","A12V","R98K","D215N","Y246C","F307L","H39Y","L50Q","G64R","S77N","T83I"]
for i in range(80):
    sample = f"PRRSV_{random.choice(['MN','IA','IL','NE','NC','OH','TX','CA','FL','PA','MI','WI','IN'])}_24{random.randint(100,999)}"
    lineage = random.choice(lineages)
    mut_count = random.randint(3,18)
    muts = random.sample(mutations_pool, min(mut_count, len(mutations_pool)))
    recomb = random.choice(["Yes","No"])
    escape_score = round(random.uniform(0.2,0.98),2)
    if escape_score > 0.8:
        immune = "Confirmed"
        risk = "Critical"
    elif escape_score > 0.6:
        immune = "Potential"
        risk = "High"
    elif escape_score > 0.4:
        immune = "Unlikely"
        risk = "Moderate"
    else:
        immune = "Unlikely"
        risk = "Monitor"
    RESULTS.append({
        "sample": sample,
        "lineage": lineage,
        "mutations": mut_count,
        "recomb": recomb,
        "escape_score": escape_score,
        "risk": risk,
        "immune_escape": immune,
        "mutations_list": muts
    })

# Current situation for map (50+ farms) – expanded
CURRENT_SITUATION = []
states_abbr = ["MN","IA","IL","NE","NC","OH","TX","CA","FL","PA","MI","WI","IN","MO","KS","SD","ND","CO","OK","AR","LA","MS","AL","GA","SC","VA","WV","KY","TN","NY"]
for i, st in enumerate(states_abbr):
    for j in range(1,4):
        cases = random.randint(0,40)
        if cases > 25:
            status = "Critical"
        elif cases > 15:
            status = "High"
        elif cases > 7:
            status = "Moderate"
        else:
            status = "Monitor"
        # Compute risk score as combination of cases and random factor
        risk_score = round(cases/40 * 100 + random.uniform(-10,10), 1)
        CURRENT_SITUATION.append({
            "site": f"{st} Farm {chr(65+j)}",
            "lat": random.uniform(30,48) if st not in ["CA","OR","WA"] else random.uniform(40,48),
            "lng": random.uniform(-125,-70),
            "cases": cases,
            "status": status,
            "type": "farm",
            "last_update": f"2025-03-{random.randint(1,31):02d}",
            "trend": random.choice(["rising","stable","falling"]),
            "lineages": random.sample(lineages, random.randint(1,4)),
            "risk_score": risk_score
        })

# Live updates (60 entries) – expanded
update_texts = [
    "New PRRSV records pulled from NCBI (45 genomes).",
    "FASTQ batch queued for de novo assembly.",
    "Immune escape variant predicted in lineage L1.",
    "Metadata validation completed for 12 samples.",
    "Surveillance map updated with new cases.",
    "Quantum kernel training finished.",
    "GNN epistatic network updated – 4 new edges.",
    "High‑risk variant detected in OH_24055.",
    "Synchronized with NCBI – 23 new genomes.",
    "Manifold embedding started for 30 genomes.",
    "Quantum SVM predicted 3 new escape mutations.",
    "Heatmap layer updated with recent outbreaks.",
    "Database backup completed.",
    "Recombination event detected in lineage L5.",
    "GNN training 80% complete.",
    "Metadata uploaded for 8 samples.",
    "Fitness landscape updated – new peaks.",
    "External database sync finished.",
    "Critical variant identified in Texas.",
    "Topological data analysis completed.",
    "Phylogenetic tree updated with 12 new sequences.",
    "Molecular clock analysis suggests faster divergence.",
    "Outbreak source likely in Iowa.",
    "New escape mutation N32S confirmed in vitro."
]
LIVE_UPDATES = []
for h in range(9, 18):
    for m in random.sample(range(0,60), 4):
        LIVE_UPDATES.append({
            "time": f"{h:02d}:{m:02d}",
            "tag": random.choice(["Sync","Run","Alert","Meta","Map","System","AI"]),
            "text": random.choice(update_texts)
        })

# Analysis modules (unchanged)
ANALYSIS_MODULES = [
    {
        "group": "Genomic Processing",
        "items": [
            "Quality assessment (FastQC)",
            "Adapter trimming (Trimmomatic)",
            "Reference-guided assembly (Bowtie2)",
            "De novo assembly (SPAdes)",
            "Consensus generation (BCFTools)",
            "Mutation detection (SNP-sites)",
            "Coverage analysis (mosdepth)",
            "Variant calling (GATK)"
        ],
    },
    {
        "group": "Evolutionary Modeling",
        "items": [
            "Recombination analysis (RDP5)",
            "Selection pressure (PAML)",
            "Epistatic network (GNN)",
            "Topological data analysis (Persistent homology)",
            "Manifold learning (Diffusion maps)",
            "Fitness landscape (VAE)",
            "Phylogenetic inference (IQ-TREE)",
            "Molecular clock (BEAST)"
        ],
    },
    {
        "group": "Predictive Intelligence",
        "items": [
            "Mutation trajectory prediction (LSTM)",
            "Immune escape prediction (Quantum SVM)",
            "Antigenic mapping (AlphaFold2)",
            "Risk scoring (ensemble)",
            "Graph neural network (GCN)",
            "Evolutionary manifold embedding",
            "Outbreak source tracking (Maximum Likelihood)",
            "Vaccine match prediction"
        ],
    },
]

# Raw and assembly results (expanded)
RAW_RESULTS = {
    "qc_pass_rate": f"{random.uniform(92,98):.1f}%",
    "assembled_genomes": random.randint(25,40),
    "mean_coverage": f"{random.uniform(85,120):.1f}x",
    "recombination_events": random.randint(2,8),
    "high_risk_variants": random.randint(3,10),
    "total_reads": f"{random.randint(150,300)}M",
    "avg_read_length": f"{random.randint(140,160)} bp",
}

ASSEMBLY_RESULTS = {
    "uploaded_genomes": random.randint(40,70),
    "metadata_linked": random.randint(35,65),
    "lineage_clusters": random.randint(4,9),
    "escape_candidates": random.randint(2,7),
    "avg_genome_length": f"{random.randint(15200,15600)} bp",
    "n50": f"{random.randint(15200,15600)}",
}

# Epistatic network edges (300 edges) – expanded
mutations_all = ["N32S","E89K","G152R","A213T","S315I","M401L","T45A","V112I","P61S","L124F","A12V","R98K","D215N","Y246C","F307L","H39Y","L50Q","G64R","S77N","T83I"]
EPISTATIC_NETWORK_EDGES = []
for _ in range(300):
    src, tgt = random.sample(mutations_all, 2)
    EPISTATIC_NETWORK_EDGES.append({
        "source": src,
        "target": tgt,
        "weight": round(random.uniform(0.2,0.98),2)
    })

PREDICTED_ESCAPE_MUTATIONS = random.sample(mutations_all, random.randint(8,15))

# Manifold points (150 samples) – expanded
MANIFOLD_POINTS = []
for i in range(150):
    sample = f"PRRSV_{random.choice(['MN','IA','IL','NE','NC','OH','TX','CA'])}_24{random.randint(100,999)}"
    lineage = random.choice(lineages)
    MANIFOLD_POINTS.append({
        "x": round(random.uniform(0,1),2),
        "y": round(random.uniform(0,1),2),
        "lineage": lineage,
        "sample": sample
    })

# Dense heatmap data (50 positions x 80 samples) – expanded
HEATMAP_DATA = []
for i in range(50):
    row = [round(random.uniform(0,1),2) for _ in range(80)]
    HEATMAP_DATA.append(row)

# Phylogenetic tree (larger) – keep original, add BEAST output separately
PHYLOGENETIC_TREE = {
    "name": "root",
    "children": [
        {
            "name": f"Lineage {lin}",
            "children": [{"name": f"PRRSV_{random.choice(['MN','IA','IL'])}_24{random.randint(100,999)}", "size": 1} for _ in range(random.randint(3,8))]
        }
        for lin in ["L1","L2","L3","L4","L5","L6","L7","L8","L9"]
    ]
}

# New endpoints for trends and timelines (expanded)
FARM_TIMELINE = {}
for farm in CURRENT_SITUATION:
    farm_name = farm["site"]
    FARM_TIMELINE[farm_name] = [random.randint(max(0, farm["cases"]-10), farm["cases"]+10) for _ in range(14)]

MUTATION_TRENDS = {
    "dates": [f"2025-03-{d:02d}" for d in range(1,29)],
    "frequencies": {
        mut: [round(random.uniform(0.1,0.9),2) for _ in range(28)]
        for mut in mutations_all[:8]
    }
}

LINEAGE_TRENDS = {
    "dates": [f"2025-03-{d:02d}" for d in range(1,29)],
    "counts": {
        lin: [random.randint(0,20) for _ in range(28)]
        for lin in lineages
    }
}

# Dashboard summary stats (expanded)
DASHBOARD_STATS = {
    "total_genomes": random.randint(250,350),
    "active_outbreaks": random.randint(5,15),
    "escape_variants": len(PREDICTED_ESCAPE_MUTATIONS),
    "recombination_events": random.randint(10,30),
    "lineage_counts": {lin: random.randint(5,40) for lin in lineages},
    "risk_distribution": {
        "Critical": random.randint(2,8),
        "High": random.randint(5,15),
        "Moderate": random.randint(10,25),
        "Monitor": random.randint(15,30)
    }
}

# ---------- NEW DUMMY DATA FOR ADDITIONAL PLOTS ----------

# BEAST output (molecular clock tree with divergence times)
def generate_beast_tree(depth=3, prefix="Lineage"):
    if depth == 0:
        return {"name": f"Sample_{random.randint(1,100)}", "value": random.uniform(0,1)}
    else:
        children = [generate_beast_tree(depth-1, f"{prefix}{i}") for i in range(random.randint(2,4))]
        return {"name": f"Node_{prefix}", "children": children, "divergence_time": random.uniform(0,10)}

BEAST_OUTPUT = {
    "tree": generate_beast_tree(3),
    "posterior": [random.uniform(0.8,1.0) for _ in range(10)],
    "ess_values": [random.randint(200,500) for _ in range(5)]
}

# Forecast data (14-day prediction of escape scores)
FORECAST_DATA = {
    "dates": [f"2025-04-{d:02d}" for d in range(1,15)],
    "mean": [round(random.uniform(0.4,0.9),2) for _ in range(14)],
    "lower_ci": [round(random.uniform(0.3,0.5),2) for _ in range(14)],
    "upper_ci": [round(random.uniform(0.8,1.0),2) for _ in range(14)]
}

# Recombination hotspots (genome positions with probability)
RECOMBINATION_HOTSPOTS = {
    "positions": [i*300 for i in range(1,51)],
    "probability": [round(random.uniform(0.1,0.9),2) for _ in range(50)]
}

# Mutation correlation matrix (simulated)
mutation_corr_mtx = []
for i in range(10):
    row = []
    for j in range(10):
        if i == j:
            row.append(1.0)
        else:
            row.append(round(random.uniform(-0.3,0.8),2))
    mutation_corr_mtx.append(row)
MUTATION_CORRELATION = {
    "mutations": mutations_all[:10],
    "matrix": mutation_corr_mtx
}

# Lineage sunburst data
LINEAGE_SUNBURST = {
    "name": "All",
    "children": [
        {
            "name": f"L{i}",
            "children": [
                {"name": f"Subtype_{chr(65+j)}", "value": random.randint(5,20)}
                for j in range(random.randint(2,5))
            ]
        }
        for i in range(1,6)
    ]
}

# ------------------------------------------------------------
# API ENDPOINTS (all original + new ones)
# ------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs():
    return JSONResponse(JOBS)

@app.get("/api/results")
def api_results():
    return JSONResponse(RESULTS)

@app.get("/api/current-situation")
def api_current_situation():
    return JSONResponse(CURRENT_SITUATION)

@app.get("/api/live-updates")
def api_live_updates():
    return JSONResponse(LIVE_UPDATES)

@app.get("/api/analysis-modules")
def api_analysis_modules():
    return JSONResponse(ANALYSIS_MODULES)

@app.get("/api/raw-results")
def api_raw_results():
    return JSONResponse(RAW_RESULTS)

@app.get("/api/assembly-results")
def api_assembly_results():
    return JSONResponse(ASSEMBLY_RESULTS)

@app.get("/api/epistatic-network-edges")
def api_epistatic_network_edges():
    return JSONResponse(EPISTATIC_NETWORK_EDGES)

@app.get("/api/predicted-escape")
def api_predicted_escape():
    return JSONResponse(PREDICTED_ESCAPE_MUTATIONS)

@app.get("/api/manifold-points")
def api_manifold_points():
    return JSONResponse(MANIFOLD_POINTS)

@app.get("/api/heatmap")
def api_heatmap():
    return JSONResponse(HEATMAP_DATA)

@app.get("/api/phylogenetic-tree")
def api_phylogenetic_tree():
    return JSONResponse(PHYLOGENETIC_TREE)

@app.get("/api/mutation-details/{sample}")
def api_mutation_details(sample: str):
    for res in RESULTS:
        if res["sample"] == sample:
            return JSONResponse({"mutations": res["mutations_list"]})
    return JSONResponse({"mutations": []})

@app.get("/api/simulation-parameters")
def api_simulation_parameters():
    return JSONResponse({
        "mutation_rate": 0.023,
        "recombination_rate": 0.017,
        "immune_pressure": 0.85,
    })

@app.get("/api/prediction-update")
def api_prediction_update(mutation_rate: float = 0.02, recombination_rate: float = 0.015):
    new_escape_score = min(1.0, max(0.0, 0.5 + mutation_rate*10 + recombination_rate*5))
    new_risk = "High" if new_escape_score > 0.7 else "Moderate" if new_escape_score > 0.4 else "Monitor"
    return JSONResponse({
        "escape_score": round(new_escape_score, 2),
        "risk": new_risk,
        "predicted_mutations": random.sample(PREDICTED_ESCAPE_MUTATIONS, 2) if new_escape_score > 0.6 else ["T45A"],
    })

@app.get("/api/farm-timeline/{farm}")
def api_farm_timeline(farm: str):
    farm = farm.replace("%20", " ")
    if farm in FARM_TIMELINE:
        return JSONResponse({"timeline": FARM_TIMELINE[farm]})
    return JSONResponse({"timeline": []})

@app.get("/api/mutation-trends")
def api_mutation_trends():
    return JSONResponse(MUTATION_TRENDS)

@app.get("/api/lineage-trends")
def api_lineage_trends():
    return JSONResponse(LINEAGE_TRENDS)

@app.get("/api/dashboard-stats")
def api_dashboard_stats():
    return JSONResponse(DASHBOARD_STATS)

# ---------- NEW ENDPOINTS ----------
@app.get("/api/beast-output")
def api_beast_output():
    return JSONResponse(BEAST_OUTPUT)

@app.get("/api/forecast")
def api_forecast():
    return JSONResponse(FORECAST_DATA)

@app.get("/api/recombination-hotspots")
def api_recombination_hotspots():
    return JSONResponse(RECOMBINATION_HOTSPOTS)

@app.get("/api/mutation-correlation")
def api_mutation_correlation():
    return JSONResponse(MUTATION_CORRELATION)

@app.get("/api/lineage-sunburst")
def api_lineage_sunburst():
    return JSONResponse(LINEAGE_SUNBURST)

# ------------------------------------------------------------
# HTML PAGE (the entire frontend) – FULLY REVAMPED UI
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():

    return HTMLResponse(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PRRSV Evolutionary Intelligence Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        :root {
            --bg: #f3f7fc;
            --panel: rgba(255,255,255,0.88);
            --panel-solid: #ffffff;
            --line: #d8e3ef;
            --line-soft: #e7eef6;
            --text: #0f172a;
            --muted: #5b6b82;
            --primary: #2563eb;
            --primary-2: #38bdf8;
            --violet: #7c3aed;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #ef4444;
            --pink: #ec4899;
            --surface-grad: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            --shadow-lg: 0 28px 64px rgba(15,23,42,.10);
            --shadow-md: 0 14px 30px rgba(15,23,42,.07);
            --radius-xl: 30px;
            --radius-lg: 22px;
            --radius-md: 16px;
        }

        * { box-sizing: border-box; }
        html, body {
            margin: 0;
            min-height: 100%;
            background:
                radial-gradient(circle at 10% 0%, rgba(59,130,246,.11), transparent 26%),
                radial-gradient(circle at 100% 10%, rgba(14,165,233,.12), transparent 22%),
                linear-gradient(180deg,#f8fbff 0%, #f2f6fb 100%);
            color: var(--text);
            font-family: Inter, sans-serif;
            scroll-behavior: smooth;
        }
        body { padding: 0; }
        button, input, select { font: inherit; }
        a { color: inherit; text-decoration: none; }

        .app-shell {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .topbar {
            position: sticky;
            top: 0;
            z-index: 100;
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: center;
            gap: 18px;
            padding: 16px 26px;
            background: rgba(255,255,255,.74);
            backdrop-filter: blur(24px);
            border-bottom: 1px solid rgba(216,227,239,.85);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 14px;
        }
        .brand-mark {
            width: 54px;
            height: 54px;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 24px;
            box-shadow: 0 18px 28px rgba(37,99,235,.24);
        }
        .brand-title {
            font-family: "Space Grotesk", Inter, sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -.04em;
        }
        .brand-sub {
            font-size: 12px;
            color: var(--muted);
            font-weight: 700;
            margin-top: 2px;
        }

        .nav-wrap {
            display: flex;
            justify-content: center;
        }
        .nav-pills {
            display: inline-flex;
            gap: 8px;
            padding: 8px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.75);
            border-radius: 999px;
            box-shadow: var(--shadow-md);
            flex-wrap: wrap;
        }
        .nav-btn {
            border: none;
            background: transparent;
            color: var(--muted);
            padding: 12px 18px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 800;
            cursor: pointer;
            transition: .2s ease;
        }
        .nav-btn:hover { background: rgba(37,99,235,.08); color: var(--text); }
        .nav-btn.active {
            background: linear-gradient(135deg, #eff6ff, #f0f9ff);
            color: var(--primary);
            box-shadow: inset 0 0 0 1px #bfdbfe;
        }

        .toolbar-right {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .badge-pill, .icon-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.84);
            color: var(--muted);
            border-radius: 999px;
            font-weight: 800;
            box-shadow: var(--shadow-md);
        }
        .badge-pill { padding: 10px 14px; font-size: 13px; }
        .icon-btn { width: 42px; height: 42px; font-size: 16px; }

        .page {
            width: min(1680px, calc(100% - 36px));
            margin: 20px auto 26px;
            display: flex;
            flex-direction: column;
            gap: 22px;
        }

        .panel {
            background: var(--panel);
            border: 1px solid rgba(216,227,239,.88);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-lg);
            overflow: hidden;
            backdrop-filter: blur(16px);
        }
        .panel-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            padding: 18px 22px;
            border-bottom: 1px solid var(--line-soft);
            background: linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(248,251,255,.86) 100%);
        }
        .panel-head h2, .panel-head h3 {
            margin: 0;
            font-size: 18px;
            font-weight: 900;
            letter-spacing: -.03em;
        }
        .panel-sub {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            margin-top: 4px;
        }
        .panel-body { padding: 20px 22px 22px; }
        .panel-body.tight { padding: 14px; }

        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 0% 20%, rgba(37,99,235,.12), transparent 28%),
                radial-gradient(circle at 100% 20%, rgba(14,165,233,.14), transparent 25%),
                linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.05fr .95fr;
            gap: 22px;
            padding: 30px;
        }
        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: var(--primary);
            font-size: 12px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: 16px;
        }
        .hero h1 {
            margin: 0;
            font-family: "Space Grotesk", Inter, sans-serif;
            font-size: 54px;
            line-height: 1;
            letter-spacing: -.06em;
            max-width: 900px;
        }
        .hero p {
            margin: 18px 0 22px;
            max-width: 860px;
            color: var(--muted);
            line-height: 1.75;
            font-size: 16px;
        }
        .hero-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 24px;
        }
        .btn-primary, .btn-secondary {
            border: none;
            border-radius: 18px;
            padding: 14px 18px;
            font-size: 14px;
            font-weight: 900;
            cursor: pointer;
            transition: .2s ease;
        }
        .btn-primary {
            color: white;
            background: linear-gradient(135deg, var(--primary), var(--primary-2));
            box-shadow: 0 18px 28px rgba(37,99,235,.26);
        }
        .btn-secondary {
            background: #fff;
            color: var(--text);
            border: 1px solid var(--line);
        }
        .btn-primary:hover, .btn-secondary:hover { transform: translateY(-1px); }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        .kpi-card {
            padding: 16px;
            border-radius: 20px;
            background: rgba(255,255,255,.88);
            border: 1px solid var(--line-soft);
        }
        .kpi-value {
            font-size: 30px;
            font-weight: 900;
            letter-spacing: -.04em;
        }
        .kpi-label {
            margin-top: 6px;
            font-size: 13px;
            color: var(--muted);
            font-weight: 700;
        }

        .hero-preview {
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 12px;
            min-height: 470px;
            border-radius: 28px;
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(248,251,255,.98) 100%);
            padding: 14px;
        }
        .mini-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .window-dots { display: flex; gap: 7px; }
        .window-dots span {
            width: 10px; height: 10px; border-radius: 50%; background: #d6e0ec;
        }
        .preview-title { color: var(--muted); font-size: 13px; font-weight: 800; }
        .preview-grid {
            display: grid;
            grid-template-columns: 1.02fr 1.15fr .95fr;
            gap: 12px;
            min-height: 0;
        }
        .soft-card {
            border: 1px solid var(--line-soft);
            border-radius: 22px;
            background: rgba(255,255,255,.92);
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow: hidden;
        }
        .soft-title { font-size: 14px; font-weight: 900; }
        .menu-stack, .pred-stack { display: flex; flex-direction: column; gap: 8px; }
        .menu-item {
            padding: 12px;
            border-radius: 14px;
            font-size: 13px;
            font-weight: 800;
            color: var(--muted);
            background: #f5f9ff;
            border: 1px solid var(--line);
        }
        .menu-item.active { background: #eff6ff; color: var(--primary); border-color: #bfdbfe; }
        .dropzone {
            min-height: 120px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1.6px dashed #b9cfea;
            border-radius: 18px;
            background: linear-gradient(180deg,#fbfdff 0%,#f4f8ff 100%);
            color: var(--muted);
            font-weight: 700;
            padding: 18px;
        }
        .node-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .node {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 72px;
            padding: 12px;
            border-radius: 16px;
            background: white;
            border: 1px solid var(--line);
            text-align: center;
            font-size: 13px;
            font-weight: 900;
            line-height: 1.35;
        }
        .node.primary { background: linear-gradient(180deg,#eff6ff 0%,#fff 100%); border-color: #bfdbfe; color: var(--primary); }
        .node.green { background: linear-gradient(180deg,#ecfdf5 0%,#fff 100%); border-color: #bbf7d0; color: #047857; }
        .node.amber { background: linear-gradient(180deg,#fff7ed 0%,#fff 100%); border-color: #fed7aa; color: #c2410c; }
        .node.violet { background: linear-gradient(180deg,#f5f3ff 0%,#fff 100%); border-color: #ddd6fe; color: #6d28d9; }
        .pred-box {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: #fff;
            padding: 12px;
        }
        .pred-box h4 { margin: 0 0 8px; font-size: 12px; color: var(--muted); font-weight: 800; }
        .pred-bar {
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
        }
        .pred-score { margin-top: 10px; font-size: 30px; font-weight: 900; }

        .content-grid {
            display: grid;
            grid-template-columns: 295px minmax(0,1fr);
            gap: 22px;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .quick-link {
            padding: 14px;
            border-radius: 16px;
            border: 1px solid var(--line);
            background: #fff;
            font-weight: 800;
            color: var(--muted);
            cursor: pointer;
            transition: .2s ease;
        }
        .quick-link:hover { color: var(--text); border-color: #bed1ea; transform: translateY(-1px); }
        .stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            padding: 9px 0;
            border-bottom: 1px dashed var(--line-soft);
            font-size: 14px;
            font-weight: 700;
        }
        .stat-row:last-child { border-bottom: none; }
        .stat-row span:last-child { color: var(--muted); }

        .workspace-grid, .split-grid, .tri-grid {
            display: grid;
            gap: 22px;
        }
        .workspace-grid { grid-template-columns: 1.18fr .92fr; }
        .split-grid { grid-template-columns: 1fr 1fr; }
        .tri-grid { grid-template-columns: repeat(3, 1fr); gap: 16px; }

        .hook-box {
            border: 1.6px dashed #bcd0ea;
            background: linear-gradient(180deg,#fbfdff 0%,#f6faff 100%);
            border-radius: 18px;
            padding: 18px;
            color: var(--muted);
            line-height: 1.7;
        }
        .hook-box h4 {
            margin: 0 0 10px;
            color: var(--text);
            font-size: 15px;
            font-weight: 900;
        }
        .hook-box code {
            display: block;
            margin-top: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: #fff;
            white-space: pre-wrap;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px;
            color: #334155;
        }

        .chart-box, .plot-box, .network-box, .table-wrap {
            width: 100%;
            min-height: 320px;
            border: 1px solid var(--line);
            border-radius: 22px;
            background: linear-gradient(180deg,#ffffff 0%,#f9fbff 100%);
            overflow: hidden;
        }
        .plot-box.large { min-height: 420px; }
        .plot-box.xl { min-height: 520px; }
        .network-box { min-height: 420px; }
        .table-wrap { min-height: unset; overflow: auto; }

        .table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .table th, .table td {
            text-align: left;
            padding: 14px 12px;
            border-bottom: 1px solid var(--line-soft);
            vertical-align: top;
        }
        .table thead th {
            position: sticky;
            top: 0;
            background: #fbfdff;
            z-index: 2;
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-weight: 900;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            padding: 7px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 900;
            border: 1px solid transparent;
        }
        .pill.blue { color: var(--primary); background: #eff6ff; border-color: #dbeafe; }
        .pill.green { color: #047857; background: #ecfdf5; border-color: #bbf7d0; }
        .pill.amber { color: #c2410c; background: #fff7ed; border-color: #fed7aa; }
        .pill.red { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
        .pill.violet { color: #6d28d9; background: #f5f3ff; border-color: #ddd6fe; }

        .workspace-toolbar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .mini-chip {
            padding: 9px 12px;
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.84);
            font-size: 12px;
            font-weight: 900;
            color: var(--muted);
        }

        .subnav {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 8px;
        }
        .subnav-btn {
            border: 1px solid var(--line);
            background: #fff;
            color: var(--muted);
            padding: 10px 14px;
            border-radius: 14px;
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
        }
        .subnav-btn.active { background: #eff6ff; color: var(--primary); border-color: #bfdbfe; }

        .hidden { display: none !important; }

        .footer-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            padding: 4px 0 18px;
        }

        @media (max-width: 1320px) {
            .hero-grid, .content-grid, .workspace-grid, .split-grid, .preview-grid {
                grid-template-columns: 1fr;
            }
            .kpi-grid, .tri-grid { grid-template-columns: repeat(2, 1fr); }
        }

        @media (max-width: 820px) {
            .page { width: min(100% - 16px, 100%); }
            .topbar {
                grid-template-columns: 1fr;
                justify-items: center;
                text-align: center;
                gap: 14px;
            }
            .hero h1 { font-size: 38px; }
            .kpi-grid, .tri-grid { grid-template-columns: 1fr; }
            .panel-head { align-items: flex-start; flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="app-shell">
    <header class="topbar">
        <div class="brand">
            <div class="brand-mark">P</div>
            <div>
                <div class="brand-title">PRRSV Evolutionary Intelligence</div>
                <div class="brand-sub">high-end research interface · live predictive analytics</div>
            </div>
        </div>

        <div class="nav-wrap">
            <div class="nav-pills">
                <button class="nav-btn active" data-page="landing">Landing</button>
                <button class="nav-btn" data-page="workspace">Workspace</button>
                <button class="nav-btn" data-page="analysis">Analysis</button>
                <button class="nav-btn" data-page="evolution">Evolution</button>
                <button class="nav-btn" data-page="results">Results</button>
                <button class="nav-btn" data-page="intelligence">Intelligence</button>
            </div>
        </div>

        <div class="toolbar-right">
            <div class="badge-pill" id="liveChip">Queue · --</div>
            <div class="icon-btn" title="Notifications">🔔</div>
            <div class="icon-btn" title="Profile">NJ</div>
        </div>
    </header>

    <main class="page">
        <section id="page-landing">
            <div class="panel hero">
                <div class="hero-grid">
                    <div>
                        <div class="eyebrow">AI-driven predictive surveillance</div>
                        <h1>Genomic analysis, evolutionary modeling, and interactive PRRSV intelligence in one modern workspace</h1>
                        <p>
                            Upload raw reads or assemblies, attach epidemiological metadata, launch modular genomic workflows,
                            visualize 3D evolutionary structure, explore mutation networks, evaluate immune escape, and keep
                            every placeholder ready for direct connection to your real analysis code without deleting existing logic.
                        </p>
                        <div class="hero-actions">
                            <button class="btn-primary" onclick="goPage('workspace')">Open workspace</button>
                            <button class="btn-secondary" onclick="goPage('evolution')">Explore 3D evolution</button>
                        </div>
                        <div class="kpi-grid" id="heroKpis"></div>
                    </div>

                    <div class="hero-preview">
                        <div class="mini-bar">
                            <div class="window-dots"><span></span><span></span><span></span></div>
                            <div class="preview-title">Live platform overview</div>
                        </div>
                        <div class="preview-grid">
                            <div class="soft-card">
                                <div class="soft-title">Data ingestion</div>
                                <div class="menu-stack">
                                    <div class="menu-item active">Load raw FASTQ / FASTA</div>
                                    <div class="menu-item">Link metadata</div>
                                    <div class="menu-item">Sync external database</div>
                                </div>
                                <div class="dropzone">Attach raw reads, assemblies, metadata, or batch jobs</div>
                            </div>
                            <div class="soft-card">
                                <div class="soft-title">Pipeline orchestration</div>
                                <div class="node-grid">
                                    <div class="node primary">Quality control</div>
                                    <div class="node green">Assembly</div>
                                    <div class="node">Variant detection</div>
                                    <div class="node amber">Recombination</div>
                                    <div class="node">Annotation</div>
                                    <div class="node violet">AI prediction</div>
                                </div>
                            </div>
                            <div class="soft-card">
                                <div class="soft-title">Prediction layer</div>
                                <div class="pred-stack">
                                    <div class="pred-box">
                                        <h4>Mutation risk forecast</h4>
                                        <div class="pred-bar"></div>
                                    </div>
                                    <div class="pred-box">
                                        <h4>Immune escape</h4>
                                        <div class="pred-score" id="heroEscapeScore">--%</div>
                                    </div>
                                    <div class="pred-box">
                                        <h4>Current risk level</h4>
                                        <div id="heroRiskPill"><span class="pill amber">Loading</span></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="content-grid">
                <aside class="sidebar">
                    <div class="panel">
                        <div class="panel-head">
                            <div>
                                <h3>Quick access</h3>
                                <div class="panel-sub">Multi-page scientific interface</div>
                            </div>
                        </div>
                        <div class="panel-body">
                            <div class="quick-link" onclick="goPage('workspace')">Load genomic data</div>
                            <div class="quick-link" onclick="goPage('analysis')">Configure modular analysis</div>
                            <div class="quick-link" onclick="goPage('evolution')">Open 3D manifold + landscape</div>
                            <div class="quick-link" onclick="goPage('results')">Review result tables</div>
                            <div class="quick-link" onclick="goPage('intelligence')">Open outbreak intelligence</div>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-head">
                            <div>
                                <h3>System status</h3>
                                <div class="panel-sub">Directly linked to live endpoints</div>
                            </div>
                        </div>
                        <div class="panel-body" id="statusPanel"></div>
                    </div>
                </aside>

                <section class="workspace-grid">
                    <div class="panel">
                        <div class="panel-head">
                            <div>
                                <h2>3D evolutionary landscape</h2>
                                <div class="panel-sub">Interactive surface + sample manifold + lineage overlay</div>
                            </div>
                        </div>
                        <div class="panel-body tight">
                            <div id="landingEvolution3D" class="plot-box xl"></div>
                        </div>
                    </div>

                    <div class="panel">
                        <div class="panel-head">
                            <div>
                                <h2>Current situation map</h2>
                                <div class="panel-sub">Live outbreak spread, site risk, and lineage activity</div>
                            </div>
                        </div>
                        <div class="panel-body tight">
                            <div id="landingMap" class="plot-box xl"></div>
                        </div>
                    </div>
                </section>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Code attachment zones</h2>
                            <div class="panel-sub">Keep your real backend, replace only rendering hooks</div>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div class="hook-box">
                            <h4>Safe integration pattern</h4>
                            Attach your real callbacks to these visual actions instead of deleting old code.
                            <code>POST /api/run-qc
POST /api/run-assembly
POST /api/run-variant-call
POST /api/run-evolution-model
POST /api/run-prediction

// keep existing logic intact:
def run_existing_pipeline(payload):
    return your_real_2338_line_logic(payload)</code>
                        </div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Platform modules</h2>
                            <div class="panel-sub">Proposal-aligned system architecture</div>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div class="tri-grid" id="landingModules"></div>
                    </div>
                </div>
            </div>
        </section>

        <section id="page-workspace" class="hidden">
            <div class="panel">
                <div class="panel-head">
                    <div>
                        <h2>Interactive workbench</h2>
                        <div class="panel-sub">Premium white scientific workspace with analysis attachment slots</div>
                    </div>
                    <div class="workspace-toolbar">
                        <div class="mini-chip">Workspace state: live</div>
                        <div class="mini-chip">Hybrid DB linked</div>
                        <div class="mini-chip">Upload ready</div>
                    </div>
                </div>
                <div class="panel-body">
                    <div class="split-grid">
                        <div class="panel" style="box-shadow:none;">
                            <div class="panel-head">
                                <div>
                                    <h3>Load data</h3>
                                    <div class="panel-sub">Raw reads, assemblies, metadata, external sync</div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="dropzone" style="min-height:220px;">Drop FASTQ / FASTA / metadata / batch archives here</div>
                                <div style="height:16px;"></div>
                                <div class="tri-grid">
                                    <div class="hook-box"><h4>Raw read mode</h4>Trigger QC, trimming, reference-guided or de novo assembly.</div>
                                    <div class="hook-box"><h4>Assembly mode</h4>Load preassembled genomes and metadata-driven downstream analysis.</div>
                                    <div class="hook-box"><h4>Database sync</h4>Pull PRRSV records from local curated DB and external repositories.</div>
                                </div>
                            </div>
                        </div>
                        <div class="panel" style="box-shadow:none;">
                            <div class="panel-head">
                                <div>
                                    <h3>Pipeline builder</h3>
                                    <div class="panel-sub">Do not remove real code — bind each node to it</div>
                                </div>
                            </div>
                            <div class="panel-body">
                                <div class="node-grid">
                                    <div class="node primary">FastQC / QC metrics</div>
                                    <div class="node">Adapter trimming</div>
                                    <div class="node green">Reference-guided assembly</div>
                                    <div class="node green">De novo assembly</div>
                                    <div class="node">Consensus generation</div>
                                    <div class="node">Variant calling</div>
                                    <div class="node amber">Recombination detection</div>
                                    <div class="node amber">Selection pressure</div>
                                    <div class="node violet">TDA + manifold learning</div>
                                    <div class="node violet">Mutation prediction</div>
                                </div>
                                <div style="height:18px;"></div>
                                <div class="hook-box">
                                    <h4>Launch hooks</h4>
                                    <code>onclick=runQC()
onclick=runAssembly()
onclick=runVariants()
onclick=runTopology()
onclick=runQuantumPrediction()</code>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div style="height:22px"></div>
                    <div class="split-grid">
                        <div class="panel" style="box-shadow:none;">
                            <div class="panel-head"><div><h3>Job queue</h3><div class="panel-sub">Live processing and workflow state</div></div></div>
                            <div class="panel-body tight"><div id="workspaceJobs" class="table-wrap"></div></div>
                        </div>
                        <div class="panel" style="box-shadow:none;">
                            <div class="panel-head"><div><h3>Current situation</h3><div class="panel-sub">Map panel for active outbreak monitoring</div></div></div>
                            <div class="panel-body tight"><div id="workspaceMap" class="plot-box large"></div></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="page-analysis" class="hidden">
            <div class="panel">
                <div class="panel-head">
                    <div>
                        <h2>Analysis architecture</h2>
                        <div class="panel-sub">Visual modules, data flow, and hook points for your real backend</div>
                    </div>
                    <div class="subnav">
                        <button class="subnav-btn active">Genomic processing</button>
                        <button class="subnav-btn">Evolutionary structure</button>
                        <button class="subnav-btn">Predictive AI</button>
                    </div>
                </div>
                <div class="panel-body">
                    <div class="tri-grid" id="analysisModuleCards"></div>
                </div>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Heatmap + recombination layer</h2>
                            <div class="panel-sub">Genome-position heatmap for attached real outputs</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="heatmapPlot" class="plot-box large"></div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Recombination hotspots</h2>
                            <div class="panel-sub">Genome-wide probability profile</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="hotspotPlot" class="plot-box large"></div>
                    </div>
                </div>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Mutation correlation</h2>
                            <div class="panel-sub">Interactive matrix for co-mutation structure</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="corrPlot" class="plot-box large"></div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Real code integration guide</h2>
                            <div class="panel-sub">Zero-deletion backend binding pattern</div>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div class="hook-box">
                            <h4>Recommended wrapper strategy</h4>
                            Keep all legacy computations in separate functions, then map new UI requests into those functions.
                            <code>@app.post('/api/run-topology')
def run_topology(payload: dict):
    return existing_topology_pipeline(payload)

@app.post('/api/run-risk-score')
def run_risk(payload: dict):
    return existing_ai_prediction(payload)</code>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <section id="page-evolution" class="hidden">
            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>3D evolutionary manifold</h2>
                            <div class="panel-sub">Interactive scatter manifold with lineage-aware projection</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="manifold3D" class="plot-box xl"></div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Fitness landscape surface</h2>
                            <div class="panel-sub">High-end 3D surface with projected mutation trajectories</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="surface3D" class="plot-box xl"></div>
                    </div>
                </div>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Epistatic mutation network</h2>
                            <div class="panel-sub">Interactive graph with hover, zoom, and cluster density</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="epistaticNetwork" class="network-box"></div>
                    </div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Phylogeny + BEAST summary</h2>
                            <div class="panel-sub">Evolution structure and posterior diagnostics</div>
                        </div>
                    </div>
                    <div class="panel-body tight">
                        <div id="beastTreePlot" class="plot-box large"></div>
                        <div style="height:16px"></div>
                        <div id="beastDiagPlot" class="plot-box large"></div>
                    </div>
                </div>
            </div>
        </section>

        <section id="page-results" class="hidden">
            <div class="panel">
                <div class="panel-head">
                    <div>
                        <h2>Results explorer</h2>
                        <div class="panel-sub">Sample-level risk, mutation burden, lineage, recombination, immune escape</div>
                    </div>
                </div>
                <div class="panel-body tight">
                    <div id="resultsTable" class="table-wrap"></div>
                </div>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Risk distribution</h2>
                            <div class="panel-sub">Overall sample distribution by predicted threat level</div>
                        </div>
                    </div>
                    <div class="panel-body tight"><div id="resultsRiskPlot" class="plot-box large"></div></div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Lineage composition</h2>
                            <div class="panel-sub">Sunburst exploration of lineage and subtype hierarchy</div>
                        </div>
                    </div>
                    <div class="panel-body tight"><div id="sunburstPlot" class="plot-box large"></div></div>
                </div>
            </div>
        </section>

        <section id="page-intelligence" class="hidden">
            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Forecast and alerting</h2>
                            <div class="panel-sub">Predicted escape trend with confidence interval</div>
                        </div>
                    </div>
                    <div class="panel-body tight"><div id="forecastPlot" class="plot-box large"></div></div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Predicted escape mutations</h2>
                            <div class="panel-sub">Latest alert-ready candidate mutations</div>
                        </div>
                    </div>
                    <div class="panel-body" id="predictedEscapeList"></div>
                </div>
            </div>

            <div class="split-grid">
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Mutation trends</h2>
                            <div class="panel-sub">Temporal mutation frequency trajectories</div>
                        </div>
                    </div>
                    <div class="panel-body tight"><div id="mutationTrendPlot" class="plot-box large"></div></div>
                </div>
                <div class="panel">
                    <div class="panel-head">
                        <div>
                            <h2>Live intelligence feed</h2>
                            <div class="panel-sub">Recent updates, sync events, alerts, and runtime activity</div>
                        </div>
                    </div>
                    <div class="panel-body" id="liveFeed"></div>
                </div>
            </div>
        </section>

        <div class="footer-bar" id="footerBar"></div>
    </main>
</div>

<script>
const pages = ['landing','workspace','analysis','evolution','results','intelligence'];
const state = { cache: {} };

function goPage(name) {
    pages.forEach(p => document.getElementById('page-' + p).classList.toggle('hidden', p !== name));
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.page === name));
    if (name === 'evolution') renderEvolutionPage();
    if (name === 'analysis') renderAnalysisPage();
    if (name === 'results') renderResultsPage();
    if (name === 'intelligence') renderIntelligencePage();
    if (name === 'workspace') renderWorkspacePage();
}

document.querySelectorAll('.nav-btn').forEach(btn => btn.addEventListener('click', () => goPage(btn.dataset.page)));

async function getJSON(url) {
    if (state.cache[url]) return state.cache[url];
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed: ' + url);
    const data = await res.json();
    state.cache[url] = data;
    return data;
}

function asPill(text) {
    const t = String(text).toLowerCase();
    let cls = 'blue';
    if (t.includes('critical') || t.includes('high')) cls = 'red';
    else if (t.includes('moderate') || t.includes('potential')) cls = 'amber';
    else if (t.includes('monitor') || t.includes('unlikely') || t.includes('no')) cls = 'green';
    else if (t.includes('l')) cls = 'blue';
    return `<span class="pill ${cls}">${text}</span>`;
}

function footerChips(items) {
    document.getElementById('footerBar').innerHTML = items.map(x => `<div class="badge-pill">${x}</div>`).join('');
}

function lineColorForRisk(risk) {
    if (risk === 'Critical' || risk === 'High') return '#ef4444';
    if (risk === 'Moderate') return '#f59e0b';
    return '#10b981';
}

function lineageColor(lineage) {
    const palette = {
        L1:'#2563eb', L2:'#0ea5e9', L3:'#10b981', L4:'#22c55e', L5:'#f59e0b',
        L6:'#ef4444', L7:'#ec4899', L8:'#7c3aed', L9:'#64748b'
    };
    return palette[lineage] || '#2563eb';
}

async function boot() {
    const [stats, results, pred, mods] = await Promise.all([
        getJSON('/api/dashboard-stats'),
        getJSON('/api/results'),
        getJSON('/api/prediction-update'),
        getJSON('/api/analysis-modules')
    ]);

    document.getElementById('liveChip').textContent = `Queue · ${stats.active_outbreaks} outbreaks · ${stats.escape_variants} alerts`;
    document.getElementById('heroEscapeScore').textContent = `${Math.round(pred.escape_score * 100)}%`;
    document.getElementById('heroRiskPill').innerHTML = asPill(pred.risk);

    document.getElementById('heroKpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${stats.total_genomes}</div><div class="kpi-label">Genomes indexed</div></div>
        <div class="kpi-card"><div class="kpi-value">${stats.active_outbreaks}</div><div class="kpi-label">Active outbreaks</div></div>
        <div class="kpi-card"><div class="kpi-value">${stats.escape_variants}</div><div class="kpi-label">Escape alerts</div></div>
        <div class="kpi-card"><div class="kpi-value">${stats.recombination_events}</div><div class="kpi-label">Recombination events</div></div>
    `;

    document.getElementById('statusPanel').innerHTML = `
        <div class="stat-row"><span>Total genomes</span><span>${stats.total_genomes}</span></div>
        <div class="stat-row"><span>Outbreaks</span><span>${stats.active_outbreaks}</span></div>
        <div class="stat-row"><span>Escape variants</span><span>${stats.escape_variants}</span></div>
        <div class="stat-row"><span>Recombination events</span><span>${stats.recombination_events}</span></div>
        <div class="stat-row"><span>Top risk class</span><span>${Object.keys(stats.risk_distribution)[0]}</span></div>
    `;

    document.getElementById('landingModules').innerHTML = mods.map(group => `
        <div class="hook-box"><h4>${group.group}</h4>${group.items.map(x => '• ' + x).join('<br>')}</div>
    `).join('');

    footerChips([
        `${stats.total_genomes} genomes`,
        `${results.length} sample outputs`,
        `${stats.escape_variants} escape alerts`,
        `hybrid DB ready`,
        `3D evolution enabled`
    ]);

    await renderLanding();
}

async function renderLanding() {
    const [mapData, manifold, pred] = await Promise.all([
        getJSON('/api/current-situation'),
        getJSON('/api/manifold-points'),
        getJSON('/api/prediction-update')
    ]);

    renderScatterGeo('landingMap', mapData);
    renderCombinedEvolution3D('landingEvolution3D', manifold, pred.escape_score);
}

function renderScatterGeo(targetId, rows) {
    const trace = {
        type: 'scattergeo',
        mode: 'markers',
        lat: rows.map(r => r.lat),
        lon: rows.map(r => r.lng),
        text: rows.map(r => `${r.site}<br>Status: ${r.status}<br>Cases: ${r.cases}<br>Risk score: ${r.risk_score}`),
        marker: {
            size: rows.map(r => 8 + (r.cases || 5) * 0.35),
            color: rows.map(r => lineColorForRisk(r.status)),
            opacity: 0.82,
            line: { color: '#ffffff', width: 1.2 }
        },
        hovertemplate: '%{text}<extra></extra>'
    };
    Plotly.newPlot(targetId, [trace], {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        geo: {
            scope: 'north america',
            projection: { type: 'mercator' },
            showland: true,
            landcolor: '#f8fbff',
            showcountries: true,
            countrycolor: '#d8e3ef',
            showlakes: true,
            lakecolor: '#eef7ff',
            bgcolor: 'rgba(0,0,0,0)',
            coastlinecolor: '#c7d6e7',
            subunitcolor: '#d6e0ec'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    }, { responsive: true, displayModeBar: false });
}

function makeSurfaceData(scaleFactor=1) {
    const xs = [];
    const ys = [];
    const z = [];
    for (let i = 0; i < 40; i++) {
        xs.push(i / 4);
        ys.push(i / 4);
    }
    for (let i = 0; i < 40; i++) {
        const row = [];
        const y = ys[i];
        for (let j = 0; j < 40; j++) {
            const x = xs[j];
            const peak1 = Math.exp(-((x - 3.5)**2 + (y - 4.5)**2) / 2.8) * 1.8;
            const peak2 = Math.exp(-((x - 7.5)**2 + (y - 2.8)**2) / 1.8) * 1.3;
            const peak3 = Math.exp(-((x - 5.8)**2 + (y - 7.4)**2) / 3.2) * 1.6;
            const ridge = Math.sin(x * 1.15) * 0.16 + Math.cos(y * 1.1) * 0.12;
            row.push((peak1 + peak2 + peak3 + ridge) * scaleFactor);
        }
        z.push(row);
    }
    return { xs, ys, z };
}

function renderCombinedEvolution3D(targetId, manifold, escapeScore) {
    const surf = makeSurfaceData(1 + (escapeScore || 0.5) * 0.2);
    const groups = {};
    manifold.forEach((p, idx) => {
        const lin = p.lineage || 'L?';
        if (!groups[lin]) groups[lin] = {x:[], y:[], z:[], text:[], color: lineageColor(lin)};
        const surfaceZ = surf.z[Math.min(39, Math.floor(p.y * 39))][Math.min(39, Math.floor(p.x * 39))] + 0.12;
        groups[lin].x.push(p.x * 10);
        groups[lin].y.push(p.y * 10);
        groups[lin].z.push(surfaceZ);
        groups[lin].text.push(`${p.sample}<br>${lin}`);
    });

    const traces = [{
        type: 'surface',
        x: surf.xs,
        y: surf.ys,
        z: surf.z,
        opacity: 0.95,
        colorscale: [
            [0,'#dbeafe'], [0.18,'#93c5fd'], [0.36,'#60a5fa'], [0.54,'#10b981'],
            [0.72,'#f59e0b'], [0.88,'#f97316'], [1,'#ef4444']
        ],
        contours: { z: { show: true, usecolormap: true, highlightwidth: 1 } },
        showscale: false,
        hoverinfo: 'skip'
    }];

    Object.entries(groups).forEach(([lin, g]) => {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: g.x, y: g.y, z: g.z,
            name: lin,
            text: g.text,
            hovertemplate: '%{text}<extra></extra>',
            marker: { size: 4.5, color: g.color, opacity: 0.92, line: { color: '#ffffff', width: 0.6 } }
        });
    });

    traces.push({
        type: 'scatter3d',
        mode: 'lines+markers',
        x: [1, 2.2, 3.6, 5.1, 6.8, 8.2],
        y: [8.2, 7.4, 6.1, 5.4, 4.3, 3.4],
        z: [0.8, 1.05, 1.42, 1.7, 1.36, 1.15],
        line: { color: '#ffffff', width: 6 },
        marker: { size: 4, color: '#2563eb' },
        name: 'Predicted trajectory',
        hovertemplate: 'Predicted evolutionary trajectory<extra></extra>'
    });

    Plotly.newPlot(targetId, traces, {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: { title: 'Evolution axis 1', gridcolor: '#dce7f2', zerolinecolor: '#dce7f2', backgroundcolor: '#f9fbff' },
            yaxis: { title: 'Evolution axis 2', gridcolor: '#dce7f2', zerolinecolor: '#dce7f2', backgroundcolor: '#f9fbff' },
            zaxis: { title: 'Fitness / risk', gridcolor: '#dce7f2', zerolinecolor: '#dce7f2', backgroundcolor: '#f9fbff' },
            camera: { eye: { x: 1.5, y: 1.45, z: 0.95 } }
        },
        legend: { orientation: 'h', y: 1.02, x: 0.02, bgcolor: 'rgba(255,255,255,.7)' }
    }, { responsive: true, displaylogo: false });
}

async function renderWorkspacePage() {
    const [jobs, mapData] = await Promise.all([getJSON('/api/jobs'), getJSON('/api/current-situation')]);
    document.getElementById('workspaceJobs').innerHTML = `
        <table class="table">
            <thead><tr><th>ID</th><th>Job</th><th>Status</th><th>Progress</th><th>ETA</th><th>Model</th></tr></thead>
            <tbody>${jobs.slice(0, 14).map(j => `
                <tr>
                    <td><strong>${j.id}</strong></td>
                    <td>${j.title}</td>
                    <td>${asPill(j.status)}</td>
                    <td>${j.progress}%</td>
                    <td>${j.eta}</td>
                    <td>${j.model}</td>
                </tr>
            `).join('')}</tbody>
        </table>`;
    renderScatterGeo('workspaceMap', mapData);
}

async function renderAnalysisPage() {
    const [mods, heatmap, hotspots, corr] = await Promise.all([
        getJSON('/api/analysis-modules'),
        getJSON('/api/heatmap'),
        getJSON('/api/recombination-hotspots'),
        getJSON('/api/mutation-correlation')
    ]);

    document.getElementById('analysisModuleCards').innerHTML = mods.map(group => `
        <div class="hook-box"><h4>${group.group}</h4>${group.items.map(x => '• ' + x).join('<br>')}</div>
    `).join('');

    Plotly.newPlot('heatmapPlot', [{
        type: 'heatmap',
        z: heatmap,
        colorscale: [
            [0,'#eff6ff'], [0.2,'#bfdbfe'], [0.4,'#60a5fa'], [0.6,'#10b981'], [0.8,'#f59e0b'], [1,'#ef4444']
        ],
        showscale: false,
        hovertemplate: 'Genome position %{y}<br>Sample %{x}<br>Signal %{z}<extra></extra>'
    }], {
        margin: { l: 56, r: 14, t: 18, b: 42 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Samples', gridcolor: '#e8eef5' },
        yaxis: { title: 'Genome positions', gridcolor: '#e8eef5' }
    }, { responsive: true, displayModeBar: false });

    Plotly.newPlot('hotspotPlot', [{
        x: hotspots.positions,
        y: hotspots.probability,
        mode: 'lines+markers',
        line: { color: '#ef4444', width: 3, shape: 'spline' },
        marker: { size: 6, color: '#f59e0b' },
        fill: 'tozeroy',
        fillcolor: 'rgba(239,68,68,.12)',
        hovertemplate: 'Position %{x}<br>Probability %{y}<extra></extra>'
    }], {
        margin: { l: 56, r: 18, t: 16, b: 48 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Genome position', gridcolor: '#e8eef5' },
        yaxis: { title: 'Hotspot probability', gridcolor: '#e8eef5', range: [0,1] }
    }, { responsive: true, displayModeBar: false });

    Plotly.newPlot('corrPlot', [{
        type: 'heatmap',
        x: corr.mutations,
        y: corr.mutations,
        z: corr.matrix,
        zmid: 0,
        colorscale: [
            [0,'#1d4ed8'], [0.5,'#f8fafc'], [1,'#dc2626']
        ],
        hovertemplate: '%{x} × %{y}<br>Correlation %{z}<extra></extra>'
    }], {
        margin: { l: 78, r: 20, t: 18, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    }, { responsive: true, displayModeBar: false });
}

async function renderEvolutionPage() {
    const [manifold, pred, edges, beast] = await Promise.all([
        getJSON('/api/manifold-points'),
        getJSON('/api/prediction-update'),
        getJSON('/api/epistatic-network-edges'),
        getJSON('/api/beast-output')
    ]);

    renderManifold3D('manifold3D', manifold);
    renderSurface('surface3D', pred.escape_score);
    renderEpistaticNetwork('epistaticNetwork', edges);
    renderBeastPlots(beast);
}

function renderManifold3D(targetId, points) {
    const groups = {};
    points.forEach(p => {
        if (!groups[p.lineage]) groups[p.lineage] = {x:[], y:[], z:[], text:[], color: lineageColor(p.lineage)};
        groups[p.lineage].x.push(p.x);
        groups[p.lineage].y.push(p.y);
        groups[p.lineage].z.push((p.x * 0.45) + (p.y * 0.55) + Math.random() * 0.15);
        groups[p.lineage].text.push(`${p.sample}<br>${p.lineage}`);
    });
    const traces = Object.entries(groups).map(([lin, g]) => ({
        type: 'scatter3d',
        mode: 'markers',
        name: lin,
        x: g.x, y: g.y, z: g.z,
        text: g.text,
        hovertemplate: '%{text}<extra></extra>',
        marker: { size: 5.5, color: g.color, opacity: 0.88, line: { width: 0.6, color: '#ffffff' } }
    }));
    Plotly.newPlot(targetId, traces, {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: { title: 'Manifold 1', gridcolor: '#dce7f2' },
            yaxis: { title: 'Manifold 2', gridcolor: '#dce7f2' },
            zaxis: { title: 'Trajectory depth', gridcolor: '#dce7f2' },
            camera: { eye: { x: 1.35, y: 1.35, z: 0.95 } }
        },
        legend: { bgcolor: 'rgba(255,255,255,.75)' }
    }, { responsive: true, displaylogo: false });
}

function renderSurface(targetId, escapeScore) {
    const surf = makeSurfaceData(1 + (escapeScore || 0.4) * 0.35);
    const trajectory = { x: [], y: [], z: [] };
    for (let i = 0; i < 11; i++) {
        const x = 1.2 + i * 0.7;
        const y = 8.0 - i * 0.45;
        const xi = Math.min(39, Math.max(0, Math.round((x / 10) * 39)));
        const yi = Math.min(39, Math.max(0, Math.round((y / 10) * 39)));
        trajectory.x.push(x);
        trajectory.y.push(y);
        trajectory.z.push(surf.z[yi][xi] + 0.06);
    }

    Plotly.newPlot(targetId, [
        {
            type: 'surface',
            x: surf.xs, y: surf.ys, z: surf.z,
            colorscale: [
                [0,'#dbeafe'], [0.18,'#93c5fd'], [0.38,'#34d399'], [0.56,'#facc15'], [0.76,'#fb923c'], [1,'#ef4444']
            ],
            showscale: false,
            contours: { z: { show: true, usecolormap: true, highlightcolor: '#94a3b8', project: { z: true } } }
        },
        {
            type: 'scatter3d',
            mode: 'lines+markers',
            x: trajectory.x, y: trajectory.y, z: trajectory.z,
            line: { color: '#ffffff', width: 7 },
            marker: { size: 4.5, color: '#2563eb' },
            name: 'Predicted evolutionary path'
        }
    ], {
        margin: { l: 0, r: 0, t: 0, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
            bgcolor: 'rgba(0,0,0,0)',
            xaxis: { title: 'Genome manifold x', gridcolor: '#dce7f2' },
            yaxis: { title: 'Genome manifold y', gridcolor: '#dce7f2' },
            zaxis: { title: 'Fitness', gridcolor: '#dce7f2' },
            camera: { eye: { x: 1.6, y: 1.35, z: 0.9 } }
        }
    }, { responsive: true, displaylogo: false });
}

function renderEpistaticNetwork(targetId, edges) {
    const nodeMap = new Map();
    const edgeList = [];
    edges.slice(0, 110).forEach(e => {
        if (!nodeMap.has(e.source)) nodeMap.set(e.source, { id: e.source, label: e.source, color: '#60a5fa', font: { color: '#0f172a', face: 'Inter' } });
        if (!nodeMap.has(e.target)) nodeMap.set(e.target, { id: e.target, label: e.target, color: '#f59e0b', font: { color: '#0f172a', face: 'Inter' } });
        edgeList.push({ from: e.source, to: e.target, value: e.weight, color: { color: 'rgba(100,116,139,.35)' }, width: 1 + (e.weight * 2) });
    });
    const nodes = new vis.DataSet(Array.from(nodeMap.values()));
    const network = new vis.Network(document.getElementById(targetId), { nodes, edges: edgeList }, {
        autoResize: true,
        nodes: { shape: 'dot', size: 18, borderWidth: 1.2 },
        edges: { smooth: { type: 'dynamic' } },
        physics: { stabilization: false, barnesHut: { gravitationalConstant: -3500, springLength: 110, springConstant: 0.028 } },
        interaction: { hover: true, multiselect: false },
        layout: { improvedLayout: true }
    });
}

function flattenTree(node, parent=null, depth=0, out=[]) {
    const id = out.length + 1;
    out.push({ id, parent, name: node.name, depth, x: depth, y: Math.random() });
    if (node.children) node.children.forEach(ch => flattenTree(ch, id, depth + 1, out));
    return out;
}

function renderBeastPlots(beast) {
    const flat = flattenTree(beast.tree);
    const nodePos = {};
    flat.forEach((n, i) => { nodePos[n.id] = { x: n.x, y: i * 0.22 }; });
    const xs = [], ys = [], texts = [], parents = [];
    flat.forEach(n => {
        xs.push(nodePos[n.id].x);
        ys.push(nodePos[n.id].y);
        texts.push(n.name);
        if (n.parent) {
            const p = nodePos[n.parent];
            parents.push({ x: [p.x, n.x, null], y: [p.y, n.y, null] });
        }
    });
    const edgeX = [], edgeY = [];
    parents.forEach(e => { edgeX.push(...e.x); edgeY.push(...e.y); });
    Plotly.newPlot('beastTreePlot', [
        { x: edgeX, y: edgeY, mode: 'lines', line: { color: '#94a3b8', width: 1.5 }, hoverinfo: 'skip', showlegend: false },
        { x: xs, y: ys, mode: 'markers+text', text: texts, textposition: 'middle right', marker: { size: 8, color: '#2563eb' }, hovertemplate: '%{text}<extra></extra>', showlegend: false }
    ], {
        margin: { l: 40, r: 20, t: 10, b: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { visible: false },
        yaxis: { visible: false }
    }, { responsive: true, displayModeBar: false });

    Plotly.newPlot('beastDiagPlot', [
        {
            x: beast.posterior.map((_, i) => 'Posterior ' + (i + 1)),
            y: beast.posterior,
            type: 'bar',
            name: 'Posterior',
            marker: { color: '#7c3aed' }
        },
        {
            x: beast.ess_values.map((_, i) => 'ESS ' + (i + 1)),
            y: beast.ess_values,
            type: 'bar',
            yaxis: 'y2',
            name: 'ESS',
            marker: { color: '#10b981' }
        }
    ], {
        margin: { l: 50, r: 50, t: 10, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { gridcolor: '#e8eef5' },
        yaxis: { title: 'Posterior', gridcolor: '#e8eef5' },
        yaxis2: { title: 'ESS', overlaying: 'y', side: 'right' },
        legend: { orientation: 'h' }
    }, { responsive: true, displayModeBar: false });
}

async function renderResultsPage() {
    const [rows, stats, sunburst] = await Promise.all([
        getJSON('/api/results'),
        getJSON('/api/dashboard-stats'),
        getJSON('/api/lineage-sunburst')
    ]);

    document.getElementById('resultsTable').innerHTML = `
        <table class="table">
            <thead>
                <tr>
                    <th>Sample</th><th>Lineage</th><th>Mutations</th><th>Recombination</th><th>Escape score</th><th>Immune escape</th><th>Risk</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(r => `
                    <tr>
                        <td><strong>${r.sample}</strong><br><span style="color:var(--muted);font-size:12px;">${(r.mutations_list||[]).slice(0,5).join(', ')}</span></td>
                        <td>${asPill(r.lineage)}</td>
                        <td>${r.mutations}</td>
                        <td>${asPill(r.recomb)}</td>
                        <td>${r.escape_score}</td>
                        <td>${asPill(r.immune_escape)}</td>
                        <td>${asPill(r.risk)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>`;

    const riskKeys = Object.keys(stats.risk_distribution);
    const riskVals = Object.values(stats.risk_distribution);
    Plotly.newPlot('resultsRiskPlot', [{
        type: 'bar',
        x: riskKeys,
        y: riskVals,
        marker: { color: ['#ef4444','#fb7185','#f59e0b','#10b981'] },
        hovertemplate: '%{x}: %{y}<extra></extra>'
    }], {
        margin: { l: 50, r: 20, t: 10, b: 44 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { gridcolor: '#e8eef5' },
        yaxis: { title: 'Samples', gridcolor: '#e8eef5' }
    }, { responsive: true, displayModeBar: false });

    const labels = [], parents = [], values = [];
    function walk(node, parent='') {
        labels.push(node.name);
        parents.push(parent);
        values.push(node.value || 0);
        (node.children || []).forEach(ch => walk(ch, node.name));
    }
    walk(sunburst, '');
    Plotly.newPlot('sunburstPlot', [{
        type: 'sunburst', labels, parents, values,
        branchvalues: 'total',
        insidetextorientation: 'radial',
        marker: { colors: ['#2563eb','#38bdf8','#10b981','#f59e0b','#ef4444','#7c3aed','#ec4899'] }
    }], {
        margin: { l: 0, r: 0, t: 10, b: 0 },
        paper_bgcolor: 'rgba(0,0,0,0)'
    }, { responsive: true, displayModeBar: false });
}

async function renderIntelligencePage() {
    const [forecast, predMuts, mutTrends, live] = await Promise.all([
        getJSON('/api/forecast'),
        getJSON('/api/predicted-escape'),
        getJSON('/api/mutation-trends'),
        getJSON('/api/live-updates')
    ]);

    Plotly.newPlot('forecastPlot', [
        {
            x: forecast.dates,
            y: forecast.upper_ci,
            mode: 'lines',
            line: { width: 0 },
            hoverinfo: 'skip',
            showlegend: false
        },
        {
            x: forecast.dates,
            y: forecast.lower_ci,
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(37,99,235,.12)',
            line: { width: 0 },
            hoverinfo: 'skip',
            showlegend: false
        },
        {
            x: forecast.dates,
            y: forecast.mean,
            mode: 'lines+markers',
            line: { color: '#2563eb', width: 3, shape: 'spline' },
            marker: { size: 6, color: '#38bdf8' },
            name: 'Predicted escape score'
        }
    ], {
        margin: { l: 56, r: 20, t: 10, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Date', gridcolor: '#e8eef5' },
        yaxis: { title: 'Escape score', gridcolor: '#e8eef5', range: [0, 1.05] }
    }, { responsive: true, displayModeBar: false });

    document.getElementById('predictedEscapeList').innerHTML = `
        <div class="tri-grid">
            ${predMuts.map(m => `<div class="hook-box"><h4>${m}</h4>Predicted escape-associated mutation. Attach real evidence panel, structure impact, and vaccine relevance here.</div>`).join('')}
        </div>`;

    const traces = Object.entries(mutTrends.frequencies).map(([mut, vals]) => ({
        x: mutTrends.dates,
        y: vals,
        mode: 'lines',
        name: mut,
        line: { width: 2.4, shape: 'spline' }
    }));
    Plotly.newPlot('mutationTrendPlot', traces, {
        margin: { l: 56, r: 16, t: 10, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'Date', gridcolor: '#e8eef5' },
        yaxis: { title: 'Frequency', gridcolor: '#e8eef5', range: [0,1] },
        legend: { orientation: 'h', y: 1.12 }
    }, { responsive: true, displayModeBar: false });

    document.getElementById('liveFeed').innerHTML = live.slice(0, 18).map(item => `
        <div class="stat-row"><span><strong>${item.time}</strong> · ${item.tag}</span><span>${item.text}</span></div>
    `).join('');
}

boot();
</script>
</body>
</html>
    """)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
