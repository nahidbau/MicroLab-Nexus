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


HEATMAP_DATA = []
for i in range(50):
    row = [round(random.uniform(0,1),2) for _ in range(80)]
    HEATMAP_DATA.append(row)


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


FORECAST_DATA = {
    "dates": [f"2025-04-{d:02d}" for d in range(1,15)],
    "mean": [round(random.uniform(0.4,0.9),2) for _ in range(14)],
    "lower_ci": [round(random.uniform(0.3,0.5),2) for _ in range(14)],
    "upper_ci": [round(random.uniform(0.8,1.0),2) for _ in range(14)]
}


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
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRRSV Evolutionary Intelligence · v11.5</title>
    <!-- Leaflet with extra plugins for heatmap and clustering -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <!-- Chart.js v4 -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <!-- D3.js -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <!-- Three.js core -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.27.1.min.js" charset="utf-8"></script>
    <!-- Font Awesome 6 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <!-- Google Fonts: Inter & Roboto Mono -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,100..900;1,100..900&family=Roboto+Mono:ital,wght@0,100..700;1,100..700&display=swap" rel="stylesheet">
    <style>
        /* ---------- DYNAMIC SCIENTIFIC COLOR PALETTE (unchanged but extended) ---------- */
        :root {
            --bg-deep: #0a0f1e;
            --bg-panel: #141c2b;
            --bg-card: #1e2a3a;
            --border-glow: #2a3f5e;
            --text-primary: #e2e9f5;
            --text-secondary: #a0b8d4;
            --accent-cyan: #0ff0fc;
            --accent-blue: #3b82f6;
            --accent-purple: #a78bfa;
            --accent-magenta: #f472b6;
            --accent-green: #4ade80;
            --accent-yellow: #fbbf24;
            --accent-red: #f87171;
            --grad-1: linear-gradient(145deg, #0ff0fc, #3b82f6);
            --grad-2: linear-gradient(145deg, #a78bfa, #f472b6);
            --grad-3: linear-gradient(145deg, #fbbf24, #f87171);
            --shadow-heavy: 0 20px 40px -15px #00000080;
            --shadow-glow: 0 0 20px var(--accent-cyan);
            --font-primary: 'Inter', sans-serif;
            --font-mono: 'Roboto Mono', monospace;
            --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        [data-theme="light"] {
            --bg-deep: #f0f7ff;
            --bg-panel: #ffffff;
            --bg-card: #f9fcff;
            --border-glow: #cbd5e1;
            --text-primary: #0f172a;
            --text-secondary: #334155;
            --accent-cyan: #0891b2;
            --accent-blue: #2563eb;
            --accent-purple: #7c3aed;
            --accent-magenta: #db2777;
            --accent-green: #16a34a;
            --accent-yellow: #d97706;
            --accent-red: #dc2626;
            --shadow-heavy: 0 20px 40px -15px #94a3b880;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html, body {
            margin: 0;
            padding: 0;
            background: var(--bg-deep);
            color: var(--text-primary);
            font-family: var(--font-primary);
            transition: background 0.3s, color 0.2s;
            line-height: 1.6;
            font-size: 16px;
            scroll-behavior: smooth;
        }

        a { text-decoration: none; color: inherit; }

        /* topnav, brand, navlinks, userbox – unchanged */
        .topnav {
            position: sticky; top: 0; z-index: 1000;
            background: rgba(20, 28, 43, 0.7);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-bottom: 1px solid var(--border-glow);
            display: flex; align-items: center; justify-content: space-between;
            padding: 12px 32px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-mark {
            width: 48px; height: 48px; border-radius: 16px;
            background: var(--grad-1);
            color: #fff; font-weight: 900; font-size: 28px;
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 10px 20px -5px var(--accent-cyan);
            transition: var(--transition-smooth);
            animation: pulse 3s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 var(--accent-cyan); }
            70% { box-shadow: 0 0 20px 10px rgba(15,240,252,0); }
            100% { box-shadow: 0 0 0 0 rgba(15,240,252,0); }
        }
        .brand-mark:hover { transform: rotate(10deg) scale(1.1); }
        .brand-title {
            font-size: 20px; font-weight: 900; letter-spacing: -0.02em;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-cyan));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .brand-sub { font-size: 12px; color: var(--text-secondary); }

        .navlinks { display: flex; gap: 8px; }
        .navlink {
            padding: 12px 20px; border-radius: 16px; font-size: 15px; font-weight: 700;
            color: var(--text-secondary); transition: var(--transition-smooth);
            position: relative; overflow: hidden;
        }
        .navlink::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 3px;
            background: var(--accent-cyan);
            transition: width 0.3s;
        }
        .navlink:hover::after { width: 80%; }
        .navlink:hover { background: rgba(255,255,255,0.05); color: var(--text-primary); }
        .navlink.active {
            background: rgba(59,130,246,0.2); color: var(--accent-blue);
            box-shadow: 0 0 0 2px var(--accent-blue) inset;
        }

        .userbox { display: flex; align-items: center; gap: 20px; }
        .notification-bell { font-size: 22px; cursor: pointer; transition: var(--transition-smooth); animation: ring 4s infinite; }
        @keyframes ring {
            0% { transform: rotate(0); }
            5% { transform: rotate(15deg); }
            10% { transform: rotate(-15deg); }
            15% { transform: rotate(10deg); }
            20% { transform: rotate(-10deg); }
            25% { transform: rotate(5deg); }
            30% { transform: rotate(0); }
        }
        .notification-bell:hover { transform: scale(1.1); color: var(--accent-cyan); }
        .user-avatar {
            width: 42px; height: 42px; border-radius: 14px;
            background: var(--grad-2); color: white; display: flex;
            align-items: center; justify-content: center; font-weight: 900; cursor: pointer;
            transition: var(--transition-smooth);
        }
        .user-avatar:hover { transform: scale(1.05); box-shadow: 0 0 15px var(--accent-magenta); }
        .dark-mode-toggle { font-size: 24px; cursor: pointer; transition: var(--transition-smooth); }
        .dark-mode-toggle:hover { transform: rotate(30deg); }

        /* shell, fab, hero – unchanged */
        .shell { padding: 28px; max-width: 2000px; margin: 0 auto; }
        .fab {
            position: fixed; bottom: 32px; right: 32px; width: 68px; height: 68px;
            border-radius: 34px; background: var(--grad-1); color: white;
            display: flex; align-items: center; justify-content: center;
            font-size: 32px; box-shadow: 0 10px 30px var(--accent-cyan);
            cursor: pointer; transition: var(--transition-smooth); z-index: 99;
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0); }
        }
        .fab:hover { transform: scale(1.1) rotate(90deg); box-shadow: 0 15px 40px var(--accent-cyan); }

        .hero {
            background: var(--bg-panel); backdrop-filter: blur(8px);
            border: 1px solid var(--border-glow); border-radius: 36px;
            padding: 36px; box-shadow: var(--shadow-heavy); margin-bottom: 32px;
            transition: var(--transition-smooth);
        }
        .hero:hover {
            box-shadow: 0 30px 60px -15px var(--accent-blue);
            transform: translateY(-2px);
        }
        .eyebrow {
            font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em;
            font-weight: 900; color: var(--accent-cyan); margin-bottom: 12px;
        }
        .hero h1 {
            margin: 0 0 12px 0; font-size: 42px; font-weight: 900;
            background: linear-gradient(135deg, var(--text-primary), var(--accent-cyan));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p {
            margin: 0; max-width: 1000px; font-size: 16px; line-height: 1.7;
            color: var(--text-secondary);
        }

        /* grid system – unchanged */
        .grid { display: grid; grid-template-columns: 1fr 440px; gap: 32px; }
        .stack { display: flex; flex-direction: column; gap: 32px; }

        .card {
            background: var(--bg-panel); backdrop-filter: blur(8px);
            border: 1px solid var(--border-glow); border-radius: 36px;
            box-shadow: var(--shadow-heavy); padding: 28px;
            transition: var(--transition-smooth);
            position: relative;
            overflow: hidden;
        }
        .card::before {
            content: '';
            position: absolute;
            top: -2px; left: -2px; right: -2px; bottom: -2px;
            background: linear-gradient(45deg, var(--accent-cyan), transparent, var(--accent-purple));
            border-radius: 38px;
            z-index: -1;
            opacity: 0;
            transition: opacity 0.5s;
        }
        .card:hover::before { opacity: 0.2; }
        .card:hover {
            transform: translateY(-4px) scale(1.01);
            box-shadow: 0 30px 60px -15px #000;
            border-color: var(--accent-cyan);
        }
        .card-title {
            margin: 0 0 8px 0; font-size: 26px; font-weight: 900;
            display: flex; align-items: center; gap: 10px;
        }
        .card-sub { margin: 0 0 20px 0; font-size: 15px; color: var(--text-secondary); }

        /* pills – unchanged */
        .pill {
            display: inline-block; padding: 6px 16px; border-radius: 100px;
            font-size: 12px; font-weight: 900; text-transform: uppercase;
            transition: var(--transition-smooth);
        }
        .pill.qc-passed, .pill.completed { background: #1e3a5f; color: var(--accent-cyan); }
        .pill.annotated, .pill.validated { background: #1e3a5f; color: var(--accent-green); }
        .pill.processing { background: #5b2e3f; color: var(--accent-red); animation: pulse-bg 2s infinite; }
        @keyframes pulse-bg {
            0% { background: #5b2e3f; }
            50% { background: #7f3f4f; }
            100% { background: #5b2e3f; }
        }
        .pill.running { background: #2d3a4f; color: var(--accent-purple); }
        .pill.queued { background: #3f3a2e; color: var(--accent-yellow); }
        .pill.paused { background: #3f2e3a; color: var(--accent-magenta); }
        .pill.high { background: #3f2e2e; color: var(--accent-red); }
        .pill.moderate { background: #1e3a5f; color: var(--accent-blue); }
        .pill.critical { background: #4f2e2e; color: var(--accent-red); font-weight: 900; animation: blink 1s infinite; }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        .pill.monitor { background: #1e3a3a; color: var(--accent-green); }

        .side-card {
            background: var(--bg-panel); backdrop-filter: blur(8px);
            border: 1px solid var(--border-glow); border-radius: 32px;
            box-shadow: var(--shadow-heavy); padding: 24px;
            transition: var(--transition-smooth);
        }
        .side-card:hover {
            border-color: var(--accent-purple);
            transform: translateY(-2px);
        }
        .side-title {
            margin: 0 0 16px 0; font-size: 18px; font-weight: 900;
            display: flex; align-items: center; gap: 8px;
        }

        /* job cards – unchanged */
        .job {
            border: 1px solid var(--border-glow); border-radius: 24px; padding: 18px;
            margin-bottom: 14px; background: var(--bg-card); transition: var(--transition-smooth);
            cursor: pointer;
        }
        .job:hover {
            border-color: var(--accent-cyan); transform: scale(1.02) translateX(5px);
            box-shadow: 0 10px 20px rgba(0,255,255,0.3);
        }
        .job-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
        .job-id { font-size: 11px; color: var(--text-secondary); font-weight: 900; }
        .job-actions i { margin-left: 10px; opacity: 0.5; transition: var(--transition-smooth); }
        .job-actions i:hover { opacity: 1; color: var(--accent-cyan); transform: scale(1.2); }
        .job-name { font-size: 14px; font-weight: 700; margin-bottom: 10px; }
        .progress-bar { width: 100%; height: 8px; background: #2a3f5e; border-radius: 10px; overflow: hidden; margin: 10px 0 6px; }
        .progress-fill { height: 100%; background: var(--grad-1); border-radius: 10px; transition: width 0.5s ease; }
        .job-meta { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); }

        /* feed items – unchanged */
        .feed-item {
            border-bottom: 1px solid var(--border-glow); padding: 16px 0;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .feed-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
        .feed-time { font-size: 12px; color: var(--text-secondary); }
        .feed-tag {
            background: var(--bg-card); color: var(--accent-cyan);
            border-radius: 30px; padding: 4px 14px; font-size: 11px; font-weight: 900;
            transition: var(--transition-smooth);
        }
        .feed-tag:hover { background: var(--accent-cyan); color: black; }
        .feed-text { font-size: 14px; line-height: 1.5; }

        /* map container */
        .map-container {
            height: 400px;
            border-radius: 28px;
            overflow: hidden;
            border: 2px solid var(--border-glow);
            margin-bottom: 16px;
            transition: var(--transition-smooth);
            box-shadow: var(--shadow-heavy);
        }
        .map-container:hover {
            border-color: var(--accent-green);
            box-shadow: 0 0 30px var(--accent-green);
        }
        .leaflet-popup-content {
            background: var(--bg-card);
            color: var(--text-primary);
            border-radius: 12px;
        }
        .leaflet-container { background: var(--bg-deep); }

        /* charts – all now have explicit heights and backgrounds */
        .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 24px 0; }
        .chart-box {
            background: var(--bg-card); border-radius: 24px; padding: 24px;
            border: 1px solid var(--border-glow);
            transition: var(--transition-smooth);
        }
        .chart-box:hover {
            box-shadow: 0 10px 30px rgba(59,130,246,0.3);
            transform: scale(1.02);
        }

        .manifold-plot {
            background: var(--bg-card); border-radius: 24px; padding: 10px;
            border: 1px solid var(--border-glow); height: 400px;
        }

        .network-container {
            background: var(--bg-card); border-radius: 24px;
            border: 1px solid var(--border-glow); height: 400px;
            position: relative; overflow: hidden;
        }
        svg.network-svg { width: 100%; height: 100%; }

        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; }
        .stat-card {
            background: var(--bg-card); border: 1px solid var(--border-glow);
            border-radius: 24px; padding: 20px; text-align: center;
            transition: var(--transition-smooth);
        }
        .stat-card:hover {
            transform: translateY(-5px) scale(1.05);
            border-color: var(--accent-cyan);
            box-shadow: 0 10px 20px var(--accent-cyan);
        }
        .stat-value { font-size: 36px; font-weight: 900; color: var(--accent-cyan); }

        .escape-badge {
            background: rgba(192,132,252,0.2); border: 1px solid var(--accent-purple);
            color: var(--accent-purple); border-radius: 30px; padding: 8px 20px;
            display: inline-block; margin: 4px; font-weight: 700;
            transition: var(--transition-smooth);
        }
        .escape-badge:hover {
            background: var(--accent-purple); color: white; transform: scale(1.05) rotate(2deg);
            box-shadow: 0 0 15px var(--accent-purple);
        }

        .slider-container { margin: 20px 0; }
        .slider-label { display: flex; justify-content: space-between; }
        input[type=range] {
            width: 100%; background: var(--bg-card); height: 6px; border-radius: 3px;
            appearance: none;
        }
        input[type=range]::-webkit-slider-thumb {
            appearance: none; width: 20px; height: 20px;
            background: var(--accent-cyan); border-radius: 50%; cursor: pointer;
            transition: var(--transition-smooth);
        }
        input[type=range]::-webkit-slider-thumb:hover {
            transform: scale(1.3); box-shadow: 0 0 15px var(--accent-cyan);
        }

        .heatmap-container {
            background: var(--bg-card); border-radius: 24px;
            border: 1px solid var(--border-glow); height: 500px;
        }

        .landscape-container {
            width: 100%;
            height: 400px;
            background: var(--bg-card);
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid var(--border-glow);
            position: relative;
        }

        .loader {
            border: 4px solid var(--border-glow);
            border-top: 4px solid var(--accent-cyan);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .hidden { display: none !important; }
        .flex-row { display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        .glow-text { text-shadow: 0 0 10px var(--accent-cyan); }

        /* tabs for results page */
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border-glow);
            padding-bottom: 10px;
        }
        .tab {
            padding: 10px 20px;
            border-radius: 20px;
            background: var(--bg-card);
            cursor: pointer;
            transition: var(--transition-smooth);
            font-weight: 700;
        }
        .tab:hover { background: var(--accent-cyan); color: black; }
        .tab.active { background: var(--accent-blue); color: white; }

        .table-wrap {
            overflow-x: auto;
            max-height: 500px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th {
            background: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 700;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid var(--border-glow);
        }
        tr:hover { background: var(--bg-card); }

        .search-box {
            display: flex;
            align-items: center;
            background: var(--bg-card);
            border-radius: 30px;
            padding: 8px 16px;
            margin-bottom: 20px;
        }
        .search-box input {
            background: transparent;
            border: none;
            color: var(--text-primary);
            margin-left: 10px;
            width: 100%;
            outline: none;
        }
        .pagination {
            display: flex;
            gap: 8px;
            margin-top: 20px;
        }
        .page-btn {
            background: var(--bg-card);
            padding: 6px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: var(--transition-smooth);
        }
        .page-btn.active {
            background: var(--accent-cyan);
            color: black;
        }
        .page-btn:hover { background: var(--accent-blue); color: white; }

        @media (max-width: 1300px) {
            .grid { grid-template-columns: 1fr; }
            .chart-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <nav class="topnav">
        <div class="brand">
            <div class="brand-mark">P</div>
            <div>
                <div class="brand-title">PRRSV Evolutionary Intelligence</div>
                <div class="brand-sub">v11.5 · AI‑driven predictive platform</div>
            </div>
        </div>
        <div class="navlinks">
            <a class="navlink active" id="nav-dashboard" href="#" onclick="loadPage('dashboard')">Dashboard</a>
            <a class="navlink" id="nav-run" href="#" onclick="loadPage('run')">Run Analysis</a>
            <a class="navlink" id="nav-results" href="#" onclick="loadPage('results')">Results</a>
            <a class="navlink" id="nav-intelligence" href="#" onclick="loadPage('intelligence')">Intelligence</a>
        </div>
        <div class="userbox">
            <span class="notification-bell">🔔</span>
            <span class="user-avatar">UMN</span>
            <span class="dark-mode-toggle" id="darkModeToggle">🌙</span>
        </div>
    </nav>

    <div class="shell" id="main-content">
        <!-- Dynamic content loaded via JavaScript -->
        <div class="loader" id="global-loader" style="display:none;"></div>
    </div>

    <div class="fab" id="fab" onclick="fabAction()">⚡</div>

    <script>
        // ---------- GLOBAL UTILITIES (unchanged but enhanced) ----------
        const darkModeToggle = document.getElementById('darkModeToggle');
        const currentTheme = localStorage.getItem('theme') || 'dark';
        document.documentElement.setAttribute('data-theme', currentTheme);
        darkModeToggle.textContent = currentTheme === 'dark' ? '☀️' : '🌙';

        darkModeToggle.addEventListener('click', () => {
            let theme = document.documentElement.getAttribute('data-theme');
            theme = theme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem('theme', theme);
            darkModeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';
            // redraw all charts if needed (optional)
        });

        async function getJSON(url) {
            showLoader(true);
            try {
                const res = await fetch(url);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return await res.json();
            } finally {
                showLoader(false);
            }
        }

        function showLoader(show) {
            const loader = document.getElementById('global-loader');
            if (loader) loader.style.display = show ? 'block' : 'none';
        }

        function pillClass(value) {
            return String(value).toLowerCase().replace(/\\s+/g, '');
        }

        window.fabAction = function() {
            alert('🚀 Quick launch: New analysis wizard (simulated)');
        };

        function setActiveNav(page) {
            document.querySelectorAll('.navlink').forEach(link => link.classList.remove('active'));
            document.getElementById(`nav-${page}`).classList.add('active');
        }

        // ---------- PAGE LOADING ----------
        async function loadPage(page) {
            setActiveNav(page);
            const contentDiv = document.getElementById('main-content');
            contentDiv.innerHTML = '<div class="loader"></div>';
            if (page === 'dashboard') {
                contentDiv.innerHTML = await getDashboardHTML();
                initDashboard();
            } else if (page === 'run') {
                contentDiv.innerHTML = await getRunHTML();
                initRun();
            } else if (page === 'results') {
                contentDiv.innerHTML = await getResultsHTML();
                initResults();
            } else if (page === 'intelligence') {
                contentDiv.innerHTML = await getIntelligenceHTML();
                initIntelligence();
            }
        }

        window.onload = () => loadPage('dashboard');

        // ---------- DASHBOARD PAGE (enhanced with more KPIs) ----------
        async function getDashboardHTML() {
            return `
            <div class="hero">
                <div class="eyebrow">PRRSV Evolutionary Intelligence · Live Dashboard</div>
                <h1>Real‑time surveillance & predictive analytics</h1>
                <p>Key metrics, trends, and AI‑powered insights at a glance.</p>
            </div>

            <div class="stat-grid" style="margin-bottom:32px;" id="kpiCards"></div>

            <div class="grid">
                <div class="stack">
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-chart-line"></i> Mutation Trends (Last 28 days)</h2>
                        <div id="mutationTrendChart" style="height:300px;"></div>
                    </div>
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-chart-pie"></i> Risk Distribution</h2>
                        <div id="riskPieChart" style="height:300px;"></div>
                    </div>
                </div>
                <div class="stack">
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-map-location-dot"></i> Active Outbreaks</h2>
                        <div id="dashboardMap" class="map-container" style="height:300px;"></div>
                    </div>
                    <div class="side-card">
                        <h3 class="side-title"><i class="fa-solid fa-microchip"></i> Recent Jobs</h3>
                        <div id="dashboardJobs"></div>
                    </div>
                    <div class="side-card">
                        <h3 class="side-title"><i class="fa-solid fa-rss"></i> Live Intelligence</h3>
                        <div id="dashboardLive"></div>
                    </div>
                </div>
            </div>

            <div class="card" style="margin-top:32px;">
                <h2 class="card-title"><i class="fa-solid fa-chart-area"></i> Lineage Prevalence Over Time</h2>
                <div id="lineageAreaChart" style="height:400px;"></div>
            </div>
            `;
        }

        async function initDashboard() {
            const stats = await getJSON('/api/dashboard-stats');
            const kpiDiv = document.getElementById('kpiCards');
            kpiDiv.innerHTML = `
                <div class="stat-card"><div class="stat-value">${stats.total_genomes}</div><div>Genomes Sequenced</div></div>
                <div class="stat-card"><div class="stat-value">${stats.active_outbreaks}</div><div>Active Outbreaks</div></div>
                <div class="stat-card"><div class="stat-value">${stats.escape_variants}</div><div>Escape Variants</div></div>
                <div class="stat-card"><div class="stat-value">${stats.recombination_events}</div><div>Recombination Events</div></div>
            `;

            const mutTrends = await getJSON('/api/mutation-trends');
            const traces = Object.entries(mutTrends.frequencies).map(([mut, freqs]) => ({
                x: mutTrends.dates,
                y: freqs,
                mode: 'lines',
                name: mut,
                line: { width: 2 }
            }));
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { gridcolor: 'var(--border-glow)' },
                yaxis: { title: 'Frequency', gridcolor: 'var(--border-glow)' },
                margin: { l: 50, r: 30, t: 20, b: 50 }
            };
            Plotly.newPlot('mutationTrendChart', traces, layout, { responsive: true });

            const ctx = document.getElementById('riskPieChart').getContext('2d');
            new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: Object.keys(stats.risk_distribution),
                    datasets: [{
                        data: Object.values(stats.risk_distribution),
                        backgroundColor: ['#dc2626', '#f87171', '#3b82f6', '#4ade80'],
                        borderWidth: 0
                    }]
                },
                options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
            });

            const linTrends = await getJSON('/api/lineage-trends');
            const areaTraces = Object.entries(linTrends.counts).map(([lin, counts]) => ({
                x: linTrends.dates,
                y: counts,
                mode: 'none',
                stackgroup: 'one',
                name: lin,
                fill: 'tonexty'
            }));
            const areaLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { gridcolor: 'var(--border-glow)' },
                yaxis: { title: 'Count', gridcolor: 'var(--border-glow)' },
                margin: { l: 50, r: 30, t: 20, b: 50 }
            };
            Plotly.newPlot('lineageAreaChart', areaTraces, areaLayout, { responsive: true });

            initMap('dashboardMap', true);
            await loadJobs('dashboardJobs', 3);
            await loadLiveUpdates('dashboardLive', 3);
        }

        // ---------- RUN PAGE (unchanged) ----------
        async function getRunHTML() {
            return `
            <div class="hero">
                <div class="eyebrow">Configure Analysis</div>
                <h1>Run AI‑powered evolutionary workflows</h1>
                <p>Select input mode, tune parameters, and launch predictive pipelines.</p>
            </div>

            <div class="grid">
                <div class="stack">
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-sliders"></i> Input Mode</h2>
                        <div style="display:flex; gap:20px; margin-bottom:20px;">
                            <div class="choice active" id="rawChoice" onclick="window.selectMode('raw')" style="flex:1; border:2px solid var(--border-glow); border-radius:28px; padding:24px; background:var(--bg-card); transition:var(--transition-smooth); cursor:pointer;">
                                <h3><i class="fa-solid fa-dna"></i> Raw reads</h3>
                                <p>FASTQ → automated assembly + AI analysis</p>
                            </div>
                            <div class="choice" id="assemblyChoice" onclick="window.selectMode('assembly')" style="flex:1; border:2px solid var(--border-glow); border-radius:28px; padding:24px; background:var(--bg-card); transition:var(--transition-smooth); cursor:pointer;">
                                <h3><i class="fa-solid fa-file-code"></i> Assembled genomes</h3>
                                <p>FASTA + metadata → evolutionary modeling</p>
                            </div>
                        </div>

                        <div id="rawPanel">
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
                                <div><label>FASTQ files</label><input type="file" multiple style="width:100%; padding:10px; background:var(--bg-card); border:1px solid var(--border-glow); border-radius:12px; color:var(--text-primary);"></div>
                                <div><label>Read type</label><select style="width:100%; padding:10px; background:var(--bg-card); border:1px solid var(--border-glow); border-radius:12px; color:var(--text-primary);"><option>Paired-end</option></select></div>
                            </div>
                            <div style="margin:20px 0;">
                                <button class="btn" style="background:var(--grad-1); color:white; border:none; border-radius:20px; padding:14px 28px; font-weight:700; cursor:pointer; transition:var(--transition-smooth);" onclick="window.runRawPipeline()"><i class="fa-solid fa-play"></i> Run full pipeline</button>
                            </div>
                            <div id="rawResults" class="hidden">
                                <div class="stat-grid">
                                    <div class="stat-card"><div class="stat-value" id="raw_qc">-</div><div>QC pass</div></div>
                                    <div class="stat-card"><div class="stat-value" id="raw_asm">-</div><div>Assembled</div></div>
                                    <div class="stat-card"><div class="stat-value" id="raw_cov">-</div><div>Coverage</div></div>
                                    <div class="stat-card"><div class="stat-value" id="raw_rec">-</div><div>Recomb.</div></div>
                                    <div class="stat-card"><div class="stat-value" id="raw_hr">-</div><div>High risk</div></div>
                                </div>
                            </div>
                        </div>

                        <div id="assemblyPanel" class="hidden">
                            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-bottom:20px;">
                                <div><label>FASTA files</label><input type="file" multiple style="padding:10px; background:var(--bg-card); border:1px solid var(--border-glow); border-radius:12px; color:var(--text-primary);"></div>
                                <div><label>Metadata (CSV)</label><input type="file" style="padding:10px; background:var(--bg-card); border:1px solid var(--border-glow); border-radius:12px; color:var(--text-primary);"></div>
                            </div>
                            <div style="margin:20px 0;">
                                <button class="btn" style="background:var(--grad-1); color:white; border:none; border-radius:20px; padding:14px 28px; cursor:pointer; transition:var(--transition-smooth);" onclick="window.runAssemblyFlow()"><i class="fa-solid fa-link"></i> Link & analyze</button>
                            </div>
                            <div id="assemblyResults" class="hidden">
                                <div class="stat-grid">
                                    <div class="stat-card"><div class="stat-value" id="asm_up">-</div><div>Genomes</div></div>
                                    <div class="stat-card"><div class="stat-value" id="asm_meta">-</div><div>Metadata linked</div></div>
                                    <div class="stat-card"><div class="stat-value" id="asm_lin">-</div><div>Clusters</div></div>
                                    <div class="stat-card"><div class="stat-value" id="asm_esc">-</div><div>Escape candidates</div></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-cubes"></i> Analysis Modules</h2>
                        <p class="card-sub">Select from genomic processing, evolutionary modeling, and predictive AI.</p>
                        <div id="analysisGroups" style="display:grid; grid-template-columns:repeat(3,1fr); gap:24px;"></div>
                        <div style="margin-top:24px;">
                            <label><input type="checkbox" checked> Enable quantum‑inspired ML</label>
                        </div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-chart-line"></i> Parameter Tuning</h2>
                        <p class="card-sub">Adjust evolutionary parameters to see predicted impact.</p>
                        <div class="slider-container">
                            <div class="slider-label"><span>Mutation rate</span> <span id="mutVal">0.023</span></div>
                            <input type="range" id="mutRate" min="0.001" max="0.1" step="0.001" value="0.023" oninput="window.updatePrediction()">
                        </div>
                        <div class="slider-container">
                            <div class="slider-label"><span>Recombination rate</span> <span id="recVal">0.017</span></div>
                            <input type="range" id="recRate" min="0.001" max="0.05" step="0.001" value="0.017" oninput="window.updatePrediction()">
                        </div>
                        <div class="stat-grid" style="margin-top:20px;">
                            <div class="stat-card"><div class="stat-value" id="predEscape">0.63</div><div>Escape score</div></div>
                            <div class="stat-card"><div id="predRisk"><span class="pill high">High</span></div><div>Risk</div></div>
                            <div class="stat-card"><div id="predMuts"><span class="escape-badge">N32S</span><span class="escape-badge">E89K</span></div><div>Predicted mutations</div></div>
                        </div>
                    </div>
                </div>

                <div class="stack">
                    <div class="side-card"><h3 class="side-title">Active Jobs</h3><div id="jobsBoxRun"></div></div>
                    <div class="side-card"><h3 class="side-title">Live Intelligence</h3><div id="liveUpdatesBoxRun"></div></div>
                    <div class="side-card"><h3 class="side-title">Surveillance Map</h3><div id="mapContainerRun" class="map-container"></div></div>
                </div>
            </div>
            `;
        }

        async function initRun() {
            await loadJobs('jobsBoxRun');
            await loadLiveUpdates('liveUpdatesBoxRun');
            await loadAnalysisGroups();
            initMap('mapContainerRun', true);
            window.updatePrediction();
        }

        window.selectMode = function(mode) {
            document.getElementById('rawChoice')?.classList.toggle('active', mode === 'raw');
            document.getElementById('assemblyChoice')?.classList.toggle('active', mode === 'assembly');
            document.getElementById('rawPanel')?.classList.toggle('hidden', mode !== 'raw');
            document.getElementById('assemblyPanel')?.classList.toggle('hidden', mode !== 'assembly');
        };

        window.runRawPipeline = async function() {
            const data = await getJSON('/api/raw-results');
            document.getElementById('raw_qc').textContent = data.qc_pass_rate;
            document.getElementById('raw_asm').textContent = data.assembled_genomes;
            document.getElementById('raw_cov').textContent = data.mean_coverage;
            document.getElementById('raw_rec').textContent = data.recombination_events;
            document.getElementById('raw_hr').textContent = data.high_risk_variants;
            document.getElementById('rawResults').classList.remove('hidden');
        };

        window.runAssemblyFlow = async function() {
            const data = await getJSON('/api/assembly-results');
            document.getElementById('asm_up').textContent = data.uploaded_genomes;
            document.getElementById('asm_meta').textContent = data.metadata_linked;
            document.getElementById('asm_lin').textContent = data.lineage_clusters;
            document.getElementById('asm_esc').textContent = data.escape_candidates;
            document.getElementById('assemblyResults').classList.remove('hidden');
        };

        window.updatePrediction = async function() {
            const mutRate = parseFloat(document.getElementById('mutRate').value);
            const recRate = parseFloat(document.getElementById('recRate').value);
            document.getElementById('mutVal').textContent = mutRate.toFixed(3);
            document.getElementById('recVal').textContent = recRate.toFixed(3);
            const res = await getJSON(`/api/prediction-update?mutation_rate=${mutRate}&recombination_rate=${recRate}`);
            document.getElementById('predEscape').textContent = res.escape_score;
            document.getElementById('predRisk').innerHTML = `<span class="pill ${pillClass(res.risk)}">${res.risk}</span>`;
            document.getElementById('predMuts').innerHTML = res.predicted_mutations.map(m => `<span class="escape-badge">${m}</span>`).join(' ');
        };

        async function loadAnalysisGroups() {
            const groups = await getJSON('/api/analysis-modules');
            const container = document.getElementById('analysisGroups');
            container.innerHTML = '';
            groups.forEach(g => {
                const card = document.createElement('div');
                card.style.background = 'var(--bg-card)';
                card.style.borderRadius = '24px';
                card.style.padding = '20px';
                card.innerHTML = `<h4 style="margin-bottom:16px; color:var(--accent-cyan);">${g.group}</h4>` +
                    g.items.map((item, idx) => `
                        <label style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                            <input type="checkbox" ${idx<2?'checked':''}> <span>${item}</span>
                        </label>
                    `).join('');
                container.appendChild(card);
            });
        }

        // ---------- RESULTS PAGE (now with tabs and multiple plots) ----------
        async function getResultsHTML() {
            return `
            <div class="hero">
                <div class="eyebrow">Predictive Outputs</div>
                <h1>Analysis results & risk assessment</h1>
                <p>Immune escape scores, mutation loads, and evolutionary risk. Explore interactive plots.</p>
            </div>

            <div class="tabs">
                <div class="tab active" onclick="switchResultsTab('samples')">Samples</div>
                <div class="tab" onclick="switchResultsTab('beast')">BEAST Output</div>
                <div class="tab" onclick="switchResultsTab('forecast')">Forecast</div>
                <div class="tab" onclick="switchResultsTab('recomb')">Recombination</div>
                <div class="tab" onclick="switchResultsTab('corr')">Mutation Correlation</div>
            </div>

            <div id="resultsTabSamples" class="tab-content">
                <div class="grid">
                    <div class="stack">
                        <div class="card">
                            <h2 class="card-title"><i class="fa-solid fa-table"></i> Sample‑level predictions</h2>
                            <div class="search-box">
                                <i class="fa-solid fa-search"></i>
                                <input type="text" id="resultsSearch" placeholder="Search samples..." oninput="filterResultsTable()">
                            </div>
                            <div class="table-wrap">
                                <table>
                                    <thead><tr><th>Sample</th><th>Lineage</th><th>Mutations</th><th>Recomb.</th><th>Escape score</th><th>Immune escape</th><th>Risk</th></tr></thead>
                                    <tbody id="resultsTableBody"></tbody>
                                </table>
                            </div>
                            <div class="pagination" id="resultsPagination"></div>
                        </div>
                        <div class="chart-row">
                            <div class="chart-box"><h3>Risk distribution</h3><canvas id="lineageChartResults"></canvas></div>
                            <div class="chart-box"><h3>Escape score trend</h3><canvas id="casesChartResults"></canvas></div>
                        </div>
                    </div>
                    <div class="stack">
                        <div class="side-card"><h3 class="side-title">Active Jobs</h3><div id="jobsBoxResults"></div></div>
                        <div class="side-card"><h3 class="side-title">Live Intelligence</h3><div id="liveUpdatesBoxResults"></div></div>
                        <div class="side-card"><h3 class="side-title">Surveillance Map</h3><div id="mapContainerResults" class="map-container"></div></div>
                    </div>
                </div>
            </div>

            <div id="resultsTabBeast" class="tab-content hidden">
                <div class="card">
                    <h2 class="card-title"><i class="fa-solid fa-code-branch"></i> BEAST Molecular Clock</h2>
                    <p class="card-sub">Divergence times and posterior distributions.</p>
                    <div id="beastTree" style="height:500px;"></div>
                    <div class="chart-row">
                        <div class="chart-box"><h3>Posterior probabilities</h3><div id="beastPosterior" style="height:300px;"></div></div>
                        <div class="chart-box"><h3>ESS values</h3><div id="beastESS" style="height:300px;"></div></div>
                    </div>
                </div>
            </div>

            <div id="resultsTabForecast" class="tab-content hidden">
                <div class="card">
                    <h2 class="card-title"><i class="fa-solid fa-chart-line"></i> 14‑day Escape Score Forecast</h2>
                    <p class="card-sub">With 95% confidence intervals.</p>
                    <div id="forecastPlot" style="height:400px;"></div>
                </div>
            </div>

            <div id="resultsTabRecomb" class="tab-content hidden">
                <div class="card">
                    <h2 class="card-title"><i class="fa-solid fa-dna"></i> Recombination Hotspots</h2>
                    <p class="card-sub">Probability of recombination along the genome.</p>
                    <div id="recombPlot" style="height:400px;"></div>
                </div>
            </div>

            <div id="resultsTabCorr" class="tab-content hidden">
                <div class="card">
                    <h2 class="card-title"><i class="fa-solid fa-border-all"></i> Mutation Correlation Matrix</h2>
                    <p class="card-sub">Co‑occurrence of top mutations.</p>
                    <div id="corrHeatmap" style="height:500px;"></div>
                </div>
            </div>

            <div id="sampleModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; align-items:center; justify-content:center;" onclick="this.style.display='none'">
                <div style="background:var(--bg-card); border-radius:36px; padding:32px; max-width:500px; margin:auto;" onclick="event.stopPropagation()">
                    <h2 id="modalTitle"></h2>
                    <p><strong>Lineage:</strong> <span id="modalLineage"></span></p>
                    <p><strong>Mutations:</strong> <span id="modalMuts"></span></p>
                    <p><strong>Risk:</strong> <span id="modalRisk"></span></p>
                    <p><strong>Immune escape:</strong> <span id="modalImmune"></span></p>
                    <button onclick="document.getElementById('sampleModal').style.display='none'" style="margin-top:20px; padding:10px 20px; background:var(--accent-cyan); border:none; border-radius:20px; cursor:pointer;">Close</button>
                </div>
            </div>
            `;
        }

        let resultsData = [];
        async function initResults() {
            resultsData = await getJSON('/api/results');
            await loadJobs('jobsBoxResults');
            await loadLiveUpdates('liveUpdatesBoxResults');
            await loadResultsTable();
            initResultsCharts();
            initMap('mapContainerResults', true);

            // load additional tabs data
            await loadBeastOutput();
            await loadForecast();
            await loadRecomb();
            await loadCorrelation();
        }

        function switchResultsTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            document.querySelector(`.tab[onclick*="${tab}"]`).classList.add('active');
            document.getElementById(`resultsTab${tab.charAt(0).toUpperCase()+tab.slice(1)}`).classList.remove('hidden');
        }

        async function loadResultsTable(page = 1, filter = '') {
            const rows = resultsData.filter(r => r.sample.toLowerCase().includes(filter.toLowerCase()) || r.lineage.toLowerCase().includes(filter.toLowerCase()));
            const pageSize = 10;
            const start = (page - 1) * pageSize;
            const paginated = rows.slice(start, start + pageSize);
            const tbody = document.getElementById('resultsTableBody');
            tbody.innerHTML = '';
            paginated.forEach(r => {
                const tr = document.createElement('tr');
                tr.style.cursor = 'pointer';
                tr.onclick = () => showSampleDetails(r);
                tr.innerHTML = `
                    <td>${r.sample}</td>
                    <td>${r.lineage}</td>
                    <td>${r.mutations}</td>
                    <td>${r.recomb}</td>
                    <td>${r.escape_score}</td>
                    <td><span class="pill ${r.immune_escape === 'Confirmed' ? 'critical' : (r.immune_escape === 'Potential' ? 'high' : 'monitor')}">${r.immune_escape}</span></td>
                    <td><span class="pill ${pillClass(r.risk)}">${r.risk}</span></td>
                `;
                tbody.appendChild(tr);
            });
            const totalPages = Math.ceil(rows.length / pageSize);
            let paginationHtml = '';
            for (let i = 1; i <= totalPages; i++) {
                paginationHtml += `<span class="page-btn ${i === page ? 'active' : ''}" onclick="loadResultsTable(${i}, document.getElementById('resultsSearch').value)">${i}</span>`;
            }
            document.getElementById('resultsPagination').innerHTML = paginationHtml;
        }

        window.filterResultsTable = function() {
            loadResultsTable(1, document.getElementById('resultsSearch').value);
        };

        function showSampleDetails(sample) {
            document.getElementById('modalTitle').textContent = sample.sample;
            document.getElementById('modalLineage').textContent = sample.lineage;
            document.getElementById('modalMuts').textContent = (sample.mutations_list || []).join(', ');
            document.getElementById('modalRisk').innerHTML = `<span class="pill ${pillClass(sample.risk)}">${sample.risk}</span>`;
            document.getElementById('modalImmune').textContent = sample.immune_escape;
            document.getElementById('sampleModal').style.display = 'flex';
        }

        function initResultsCharts() {
            const ctx = document.getElementById('lineageChartResults')?.getContext('2d');
            if (ctx) {
                new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['High', 'Critical', 'Moderate', 'Monitor'],
                        datasets: [{
                            data: [5, 3, 4, 3],
                            backgroundColor: ['#f87171', '#dc2626', '#3b82f6', '#4ade80'],
                            borderWidth: 0
                        }]
                    },
                    options: { responsive: true, plugins: { legend: { position: 'bottom' } }, animation: { duration: 1000 } }
                });
            }
            const ctx2 = document.getElementById('casesChartResults')?.getContext('2d');
            if (ctx2) {
                new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: ['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5', 'Sample6', 'Sample7'],
                        datasets: [{
                            label: 'Escape Score',
                            data: [0.88, 0.93, 0.46, 0.74, 0.31, 0.89, 0.94],
                            borderColor: '#3b82f6',
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: { responsive: true, animation: { duration: 1000 } }
                });
            }
        }

        async function loadBeastOutput() {
            const data = await getJSON('/api/beast-output');
            // Simple tree visualization using Plotly (sunburst or scatter)
            // For simplicity, we'll create a sunburst from the tree
            const tree = data.tree;
            const sunburstData = [{
                type: "sunburst",
                labels: [],
                parents: [],
                values: [],
                branchvalues: 'total'
            }];
            function traverse(node, parent) {
                sunburstData[0].labels.push(node.name);
                sunburstData[0].parents.push(parent);
                sunburstData[0].values.push(node.value || 1);
                if (node.children) {
                    node.children.forEach(child => traverse(child, node.name));
                }
            }
            traverse(tree, "");
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                margin: { l: 0, r: 0, t: 30, b: 0 }
            };
            Plotly.newPlot('beastTree', sunburstData, layout, { responsive: true });

            // Posterior distribution (dummy histogram)
            const postTrace = {
                x: data.posterior,
                type: 'histogram',
                marker: { color: 'var(--accent-cyan)' }
            };
            const postLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { title: 'Posterior probability' },
                yaxis: { title: 'Frequency' }
            };
            Plotly.newPlot('beastPosterior', [postTrace], postLayout, { responsive: true });

            // ESS bar chart
            const essTrace = {
                x: ['mu', 'sigma', 'treeHeight', 'popSize', 'clockRate'],
                y: data.ess_values,
                type: 'bar',
                marker: { color: 'var(--accent-green)' }
            };
            const essLayout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { title: 'Parameter' },
                yaxis: { title: 'ESS' }
            };
            Plotly.newPlot('beastESS', [essTrace], essLayout, { responsive: true });
        }

        async function loadForecast() {
            const data = await getJSON('/api/forecast');
            const trace1 = {
                x: data.dates,
                y: data.mean,
                mode: 'lines+markers',
                name: 'Mean',
                line: { color: 'var(--accent-cyan)' }
            };
            const trace2 = {
                x: data.dates,
                y: data.lower_ci,
                mode: 'lines',
                name: 'Lower 95% CI',
                line: { dash: 'dash', color: 'rgba(15,240,252,0.5)' }
            };
            const trace3 = {
                x: data.dates,
                y: data.upper_ci,
                mode: 'lines',
                name: 'Upper 95% CI',
                line: { dash: 'dash', color: 'rgba(15,240,252,0.5)' },
                fill: 'tonexty',
                fillcolor: 'rgba(15,240,252,0.2)'
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { title: 'Date' },
                yaxis: { title: 'Escape score' }
            };
            Plotly.newPlot('forecastPlot', [trace2, trace3, trace1], layout, { responsive: true });
        }

        async function loadRecomb() {
            const data = await getJSON('/api/recombination-hotspots');
            const trace = {
                x: data.positions,
                y: data.probability,
                mode: 'lines+markers',
                type: 'scatter',
                marker: { color: 'var(--accent-purple)' },
                line: { color: 'var(--accent-purple)' }
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { title: 'Genome position (bp)' },
                yaxis: { title: 'Recombination probability' }
            };
            Plotly.newPlot('recombPlot', [trace], layout, { responsive: true });
        }

        async function loadCorrelation() {
            const data = await getJSON('/api/mutation-correlation');
            const trace = {
                z: data.matrix,
                x: data.mutations,
                y: data.mutations,
                type: 'heatmap',
                colorscale: 'Viridis',
                reversescale: true
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { tickangle: -45 },
                yaxis: { tickangle: -45 }
            };
            Plotly.newPlot('corrHeatmap', [trace], layout, { responsive: true });
        }

        // ---------- INTELLIGENCE PAGE (enhanced with more plots) ----------
        async function getIntelligenceHTML() {
            return `
            <div class="hero">
                <div class="eyebrow">Evolutionary Intelligence</div>
                <h1>Manifold · Networks · Fitness Landscape</h1>
                <p>Topology‑aware manifold learning, epistatic networks, and dynamic fitness terrain with strain labels. Click any strain to locate on map.</p>
            </div>

            <div class="grid" style="grid-template-columns:1fr 1fr;">
                <div class="stack">
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-share-alt"></i> Evolutionary Manifold</h2>
                        <p class="card-sub">Diffusion map embedding of 124 PRRSV genomes. Hover for details.</p>
                        <div id="manifoldPlot" class="manifold-plot"></div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-project-diagram"></i> Epistatic Mutation Network</h2>
                        <p class="card-sub">Statistically significant co‑mutation edges (force‑directed).</p>
                        <div class="network-container" id="networkContainer"></div>
                        <div class="flex-row" style="justify-content:space-between; margin-top:16px;">
                            <span>Nodes: <span id="networkNodes">-</span></span>
                            <span>Edges: <span id="networkEdges">-</span></span>
                        </div>
                        <div id="networkStats" style="margin-top:12px; font-size:14px; color:var(--accent-cyan);"></div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-chart-simple"></i> Mutation Load vs Escape Score</h2>
                        <p class="card-sub">Each point is a sample; size = mutation count. Click to highlight on map.</p>
                        <div id="mutationScatter" style="height:350px;"></div>
                    </div>
                </div>

                <div class="stack">
                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-shield-virus"></i> Immune Escape Predictions</h2>
                        <p class="card-sub">Mutations flagged by quantum‑inspired SVM.</p>
                        <div id="escapeList" style="margin-bottom:20px;"></div>
                        <div class="stat-card">
                            <div class="stat-value" id="escapeCount">3</div>
                            <div>high‑confidence escape variants</div>
                        </div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-mountain"></i> Fitness Landscape</h2>
                        <p class="card-sub">Mountain‑like terrain where height = immune escape score. Click spheres to locate farm.</p>
                        <div id="landscapeContainer" class="landscape-container"></div>
                    </div>

                    <div class="card">
                        <h2 class="card-title"><i class="fa-solid fa-chart-bar"></i> Mutation Frequency</h2>
                        <p class="card-sub">Occurrence of top escape mutations across all samples.</p>
                        <div id="mutationBarChart" style="height:300px;"></div>
                    </div>

                    <div class="side-card">
                        <h3 class="side-title"><i class="fa-solid fa-rss"></i> Live Intelligence Feed</h3>
                        <div id="liveUpdatesBoxIntel"></div>
                    </div>
                </div>
            </div>
            `;
        }

        async function initIntelligence() {
            await loadLiveUpdates('liveUpdatesBoxIntel');
            await loadManifold();
            await loadNetwork();
            await loadEscape();
            await initFitnessLandscape();
            await loadMutationScatter();
            await loadMutationBarChart();
        }

        // Global map references (same as before, but enhanced with clustering)
        let leafletMaps = {};
        let mapMarkers = {};
        let heatLayer = null;

        function initMap(containerId, withHeat = false) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (leafletMaps[containerId]) leafletMaps[containerId].remove();

            const map = L.map(container).setView([39.5, -95.0], 4);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap',
                className: 'leaflet-tiles'
            }).addTo(map);

            // Add marker clustering
            const markers = L.markerClusterGroup({
                showCoverageOnHover: false,
                maxClusterRadius: 50,
                iconCreateFunction: function(cluster) {
                    const count = cluster.getChildCount();
                    let color = '#3b82f6';
                    if (count > 20) color = '#dc2626';
                    else if (count > 10) color = '#f87171';
                    return L.divIcon({ html: '<div style="background:'+color+'; width:30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">'+count+'</div>', className: '', iconSize: [30,30] });
                }
            });

            fetch('/api/current-situation')
                .then(res => res.json())
                .then(farms => {
                    farms.forEach(farm => {
                        const color = farm.status === 'Critical' ? '#dc2626' :
                                     farm.status === 'High' ? '#f87171' :
                                     farm.status === 'Moderate' ? '#3b82f6' : '#4ade80';
                        const marker = L.circleMarker([farm.lat, farm.lng], {
                            radius: 8 + farm.cases/2,
                            fillColor: color,
                            color: '#fff',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.8
                        });
                        marker.bindPopup(`
                            <b>${farm.site}</b><br>
                            Cases: ${farm.cases}<br>
                            Status: ${farm.status}<br>
                            Risk Score: ${farm.risk_score}<br>
                            Trend: ${farm.trend} ${farm.trend==='rising'?'📈':farm.trend==='falling'?'📉':'➡️'}<br>
                            Last update: ${farm.last_update}<br>
                            Lineages: ${farm.lineages.join(', ')}
                        `);
                        marker.on('click', () => {
                            map.flyTo([farm.lat, farm.lng], 8);
                        });
                        mapMarkers[farm.site] = marker;
                        markers.addLayer(marker);
                    });
                    map.addLayer(markers);

                    if (withHeat) {
                        const heatPoints = farms.map(f => [f.lat, f.lng, f.cases/30]);
                        heatLayer = L.heatLayer(heatPoints, { radius: 25, blur: 15, maxZoom: 10 }).addTo(map);
                    }
                });

            leafletMaps[containerId] = map;
        }

        // Real-time update (unchanged)
        setInterval(() => {
            const farms = Object.values(mapMarkers);
            if (farms.length === 0) return;
            const randomFarm = farms[Math.floor(Math.random() * farms.length)];
            const newCases = randomFarm.getPopup().getContent().match(/Cases: (\\d+)/);
            if (newCases) {
                let cases = parseInt(newCases[1]);
                cases += Math.floor(Math.random() * 5) - 2;
                cases = Math.max(0, cases);
                const oldContent = randomFarm.getPopup().getContent();
                const newContent = oldContent.replace(/Cases: \\d+/, `Cases: ${cases}`);
                randomFarm.getPopup().setContent(newContent);
                randomFarm.setRadius(8 + cases/2);
                let status = 'Monitor';
                if (cases > 25) status = 'Critical';
                else if (cases > 15) status = 'High';
                else if (cases > 7) status = 'Moderate';
                const color = status === 'Critical' ? '#dc2626' :
                             status === 'High' ? '#f87171' :
                             status === 'Moderate' ? '#3b82f6' : '#4ade80';
                randomFarm.setStyle({ fillColor: color });
            }
        }, 8000);

        window.highlightFarm = function(stateName) {
            const stateCoords = {
                'MN': [45.0, -93.0], 'IA': [41.5, -93.5], 'IL': [40.0, -89.0],
                'NE': [41.5, -99.0], 'NC': [35.5, -79.0], 'OH': [40.3, -82.8],
                'TX': [31.0, -97.0], 'CA': [36.8, -119.8], 'FL': [27.7, -81.5],
                'PA': [40.9, -77.8]
            };
            if (stateCoords[stateName]) {
                const map = leafletMaps['dashboardMap'] || leafletMaps['mapContainerRun'] || leafletMaps['mapContainerResults'];
                if (map) {
                    map.flyTo(stateCoords[stateName], 7);
                }
            }
        };

        // Enhanced 3D landscape with more pronounced peaks
        async function initFitnessLandscape() {
            const container = document.getElementById('landscapeContainer');
            if (!container) return;

            const manifoldPoints = await getJSON('/api/manifold-points');
            const results = await getJSON('/api/results');

            const scoreMap = {};
            results.forEach(r => { scoreMap[r.sample] = r.escape_score; });

            const points = manifoldPoints.map(p => ({
                x: (p.x - 0.5) * 8,
                z: (p.y - 0.5) * 8,
                y: (scoreMap[p.sample] || 0.5) * 4 - 2,  // increased vertical range
                sample: p.sample,
                lineage: p.lineage,
                escape: scoreMap[p.sample] || 0.5
            }));

            const width = container.clientWidth;
            const height = 400;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1e2a3a);

            const camera = new THREE.PerspectiveCamera(45, width/height, 0.1, 1000);
            camera.position.set(12, 8, 12);
            camera.lookAt(0, 0, 0);

            const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
            renderer.setSize(width, height);
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.innerHTML = '';
            container.appendChild(renderer.domElement);

            const ambientLight = new THREE.AmbientLight(0x404060);
            scene.add(ambientLight);
            const dirLight = new THREE.DirectionalLight(0xffffff, 1);
            dirLight.position.set(5, 10, 7);
            dirLight.castShadow = true;
            dirLight.shadow.mapSize.width = 1024;
            dirLight.shadow.mapSize.height = 1024;
            const d = 15;
            dirLight.shadow.camera.left = -d;
            dirLight.shadow.camera.right = d;
            dirLight.shadow.camera.top = d;
            dirLight.shadow.camera.bottom = -d;
            dirLight.shadow.camera.near = 1;
            dirLight.shadow.camera.far = 25;
            scene.add(dirLight);
            const backLight = new THREE.PointLight(0x4466aa, 0.5);
            backLight.position.set(-5, 0, -5);
            scene.add(backLight);

            // Generate more mountainous terrain using interpolation + random peaks
            const gridSize = 40;
            const xs = points.map(p => p.x);
            const zs = points.map(p => p.z);
            const minX = Math.min(...xs) - 1;
            const maxX = Math.max(...xs) + 1;
            const minZ = Math.min(...zs) - 1;
            const maxZ = Math.max(...zs) + 1;

            // Create grid with synthetic peaks for dramatic effect
            const grid = [];
            for (let i = 0; i <= gridSize; i++) {
                const x = minX + (maxX - minX) * i / gridSize;
                grid[i] = [];
                for (let j = 0; j <= gridSize; j++) {
                    const z = minZ + (maxZ - minZ) * j / gridSize;
                    let totalWeight = 0;
                    let weightedHeight = 0;
                    points.forEach(p => {
                        const dx = x - p.x;
                        const dz = z - p.z;
                        const distSq = dx*dx + dz*dz;
                        if (distSq < 0.01) {
                            weightedHeight = p.y;
                            totalWeight = 1;
                            return;
                        }
                        const w = 1 / (distSq + 0.1);
                        weightedHeight += w * p.y;
                        totalWeight += w;
                    });
                    let h = (totalWeight > 0) ? weightedHeight / totalWeight : 0;
                    // Add artificial mountain peaks using sine/cosine
                    h += 0.8 * Math.sin(x * 0.8) * Math.cos(z * 0.8);
                    h += 0.5 * Math.sin(x * 1.5) * Math.cos(z * 1.5);
                    grid[i][j] = h;
                }
            }

            const vertices = [];
            const indices = [];
            for (let i = 0; i <= gridSize; i++) {
                for (let j = 0; j <= gridSize; j++) {
                    const x = minX + (maxX - minX) * i / gridSize;
                    const z = minZ + (maxZ - minZ) * j / gridSize;
                    const y = grid[i][j];
                    vertices.push(x, y, z);
                }
            }
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const a = i * (gridSize+1) + j;
                    const b = i * (gridSize+1) + j + 1;
                    const c = (i+1) * (gridSize+1) + j;
                    const dIdx = (i+1) * (gridSize+1) + j + 1;
                    indices.push(a, b, c);
                    indices.push(b, dIdx, c);
                }
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setIndex(indices);
            geometry.computeVertexNormals();

            const posAttr = geometry.attributes.position;
            const colorArray = [];
            for (let i = 0; i < posAttr.count; i++) {
                const y = posAttr.getY(i);
                const t = (y + 2) / 4.0; // normalize to 0-1
                let r, g, b;
                if (t < 0.33) {
                    r = 0;
                    g = t * 3;
                    b = 1;
                } else if (t < 0.66) {
                    r = (t - 0.33) * 3;
                    g = 1;
                    b = 1 - (t - 0.33) * 3;
                } else {
                    r = 1;
                    g = 1 - (t - 0.66) * 3;
                    b = 0;
                }
                colorArray.push(r, g, b);
            }
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));

            const material = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 30, side: THREE.DoubleSide, emissive: 0x000000 });
            const terrain = new THREE.Mesh(geometry, material);
            terrain.castShadow = true;
            terrain.receiveShadow = true;
            scene.add(terrain);

            points.forEach(p => {
                const sphereGeo = new THREE.SphereGeometry(0.25, 32, 16);
                let color;
                if (p.lineage === 'L1') color = 0x3b82f6;
                else if (p.lineage === 'L5') color = 0xa78bfa;
                else if (p.lineage === 'L8') color = 0xfbbf24;
                else if (p.lineage === 'L3') color = 0x4ade80;
                else if (p.lineage === 'L2') color = 0xf87171;
                else color = 0xffaa00;
                const sphereMat = new THREE.MeshStandardMaterial({ color: color, emissive: 0x222222 });
                const sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.set(p.x, p.y + 0.15, p.z);
                sphere.castShadow = true;
                sphere.receiveShadow = true;
                sphere.userData = { sample: p.sample, state: p.sample.split('_')[1] };
                sphere.callback = () => highlightFarm(sphere.userData.state);
                sphere.onClick = function() { this.callback(); };
                scene.add(sphere);
            });

            const gridHelper = new THREE.GridHelper(16, 20, 0x3b82f6, 0x2a3f5e);
            gridHelper.position.y = -2.1;
            scene.add(gridHelper);

            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            renderer.domElement.addEventListener('click', (event) => {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(scene.children.filter(obj => obj.isMesh && obj.geometry.type === 'SphereGeometry'));
                if (intersects.length > 0) {
                    const hit = intersects[0].object;
                    if (hit.onClick) hit.onClick();
                }
            });

            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }
            animate();

            window.addEventListener('resize', () => {
                const newWidth = container.clientWidth;
                const newHeight = 400;
                camera.aspect = newWidth / newHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);
            });
        }

        async function loadManifold() {
            const points = await getJSON('/api/manifold-points');
            const trace = {
                x: points.map(p => p.x),
                y: points.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                text: points.map(p => `${p.sample} (${p.lineage})`),
                marker: { 
                    size: 12,
                    color: points.map(p => {
                        if (p.lineage === 'L1') return '#3b82f6';
                        if (p.lineage === 'L5') return '#a78bfa';
                        if (p.lineage === 'L8') return '#fbbf24';
                        return '#4ade80';
                    }),
                    line: { color: 'white', width: 1 }
                }
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { gridcolor: 'var(--border-glow)' },
                yaxis: { gridcolor: 'var(--border-glow)' },
                margin: { l: 40, r: 40, t: 40, b: 40 }
            };
            Plotly.newPlot('manifoldPlot', [trace], layout, { responsive: true });
        }

        async function loadNetwork() {
            const edges = await getJSON('/api/epistatic-network-edges');
            const nodesMap = new Map();
            edges.forEach(e => {
                if (!nodesMap.has(e.source)) nodesMap.set(e.source, {id: e.source});
                if (!nodesMap.has(e.target)) nodesMap.set(e.target, {id: e.target});
            });
            const nodes = Array.from(nodesMap.values());
            const width = document.getElementById('networkContainer').clientWidth;
            const height = 400;

            const svg = d3.select("#networkContainer").html("").append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("class", "network-svg")
                .call(d3.zoom().on("zoom", (event) => svg.attr("transform", event.transform)));

            const container = svg.append("g");

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(edges).id(d => d.id).distance(80))
                .force("charge", d3.forceManyBody().strength(-150))
                .force("center", d3.forceCenter(width/2, height/2));

            const link = container.append("g")
                .selectAll("line")
                .data(edges)
                .enter().append("line")
                .attr("stroke", "var(--border-glow)")
                .attr("stroke-width", d => d.weight * 2);

            const node = container.append("g")
                .selectAll("circle")
                .data(nodes)
                .enter().append("circle")
                .attr("r", 12)
                .attr("fill", "var(--accent-cyan)")
                .attr("stroke", "var(--bg-deep)")
                .attr("stroke-width", 2)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            node.append("title").text(d => d.id);

            simulation.on("tick", () => {
                link.attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);
                node.attr("cx", d => d.x)
                    .attr("cy", d => d.y);
            });

            function dragstarted(event) { if (!event.active) simulation.alphaTarget(0.3).restart(); event.subject.fx = event.subject.x; event.subject.fy = event.subject.y; }
            function dragged(event) { event.subject.fx = event.x; event.subject.fy = event.y; }
            function dragended(event) { if (!event.active) simulation.alphaTarget(0); event.subject.fx = null; event.subject.fy = null; }

            document.getElementById('networkNodes').textContent = nodes.length;
            document.getElementById('networkEdges').textContent = edges.length;

            const degree = {};
            edges.forEach(e => {
                degree[e.source] = (degree[e.source] || 0) + 1;
                degree[e.target] = (degree[e.target] || 0) + 1;
            });
            const sorted = Object.entries(degree).sort((a,b) => b[1] - a[1]).slice(0,3);
            const statsDiv = document.getElementById('networkStats');
            statsDiv.innerHTML = `🔹 Top hub mutations: ${sorted.map(([m, d]) => `${m} (${d})`).join(', ')} – strong epistasis.`;
        }

        async function loadEscape() {
            const escapes = await getJSON('/api/predicted-escape');
            const escapeDiv = document.getElementById('escapeList');
            escapeDiv.innerHTML = escapes.map(m => `<span class="escape-badge">${m}</span>`).join(' ');
            document.getElementById('escapeCount').textContent = escapes.length;
        }

        async function loadMutationScatter() {
            const results = await getJSON('/api/results');
            const trace = {
                x: results.map(r => r.mutations),
                y: results.map(r => r.escape_score),
                text: results.map(r => r.sample),
                mode: 'markers',
                type: 'scatter',
                marker: {
                    size: results.map(r => 10 + r.mutations),
                    color: results.map(r => {
                        if (r.risk === 'Critical') return '#dc2626';
                        if (r.risk === 'High') return '#f87171';
                        if (r.risk === 'Moderate') return '#3b82f6';
                        return '#4ade80';
                    }),
                    line: { color: 'white', width: 1 }
                }
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { title: 'Mutation count', gridcolor: 'var(--border-glow)' },
                yaxis: { title: 'Escape score', gridcolor: 'var(--border-glow)' },
                margin: { l: 50, r: 30, t: 20, b: 40 }
            };
            Plotly.newPlot('mutationScatter', [trace], layout, { responsive: true });
        }

        async function loadMutationBarChart() {
            const escapes = await getJSON('/api/predicted-escape');
            // Use real frequencies from mutation trends if available
            const mutTrends = await getJSON('/api/mutation-trends');
            const freqs = mutTrends.frequencies;
            const trace = {
                x: escapes,
                y: escapes.map(m => freqs[m] ? freqs[m][freqs[m].length-1] : 0.5),
                type: 'bar',
                marker: { color: 'var(--accent-cyan)' }
            };
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: 'var(--text-primary)' },
                xaxis: { gridcolor: 'var(--border-glow)' },
                yaxis: { title: 'Frequency', gridcolor: 'var(--border-glow)' },
                margin: { l: 40, r: 20, t: 20, b: 50 }
            };
            Plotly.newPlot('mutationBarChart', [trace], layout, { responsive: true });
        }

        // ---------- COMMON COMPONENTS (unchanged) ----------
        async function loadJobs(targetId, limit = 5) {
            const box = document.getElementById(targetId);
            if (!box) return;
            const jobs = await getJSON('/api/jobs');
            box.innerHTML = '';
            jobs.slice(0, limit).forEach(job => {
                const div = document.createElement('div');
                div.className = 'job';
                div.onclick = () => alert(`Details for ${job.id}: ${job.model}`);
                div.innerHTML = `
                    <div class="job-header">
                        <span class="job-id">${job.id} · ${job.model}</span>
                        <div class="job-actions" onclick="event.stopPropagation()">
                            <i class="fa-solid fa-play" onclick="window.controlJob('start', '${job.id}')"></i>
                            <i class="fa-solid fa-pause" onclick="window.controlJob('pause', '${job.id}')"></i>
                            <i class="fa-solid fa-stop" onclick="window.controlJob('stop', '${job.id}')"></i>
                        </div>
                    </div>
                    <div class="job-name">${job.title}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:${job.progress}%"></div>
                    </div>
                    <div class="job-meta">
                        <span>Started: ${job.started}</span>
                        <span>ETA: ${job.eta}</span>
                    </div>
                `;
                box.appendChild(div);
            });
        }

        window.controlJob = function(action, jobId) {
            alert(`${action} job ${jobId} (simulated)`);
        };

        async function loadLiveUpdates(targetId, limit = 5) {
            const box = document.getElementById(targetId);
            if (!box) return;
            const items = await getJSON('/api/live-updates');
            box.innerHTML = '';
            items.slice(0, limit).forEach(item => {
                const div = document.createElement('div');
                div.className = 'feed-item';
                div.innerHTML = `
                    <div class="feed-top">
                        <span class="feed-time">${item.time}</span>
                        <span class="feed-tag">${item.tag}</span>
                    </div>
                    <div class="feed-text">${item.text}</div>
                `;
                box.appendChild(div);
            });
        }

        setInterval(async () => {
            const boxes = ['dashboardLive', 'liveUpdatesBoxRun', 'liveUpdatesBoxResults', 'liveUpdatesBoxIntel'];
            for (const id of boxes) {
                const box = document.getElementById(id);
                if (box) {
                    const newUpdate = {
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                        tag: ['AI','Alert','Sync','Run'][Math.floor(Math.random()*4)],
                        text: `[SIM] ${['Quantum kernel updated','New escape variant predicted','Manifold embedded','GNN training finished'][Math.floor(Math.random()*4)]}`
                    };
                    const div = document.createElement('div');
                    div.className = 'feed-item';
                    div.style.animation = 'slideIn 0.5s';
                    div.innerHTML = `
                        <div class="feed-top">
                            <span class="feed-time">${newUpdate.time}</span>
                            <span class="feed-tag">${newUpdate.tag}</span>
                        </div>
                        <div class="feed-text">${newUpdate.text}</div>
                    `;
                    box.prepend(div);
                    if (box.children.length > 6) box.removeChild(box.lastChild);
                }
            }
        }, 8000);
    </script>
</body>
</html>
    """)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)