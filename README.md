# PRRSV Predictive Evolution Platform

## Overview
The **PRRSV Predictive Evolution Platform** is an interactive web-based analytical system designed to integrate **genomic**, **epidemiological**, and **geospatial** data for understanding and predicting the evolution and spread of Porcine Reproductive and Respiratory Syndrome Virus (PRRSV).

This platform was developed as part of a PhD-level research framework focusing on multidimensional viral evolution and early warning prediction.

---

## Key Features

- Genomic diversity analysis
- Epistatic interaction network
- Evolutionary trajectory modeling
- Vaccine escape prediction
- Spatiotemporal diffusion analysis
- Geospatial clustering (DBSCAN)
- Temporal trend detection
- Interactive visualization dashboards
- Automated report generation

---

## Technology Stack

- **Backend:** FastAPI (Python)
- **Data Analysis:** NumPy, Pandas, SciPy, Scikit-learn
- **Bioinformatics:** BioPython
- **Visualization:** Plotly
- **Network Analysis:** NetworkX
- **Deployment:** Render (Cloud Hosting)

---

## Input Requirements

The platform accepts:

### 1. Genomic Data
- FASTA formatted aligned sequences

### 2. Epidemiological Data (CSV)
Required columns:

- `sample_id`
- `farm_id`
- `date`
- `latitude`
- `longitude`
- optional metadata fields

---

## Installation (Local Use)

Clone repository:

```bash
git clone https://github.com/Nahid141/prrsv-platform.git