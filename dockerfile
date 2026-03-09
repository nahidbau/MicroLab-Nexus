FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    wget \
    unzip \
    gzip \
    bzip2 \
    xz-utils \
    procps \
    build-essential \
    default-jre \
    fastqc \
    fastp \
    bwa \
    samtools \
    bcftools \
    bedtools \
    trimmomatic \
    iqtree \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Optional: Miniforge for bioconda tools
RUN mkdir -p /opt/conda && \
    curl -L -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda install -y -c conda-forge -c bioconda \
    multiqc \
    spades \
    prokka \
    mlst \
    abricate \
    seqkit \
    canu \
    && conda clean -afy

COPY app.py /app/app.py

RUN mkdir -p /app/data/uploads /app/data/results /app/data/logs /app/data/workspace

EXPOSE 10000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]