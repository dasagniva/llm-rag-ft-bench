# ragbench API — CPU inference image
#
# GPU path (vLLM): documented in finetune/README.md; not a hard dependency here.
# CPU path is slow (~60 s/question on 8B) but works without CUDA drivers.
#
# Build:  docker build -t ragbench-api .
# Run:    docker compose up -d

FROM python:3.12-slim

# Install uv
RUN pip install --no-cache-dir uv==0.7.13

WORKDIR /app

# Copy dependency manifest first for layer-cache efficiency
COPY pyproject.toml uv.lock* ./

# CPU-only PyTorch wheels — overrides the cu126 index declared in pyproject.toml.
# This keeps the image to ~3 GB instead of ~8 GB for the CUDA wheels.
RUN uv sync --no-dev \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

# Application source and configs
COPY src/ src/
COPY configs/ configs/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "ragbench.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
