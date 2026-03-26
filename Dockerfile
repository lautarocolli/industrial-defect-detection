FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install uv
RUN pip install uv

# Copy deployment-specific dependency files
COPY deployment/pyproject.toml deployment/uv.lock ./

# Install all deployment dependencies from deployment lockfile
# CPU-only torch is pinned in deployment/uv.lock
RUN uv sync

# Copy application code
COPY src ./src
COPY deployment ./deployment

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "deployment.app.main:app", "--host", "0.0.0.0", "--port", "8000"]