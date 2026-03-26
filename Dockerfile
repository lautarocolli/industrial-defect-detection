# Use the official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install uv
RUN pip install uv

# Copy dependency files 
COPY pyproject.toml uv.lock ./

# Install base + deploy dependencies only — skip dev group
RUN uv sync --group deploy --no-dev

# Copy application code
COPY src ./src
COPY deployment ./deployment

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "deployment.app.main:app", "--host", "0.0.0.0", "--port", "8000"]