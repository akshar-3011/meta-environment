FROM python:3.10-slim

WORKDIR /app

# Ensure we have git just in case
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set PYTHONPATH
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server natively
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
