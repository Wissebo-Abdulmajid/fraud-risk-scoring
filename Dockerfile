FROM python:3.11-slim

# Security + predictable behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . /app
RUN pip install -e .
COPY assets/demo_run /app/runs/demo

# Ensure folders exist (optional, but nice)
RUN mkdir -p /app/runs /app/reports

# make sure these exist
RUN mkdir -p /app/runs /app/reports

# copy demo run into container
COPY runs/demo_run /app/runs/demo_run

# Drop privileges
USER app

# Expose API port
EXPOSE 8000

# Production server
CMD ["uvicorn", "frs.api:app", "--host", "0.0.0.0", "--port", "8000"]
