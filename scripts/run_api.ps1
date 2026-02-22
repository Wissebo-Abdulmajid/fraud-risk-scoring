param(
  [string]$HostAddr = "127.0.0.1",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Go to repo root
Set-Location (Split-Path -Parent $PSScriptRoot)

# Activate venv
.\.venv\Scripts\Activate.ps1

# Run API
python -m uvicorn frs.api:app --host $HostAddr --port $Port --reload
