"""Run Alembic migrations with .env file loaded."""
import os
import subprocess
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

import sys

# Run alembic
result = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=Path(__file__).parent)
exit(result.returncode)
