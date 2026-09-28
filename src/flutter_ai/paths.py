"""Project directories (override with FLUTTER_AI_HOME / FLUTTER_AI_MODELS / FLUTTER_AI_OUTPUTS)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FLUTTER_AI_HOME", Path(__file__).resolve().parents[2]))
MODELS_DIR = Path(os.environ.get("FLUTTER_AI_MODELS", PROJECT_ROOT / "models"))
OUTPUTS_DIR = Path(os.environ.get("FLUTTER_AI_OUTPUTS", PROJECT_ROOT / "outputs"))
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
