"""Dyn2D feature extraction reuses Dyn3D's answer-level raw activations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.extract_features import main

if __name__ == "__main__":
    main()
