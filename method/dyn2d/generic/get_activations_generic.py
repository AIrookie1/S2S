"""Dyn2D generic feature extraction reuses Dyn3D's implementation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.generic.get_activations_generic import main

if __name__ == "__main__":
    main()
