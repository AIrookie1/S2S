"""Dyn2D reuses the same direct-vector MLP training objective as Dyn3D."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.dyn3d_sga.train_dyn_mlp import main

if __name__ == "__main__":
    main()
