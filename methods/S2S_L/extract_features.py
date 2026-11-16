"""S2S_L feature extraction reuses S2S_R's answer-level raw activations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.S2S_R.extract_features import main

if __name__ == "__main__":
    main()
