"""Reuse the canonical generic feature extractor for L2S-LINEAR.

The generic feature representation (raw answer-level activations, binary
labels, and question indices) is method-agnostic. L2S-LINEAR consumes the
same files as S2S_R and reconstructs raw positive/negative pairs in its
existing loader.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.S2S_R.generic.get_activations_generic import main


if __name__ == "__main__":
    main()
