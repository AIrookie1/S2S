"""Compatibility entry point for the canonical S2S_R generic evaluator.

The implementation lives in ``method.S2S_R.generic.evaluate_generic``.
Keeping this tiny wrapper preserves the historical command
``python generic/evaluate_generic.py ...`` without maintaining a second,
diverging copy of the steering code.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from method.S2S_R.generic.evaluate_generic import main


if __name__ == "__main__":
    main()
