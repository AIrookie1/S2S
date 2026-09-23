#!/usr/bin/env python3
"""Compatibility wrapper for older scripts.

All plotting configuration now lives in plot_sizes.py, plot_theme.py and
plot_specs.py.  New code should import plot_common directly.
"""
from plot_common import *  # noqa: F401,F403
