#!/usr/bin/env python3
"""Convenience entry point for coverage-driven case generation.

Usage:
    scripts/gen_cases_from_stats.py STATS.json [--top N]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "x86_diff", "from_stats.py")

if __name__ == "__main__":
    os.execv(sys.executable, [sys.executable, TARGET] + sys.argv[1:])
