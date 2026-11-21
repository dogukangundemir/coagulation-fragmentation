"""Make the flat test modules importable under any pytest import mode.

`maze_synthetic` lives next to the tests; with --import-mode=importlib pytest
no longer inserts the test directory into sys.path, so do it here.
"""
import sys
from pathlib import Path

TESTS_DIR = str(Path(__file__).resolve().parent)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
