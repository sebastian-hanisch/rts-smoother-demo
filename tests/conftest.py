"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_state():
    """Zustand (Position 0, Geschwindigkeit 2) mit Kovarianz diag(1, 1), Zeitschritt 1 s.

    Handrechnung Vorhersage mit q = 1: F P F^T = [[2, 1], [1, 1]], Q = [[1/3, 1/2], [1/2, 1]], Summe [[7/3, 3/2], [3/2, 2]].
    Handrechnung Korrektur mit GPS z = 2, r = 1 auf diag(1, 1): S = 2, K = (1/2, 0), x = (1, 2), P = diag(1/2, 1).
    """
    return np.array([0.0, 2.0]), np.eye(2)
