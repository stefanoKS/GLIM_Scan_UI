"""The Houdini helper works without Houdini for loading and fails clearly for BGEO."""
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools' / 'houdini'
sys.path.insert(0, str(TOOLS))
import export_colored_bgeo as h  # noqa: E402


def test_load_input_requires_arrays(tmp_path):
    path = tmp_path/'colored_points.npz'
    np.savez(path, points=np.zeros((2, 3), np.float32),
             Cd=np.ones((2, 3), np.float32),
             color_confidence=np.array([0.5, 1.0], np.float32),
             color_count=np.array([1, 2], np.int32))
    data = h._load_input(path)
    assert data['points'].shape == (2, 3)
    assert data['Cd'].shape == (2, 3)


def test_load_input_missing_required_array(tmp_path):
    path = tmp_path/'bad.npz'
    np.savez(path, points=np.zeros((2, 3), np.float32))
    with pytest.raises(ValueError, match='required'):
        h._load_input(path)


def test_build_geometry_requires_houdini(tmp_path):
    path = tmp_path/'colored_points.npz'
    np.savez(path, points=np.zeros((2, 3), np.float32),
             Cd=np.ones((2, 3), np.float32),
             color_confidence=np.array([0.5, 1.0], np.float32),
             color_count=np.array([1, 2], np.int32))
    data = h._load_input(path)
    if sys.modules.get('hou') is not None:
        pytest.skip('Houdini hou module is installed')
    with pytest.raises(RuntimeError, match='hython'):
        h.build_geometry(data)
