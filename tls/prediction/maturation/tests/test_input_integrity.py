#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/features.py'
spec = importlib.util.spec_from_file_location('maturation_features', MODULE_PATH)
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))

    def forward(self, image):
        return torch.zeros(len(image), features.FEAT_DIM)


class InputIntegrityTests(unittest.TestCase):
    def test_centers_outside_lower_grid_edges_are_excluded(self):

        x = features.build_tiles(np.ones((50, 1), int), 70, 0, 10, 1)
        y = features.build_tiles(np.ones((1, 50), int), 0, 70, 10, 1)
        self.assertEqual(len(x), 0)
        self.assertEqual(len(y), 0)

    def test_lower_edge_inclusive_upper_edge_exclusive(self):
        inside = features.build_tiles(np.ones((1, 1), int), 64, 64, 10, 1)
        outside = features.build_tiles(np.ones((1, 1), int), 54, 64, 10, 1)
        self.assertEqual(inside[['x_px', 'y_px', 'region']].values.tolist(), [[0, 0, 1]])
        self.assertEqual(len(outside), 0)

    def test_physical_and_region_contracts(self):
        for grid, xmin, resolution, mpp in [
            (np.ones((1, 1)), np.nan, 10, 1),
            (np.ones((1, 1)), 0, np.inf, 1),
            (np.ones((1, 1)), 0, 10, 0),
            (np.array([[0.5]]), 0, 10, 1),
            (np.array([[-1]]), 0, 10, 1),
            (np.array([[np.nan]]), 0, 10, 1),
        ]:
            with self.subTest(xmin=xmin, resolution=resolution, mpp=mpp, grid=grid):
                with self.assertRaises(ValueError):
                    features.build_tiles(grid, xmin, 0, resolution, mpp)

    def test_checkpoint_requires_complete_matching_state(self):
        calls = []
        timm = types.ModuleType('timm')
        layers = types.ModuleType('timm.layers')
        layers.SwiGLUPacked = object

        def create_model(name, **kwargs):
            calls.append((name, kwargs))
            return TinyBackbone()

        timm.create_model = create_model
        with tempfile.TemporaryDirectory(prefix='maturation-integrity-') as tmp:
            path = Path(tmp) / 'weights.pt'
            with patch.dict(sys.modules, {'timm': timm, 'timm.layers': layers}), patch.object(features, 'UNI2_WEIGHTS', path):
                torch.save({'weight': torch.tensor([2., 3.])}, path)
                model = features.load_uni2(torch.device('cpu'))
                self.assertTrue(torch.equal(model.weight, torch.tensor([2., 3.])))
                self.assertFalse(model.weight.requires_grad)
                self.assertFalse(model.training)
                for state in [{}, {'weight': torch.ones(2), 'extra': torch.ones(1)}, {'weight': torch.ones(3)}]:
                    torch.save(state, path)
                    with self.subTest(keys=list(state)):
                        with self.assertRaises(RuntimeError):
                            features.load_uni2(torch.device('cpu'))
        self.assertEqual(calls[0][0], 'vit_giant_patch14_224')
        self.assertEqual(calls[0][1]['embed_dim'], 1536)
        self.assertEqual(calls[0][1]['reg_tokens'], 8)
        self.assertEqual(calls[0][1]['depth'], 24)


if __name__ == '__main__':
    unittest.main(verbosity=2)
