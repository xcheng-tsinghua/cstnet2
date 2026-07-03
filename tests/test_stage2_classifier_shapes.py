from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on the active training env
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class Stage2ClassifierShapeTest(unittest.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest("PyTorch is required for Stage2 classifier shape tests")

        from functional.constraints import CONSTRAINT_DIM

        self.n_classes = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xyz = torch.randn(2, 1024, 3, device=self.device)
        self.constraints = torch.randn(2, 1024, CONSTRAINT_DIM, device=self.device)

    def _check_log_probs(self, model):
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            out = model(self.xyz, self.constraints)
        self.assertEqual(tuple(out.shape), (2, self.n_classes))

    def test_baseline_classifier_shape(self):
        from networks.stage2 import CstNetStage2Classifier

        self._check_log_probs(CstNetStage2Classifier(n_classes=self.n_classes))

    def test_discriminative_classifier_shapes(self):
        from networks.stage2 import CstNetStage2ClassifierDiscriminative

        model = CstNetStage2ClassifierDiscriminative(n_classes=self.n_classes)
        self._check_log_probs(model)
        model.eval()
        with torch.no_grad():
            aux = model(self.xyz, self.constraints, return_aux=True)

        self.assertEqual(tuple(aux["log_probs"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["main_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_component_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_constraint_logits"].shape), (2, self.n_classes))

    def test_token_fusion_classifier_shapes(self):
        from networks.stage2 import CstNetStage2ClassifierTokenFusion

        model = CstNetStage2ClassifierTokenFusion(n_classes=self.n_classes)
        self._check_log_probs(model)
        model.eval()
        with torch.no_grad():
            aux = model(self.xyz, self.constraints, return_aux=True)

        self.assertEqual(tuple(aux["log_probs"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["main_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_final_constraint_logits"].shape), (2, self.n_classes))
        self.assertEqual(tuple(aux["aux_component_token_logits"].shape), (2, self.n_classes))


if __name__ == "__main__":
    unittest.main()
