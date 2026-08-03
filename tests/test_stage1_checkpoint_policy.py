from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from functional.stage1_checkpoint_policy import resolve_stage1_checkpoint


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


class Stage1CheckpointPolicyTest(unittest.TestCase):
    def test_model_and_phase_determine_save_directory(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=temporary,
                model="pointnet2",
                phase="semantic",
            )
        self.assertEqual(
            resolution.checkpoint_dir,
            Path(temporary) / "pointnet2" / "semantic",
        )
        self.assertEqual(resolution.action, "scratch")

    def test_auto_resumes_current_phase_last(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            last = Path(temporary) / "pointnet" / "geometry" / "last.pth"
            _touch(last)
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=temporary,
                model="pointnet",
                phase="geometry",
            )
        self.assertEqual(resolution.action, "resume")
        self.assertEqual(resolution.source, last)

    def test_geometry_uses_semantic_fallback_order(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            semantic = Path(temporary) / "attn_3dgcn" / "semantic"
            _touch(semantic / "last.pth")
            _touch(semantic / "best_pmt_miou.pth")
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=temporary,
                model="attn_3dgcn",
                phase="geometry",
            )
            self.assertEqual(resolution.source, semantic / "best_pmt_miou.pth")

            _touch(semantic / "best_constraint_score.pth")
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=temporary,
                model="attn_3dgcn",
                phase="geometry",
            )
        self.assertEqual(resolution.action, "init")
        self.assertEqual(resolution.source, semantic / "best_constraint_score.pth")

    def test_joint_uses_geometry_last(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            geometry_last = Path(temporary) / "pointnet" / "geometry" / "last.pth"
            _touch(geometry_last)
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=temporary,
                model="pointnet",
                phase="joint",
            )
        self.assertEqual(resolution.action, "init")
        self.assertEqual(resolution.source, geometry_last)

    def test_restart_ignores_current_phase_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            root = Path(temporary)
            _touch(root / "pointnet" / "geometry" / "last.pth")
            semantic_best = root / "pointnet" / "semantic" / "best_constraint_score.pth"
            _touch(semantic_best)
            resolution = resolve_stage1_checkpoint(
                checkpoint_root=root,
                model="pointnet",
                phase="geometry",
                policy="restart",
            )
        self.assertEqual(resolution.action, "init")
        self.assertEqual(resolution.source, semantic_best)

    def test_missing_required_checkpoint_is_an_error(self):
        with tempfile.TemporaryDirectory(dir=".") as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "previous-phase"):
                resolve_stage1_checkpoint(
                    checkpoint_root=temporary,
                    model="pointnet",
                    phase="joint",
                )
            with self.assertRaisesRegex(FileNotFoundError, "requires"):
                resolve_stage1_checkpoint(
                    checkpoint_root=temporary,
                    model="pointnet",
                    phase="semantic",
                    policy="resume",
                )


if __name__ == "__main__":
    unittest.main()
