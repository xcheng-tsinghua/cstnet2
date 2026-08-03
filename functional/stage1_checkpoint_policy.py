from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STAGE1_PHASES = ("semantic", "geometry", "joint")
CHECKPOINT_POLICIES = ("auto", "restart", "resume")


@dataclass(frozen=True)
class Stage1CheckpointResolution:
    checkpoint_dir: Path
    action: str
    source: Path | None
    reason: str

    def print_summary(self, *, model: str, phase: str, policy: str) -> None:
        print("Stage 1 checkpoint resolution:")
        print(f"  model: {model}")
        print(f"  phase: {phase}")
        print(f"  policy: {policy}")
        print(f"  save directory: {self.checkpoint_dir}")
        print(f"  action: {self.action}")
        print(f"  source: {self.source if self.source is not None else '<none>'}")
        print(f"  reason: {self.reason}")


def stage1_phase_directory(checkpoint_root, model, phase) -> Path:
    if phase not in STAGE1_PHASES:
        raise ValueError(f"unsupported Stage 1 phase: {phase}")
    model = str(model).strip()
    if not model:
        raise ValueError("Stage 1 model name cannot be empty")
    return Path(checkpoint_root) / model / phase


def _initialization_candidates(checkpoint_root, model, phase) -> list[Path]:
    if phase == "geometry":
        semantic_dir = stage1_phase_directory(checkpoint_root, model, "semantic")
        return [
            semantic_dir / "best_constraint_score.pth",
            semantic_dir / "best_pmt_miou.pth",
            semantic_dir / "last.pth",
        ]
    if phase == "joint":
        geometry_dir = stage1_phase_directory(checkpoint_root, model, "geometry")
        return [geometry_dir / "last.pth"]
    return []


def _find_initialization_source(checkpoint_root, model, phase) -> Path:
    candidates = _initialization_candidates(checkpoint_root, model, phase)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"cannot start Stage 1 {phase!r} phase because no previous-phase "
        f"checkpoint was found. Searched:\n{searched}"
    )


def resolve_stage1_checkpoint(
    *,
    checkpoint_root,
    model,
    phase,
    policy="auto",
) -> Stage1CheckpointResolution:
    if phase not in STAGE1_PHASES:
        raise ValueError(f"unsupported Stage 1 phase: {phase}")
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(f"unsupported Stage 1 checkpoint policy: {policy}")

    checkpoint_dir = stage1_phase_directory(checkpoint_root, model, phase)
    current_last = checkpoint_dir / "last.pth"

    if policy == "resume":
        if not current_last.is_file():
            raise FileNotFoundError(
                f"checkpoint_policy='resume' requires: {current_last}"
            )
        return Stage1CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            action="resume",
            source=current_last,
            reason="resume policy requires the current phase last checkpoint",
        )

    if policy == "auto" and current_last.is_file():
        return Stage1CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            action="resume",
            source=current_last,
            reason="current phase last checkpoint exists",
        )

    if phase == "semantic":
        reason = (
            "restart policy ignores the current semantic checkpoint"
            if policy == "restart"
            else "no semantic checkpoint exists"
        )
        return Stage1CheckpointResolution(
            checkpoint_dir=checkpoint_dir,
            action="scratch",
            source=None,
            reason=reason,
        )

    source = _find_initialization_source(checkpoint_root, model, phase)
    reason = (
        f"restart policy reinitializes {phase} from the previous phase"
        if policy == "restart"
        else f"no current {phase} checkpoint exists; initialize from previous phase"
    )
    return Stage1CheckpointResolution(
        checkpoint_dir=checkpoint_dir,
        action="init",
        source=source,
        reason=reason,
    )
