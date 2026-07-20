"""Required Weights & Biases setup shared by all training entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
INVALID_API_KEYS = {
    "",
    "your_wandb_api_key_here",
    "replace_with_your_wandb_api_key",
}


def read_env_file(path: str | os.PathLike[str] = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Read a small dotenv file without adding a python-dotenv dependency."""
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise FileNotFoundError(
            f"WandB configuration file was not found: {env_path}. "
            "Copy .env.example to .env and set WANDB_API_KEY."
        )

    values: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                raise ValueError(
                    f"invalid .env entry at {env_path}:{line_number}; expected KEY=VALUE"
                )
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if not key:
                raise ValueError(f"empty .env key at {env_path}:{line_number}")
            values[key] = value
    return values


def require_wandb_api_key(
    path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
) -> str:
    """Load WANDB_API_KEY from the project .env and expose it to WandB."""
    values = read_env_file(path)
    api_key = values.get("WANDB_API_KEY", "").strip()
    if api_key.lower() in INVALID_API_KEYS:
        raise ValueError(
            f"WANDB_API_KEY is missing or still a placeholder in {Path(path).resolve()}"
        )
    os.environ["WANDB_API_KEY"] = api_key
    return api_key


def initialize_wandb_run(
    *,
    project: str,
    name: str,
    config: Mapping[str, Any],
    entity: str = "",
    run_id: str = "",
    env_path: str | os.PathLike[str] = DEFAULT_ENV_PATH,
):
    """Authenticate from .env and create the mandatory online WandB run."""
    api_key = require_wandb_api_key(env_path)
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is required for training; install it with `python -m pip install wandb`"
        ) from exc

    wandb.login(key=api_key, relogin=True)
    init_kwargs = dict(
        project=project,
        entity=entity or None,
        name=name,
        config=dict(config),
        mode="online",
    )
    if run_id:
        init_kwargs.update(id=str(run_id), resume="must")
    return wandb.init(**init_kwargs)


def read_wandb_run_id_from_checkpoint(
    path: str | os.PathLike[str] | None,
) -> str:
    """Read a persisted WandB Run ID without constructing a training model."""
    if not path:
        return ""
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect a checkpoint") from exc
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch versions before weights_only was added
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return ""
    return str(checkpoint.get("wandb_run_id", "") or "").strip()


def wandb_run_id(run: Any) -> str:
    """Return a live run's stable ID, or an empty string for test doubles."""
    return str(getattr(run, "id", "") or "").strip()


def flatten_wandb_metrics(prefix: str, data: Any) -> dict[str, float]:
    """Flatten nested metric dicts, per-class vectors, and matrices to scalars."""
    output: dict[str, float] = {}

    def visit(path: str, value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "numel"):
            value = value.detach().float().cpu()
            if value.numel() == 1:
                output[path] = float(value.item())
                return
            visit(path, value.tolist())
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{path}/{key}", item)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{path}/{index}", item)
            return
        if isinstance(value, bool):
            output[path] = float(value)
        elif isinstance(value, (int, float)):
            output[path] = float(value)

    visit(prefix.rstrip("/"), data)
    return output


__all__ = [
    "DEFAULT_ENV_PATH",
    "flatten_wandb_metrics",
    "initialize_wandb_run",
    "read_wandb_run_id_from_checkpoint",
    "read_env_file",
    "require_wandb_api_key",
    "wandb_run_id",
]
