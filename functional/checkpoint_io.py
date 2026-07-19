"""Shared fault-tolerant checkpoint writing for all training tasks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import torch


CHECKPOINT_SAVE_ATTEMPTS = 3
CHECKPOINT_RETRY_SECONDS = 1.0


def safe_torch_save(
    payload: Any,
    path: str | os.PathLike[str],
    *,
    attempts: int = CHECKPOINT_SAVE_ATTEMPTS,
    retry_seconds: float = CHECKPOINT_RETRY_SECONDS,
    logger: Callable[[str], None] = print,
) -> bool:
    """Atomically save a checkpoint, returning False after storage failures.

    Only errors raised by the storage operations are downgraded. Constructing
    the payload remains the caller's responsibility, so model/configuration
    bugs are not silently swallowed.
    """
    attempts = int(attempts)
    retry_seconds = float(retry_seconds)
    if attempts < 1:
        raise ValueError("checkpoint save attempts must be at least 1")
    if retry_seconds < 0:
        raise ValueError("checkpoint retry delay must not be negative")

    destination = Path(path)
    temporary = destination.with_suffix(
        destination.suffix + f".tmp.{os.getpid()}"
    )
    for attempt in range(1, attempts + 1):
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
            return True
        except (OSError, RuntimeError) as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger(
                    "WARNING: failed to remove temporary checkpoint "
                    f"{temporary}: {cleanup_error}"
                )
            if attempt < attempts:
                logger(
                    "WARNING: checkpoint save failed "
                    f"({attempt}/{attempts}) for {destination}: {error}; "
                    f"retrying in {retry_seconds:g}s"
                )
                time.sleep(retry_seconds)
                continue
            logger(
                "WARNING: checkpoint save skipped after "
                f"{attempts} failed attempts for {destination}: {error}. "
                "Training will continue, but this epoch is not recoverable from "
                "that checkpoint file."
            )
            return False

    return False  # pragma: no cover - the loop always returns
