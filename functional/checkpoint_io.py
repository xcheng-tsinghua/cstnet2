"""Shared fault-tolerant checkpoint writing for all training tasks."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from functional.console_io import safe_print


CHECKPOINT_SAVE_ATTEMPTS = 3
CHECKPOINT_RETRY_SECONDS = 1.0


def safe_torch_save(
    payload: Any,
    path: str | os.PathLike[str],
    *,
    attempts: int = CHECKPOINT_SAVE_ATTEMPTS,
    retry_seconds: float = CHECKPOINT_RETRY_SECONDS,
    logger: Callable[[str], None] = safe_print,
) -> bool:
    """Atomically save a checkpoint, returning False after storage failures.

    Only errors raised by the storage operations are downgraded. Constructing
    the payload remains the caller's responsibility, so model/configuration
    bugs are not silently swallowed.
    """
    return _atomic_save_with_retries(
        lambda temporary: torch.save(payload, temporary), path,
        attempts=attempts, retry_seconds=retry_seconds, logger=logger,
        errors=(OSError, RuntimeError), label="checkpoint",
    )


def safe_json_save(payload, path, *, attempts=CHECKPOINT_SAVE_ATTEMPTS,
                   retry_seconds=CHECKPOINT_RETRY_SECONDS, logger=safe_print):
    """Serialize before storage handling so invalid payloads still raise."""
    encoded = json.dumps(payload, ensure_ascii=False, indent=4)
    return _atomic_save_with_retries(
        lambda temporary: temporary.write_text(encoded, encoding="utf-8"), path,
        attempts=attempts, retry_seconds=retry_seconds, logger=logger,
        errors=(OSError,), label="JSON log",
    )


def _atomic_save_with_retries(writer, path, *, attempts, retry_seconds, logger,
                             errors, label):
    def report(message):
        try:
            logger(message)
        except OSError:
            pass

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
            writer(temporary)
            os.replace(temporary, destination)
            return True
        except errors as error:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                report(
                    f"WARNING: failed to remove temporary {label} "
                    f"{temporary}: {cleanup_error}"
                )
            if attempt < attempts:
                report(
                    f"WARNING: {label} save failed "
                    f"({attempt}/{attempts}) for {destination}: {error}; "
                    f"retrying in {retry_seconds:g}s"
                )
                time.sleep(retry_seconds)
                continue
            report(
                f"WARNING: {label} save skipped after "
                f"{attempts} failed attempts for {destination}: {error}. "
                "Training will continue, but this file was not updated."
            )
            return False

    return False  # pragma: no cover - the loop always returns
