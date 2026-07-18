from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch


def preload_cuda_nvrtc(cuda_version: str | None = None) -> str | None:
    """Load pip-packaged NVRTC globally for cuDNN attention on Linux.

    NVIDIA's pip wheels keep NVRTC below ``site-packages/nvidia``. That
    directory is not necessarily present in the dynamic loader search path of
    non-interactive processes such as ``nohup`` jobs, while cuDNN Frontend
    loads NVRTC by its soname at runtime.
    """
    if not sys.platform.startswith("linux"):
        return None
    cuda_version = cuda_version or torch.version.cuda
    if not cuda_version:
        return None

    cuda_major = cuda_version.split(".", maxsplit=1)[0]
    soname = f"libnvrtc.so.{cuda_major}"
    load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        ctypes.CDLL(soname, mode=load_mode)
        return soname
    except OSError as soname_error:
        candidates = []
        for entry in sys.path:
            if not entry:
                continue
            candidate = (
                Path(entry) / "nvidia" / "cuda_nvrtc" / "lib" / soname
            )
            if candidate.is_file():
                candidates.append(candidate)

        load_errors = []
        for candidate in candidates:
            try:
                ctypes.CDLL(str(candidate), mode=load_mode)
                return str(candidate)
            except OSError as error:
                load_errors.append(f"{candidate}: {error}")

        details = "; ".join(load_errors) if load_errors else str(soname_error)
        raise RuntimeError(
            f"Could not load {soname}, which is required by CUDA attention "
            "kernels. Add the matching nvidia/cuda_nvrtc/lib directory to "
            "LD_LIBRARY_PATH or reinstall the PyTorch CUDA runtime. "
            f"Details: {details}"
        ) from soname_error
