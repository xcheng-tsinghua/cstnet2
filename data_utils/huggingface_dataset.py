"""Resolve a Stage 1 dataset URL to Hugging Face's reusable disk cache."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlsplit


def parse_dataset_url(source: str):
    """Return (repo_id, revision, subdirectory), or None for a local path."""
    if "://" not in source:
        return None
    url = urlsplit(source)
    if (
        url.scheme != "https"
        or url.netloc != "huggingface.co"
        or url.username is not None
    ):
        raise ValueError("data_root URL must be an https://huggingface.co/datasets/... link")
    parts = url.path.strip("/").split("/")
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError("Expected a Hugging Face dataset repository URL")
    owner, repository = (unquote(part) for part in parts[1:3])
    for part in (owner, repository):
        if not part or part in {".", ".."} or any(c in part for c in "/\\:*?[]"):
            raise ValueError("Invalid Hugging Face dataset repository name")
    revision, subdirectory = "main", ""
    if len(parts) > 3:
        if len(parts) < 5 or parts[3] != "tree":
            raise ValueError("Use a repository or /tree/<revision> directory URL, not a file link")
        revision = unquote(parts[4])
        if not revision or "\\" in revision or any(p in {"", ".", ".."} for p in revision.split("/")):
            raise ValueError("Invalid Hugging Face revision")
        folders = [unquote(part) for part in parts[5:]]
        for part in folders:
            if not part or part in {".", ".."} or any(c in part for c in "/\\:*?[]"):
                raise ValueError("Invalid dataset subdirectory")
        subdirectory = "/".join(folders)
    return f"{owner}/{repository}", revision, subdirectory


def resolve_stage1_data_root(source, *, cache_dir=None, storage_format="auto") -> str:
    """Download missing dataset files before DataLoader workers are started.

    snapshot_download reuses unchanged files, validates the remote revision,
    and only returns after the selected snapshot files have been downloaded.
    Local datasets do not import or require huggingface_hub.
    """
    source = str(source)
    parsed = parse_dataset_url(source)
    if parsed is None:
        return source
    if storage_format not in {"auto", "h5", "txt"}:
        raise ValueError("storage_format must be auto, h5, or txt")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Hugging Face URLs require huggingface_hub. Install it in your training "
            "environment with: python -m pip install huggingface_hub"
        ) from error

    repo_id, revision, subdirectory = parsed
    patterns = []
    if storage_format in {"auto", "h5"}:
        patterns.extend(["*.[hH]5", "*.[hH][dD][fF]5"])
    if storage_format in {"auto", "txt"}:
        patterns.append("*.[tT][xX][tT]")
    if subdirectory:
        patterns = [f"{subdirectory}/{pattern}" for pattern in patterns]
    print(f"Stage 1 Hugging Face dataset: {repo_id}, revision={revision}")
    print("Downloading missing files; unchanged files reuse the Hugging Face cache.")
    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir else None,
        allow_patterns=patterns,
    )
    root = Path(snapshot) / subdirectory
    suffixes = {".h5", ".hdf5"} if storage_format == "h5" else {".txt"}
    if storage_format == "auto":
        suffixes.update({".h5", ".hdf5"})
    if not root.is_dir() or not any(
        path.is_file() and path.suffix.lower() in suffixes for path in root.rglob("*")
    ):
        raise FileNotFoundError(
            f"No Stage 1 {storage_format} data files in {repo_id}/{subdirectory}. "
            "Upload uncompressed Stage 1 HDF5 shards or TXT samples."
        )
    print(f"Stage 1 local dataset cache: {root}")
    return str(root)
