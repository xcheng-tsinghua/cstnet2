"""Visualize constraints from a local TXT/HDF5 sample or a remote HF dataset.

TXT files use the column order::

    x y z primitive_type dir_x dir_y dir_z dimension loc_x loc_y loc_z face_idx

Hugging Face HDF5 shards are read with HTTP Range requests, so selecting one
sample does not download the complete dataset or shard.
"""

from __future__ import annotations

import argparse
import colorsys
import io
import os
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote, urlsplit

import numpy as np

from data_utils.constraint_dataset_common import (
    load_constraint_point_file,
    split_constraint_columns,
)
from data_utils.stage1_h5 import (
    STAGE1_H5_FIELDS,
    STAGE1_H5_FORMAT,
    STAGE1_H5_VERSION,
    import_h5py,
)


PRIMITIVE_NAMES = ("plane", "cylinder", "cone", "sphere", "free-form/other")
PRIMITIVE_COLORS = np.asarray(
    [
        (0.1216, 0.4667, 0.7059),
        (1.0000, 0.4980, 0.0549),
        (0.1725, 0.6275, 0.1725),
        (0.8392, 0.1529, 0.1569),
        (0.5804, 0.4039, 0.7412),
    ],
    dtype=np.float32,
)
INVALID_COLOR = np.asarray((0.65, 0.65, 0.65), dtype=np.float32)
H5_POINT_FIELDS = (
    "xyz",
    "pmt",
    "direction",
    "dimension",
    "location",
    "affiliate_idx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize constraint components and Face indices of a point cloud"
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        help="local 12-column TXT file or a local Stage 1 HDF5 shard",
    )
    parser.add_argument(
        "--hf-repo",
        default="ZXCCHENGXI/cstnet2_stage1_mini",
        help="Hugging Face dataset repo, e.g. ZXCCHENGXI/cstnet2_stage1_mini",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face branch, tag, or commit (default: main)",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="global HF sample index, or sample index inside a local HDF5 shard",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="also save the complete view as PNG/PDF/SVG",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the interactive window (normally used together with --save)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="randomly display at most this many points; 0 displays all points",
    )
    parser.add_argument("--seed", type=int, default=0, help="downsampling seed")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-55.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--stream-block-mib",
        type=int,
        default=1,
        help="remote HTTP cache block size in MiB (default: 1)",
    )
    parser.add_argument(
        "--stream-cache-blocks",
        type=int,
        default=16,
        help="maximum remote blocks kept in memory (default: 16)",
    )
    args = parser.parse_args()

    if (args.source is None) == (args.hf_repo is None):
        parser.error("provide exactly one local source or --hf-repo")
    if args.max_points < 0:
        parser.error("--max-points must be non-negative")
    if args.point_size <= 0:
        parser.error("--point-size must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.no_show and args.save is None:
        parser.error("--no-show requires --save")
    if args.stream_block_mib <= 0:
        parser.error("--stream-block-mib must be positive")
    if args.stream_cache_blocks <= 0:
        parser.error("--stream-cache-blocks must be positive")
    return args


class HttpRangeReader(io.RawIOBase):
    """Seekable read-only HTTP file backed by a bounded in-memory block cache."""

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        block_size: int = 1 << 20,
        max_blocks: int = 16,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        try:
            import httpx
        except ImportError as error:
            raise ImportError(
                "remote streaming requires httpx; install huggingface_hub in "
                "the visualization environment"
            ) from error

        self._httpx = httpx
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout),
        )
        request_headers = dict(headers or {})
        first_headers = dict(request_headers)
        first_headers["Range"] = f"bytes=0-{block_size - 1}"
        try:
            response = self._client.get(url, headers=first_headers)
        except BaseException:
            self._client.close()
            raise
        if response.status_code != 206:
            self._client.close()
            raise RuntimeError(
                "remote server ignored the initial HTTP Range request; refusing "
                f"to download the complete HDF5 shard (status={response.status_code})"
            )
        content_range = response.headers.get("content-range", "")
        try:
            size = int(content_range.rsplit("/", maxsplit=1)[1])
        except (IndexError, ValueError) as error:
            self._client.close()
            raise RuntimeError(
                f"remote server returned an invalid Content-Range: {content_range!r}"
            ) from error

        self._url = str(response.url)
        self._headers = (
            request_headers
            if urlsplit(self._url).netloc == urlsplit(url).netloc
            else {}
        )
        self._size = size
        self._position = 0
        self._block_size = int(block_size)
        self._max_blocks = int(max_blocks)
        self._cache: OrderedDict[int, bytes] = OrderedDict({0: response.content})
        self.request_count = 1
        self.bytes_fetched = len(response.content)

    @property
    def size(self) -> int:
        return self._size

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._checkClosed()
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = int(position)
        return self._position

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self._cache.pop(block_index, None)
        if cached is not None:
            self._cache[block_index] = cached
            return cached

        start = block_index * self._block_size
        stop = min(start + self._block_size, self._size)
        if start >= self._size:
            return b""
        headers = dict(self._headers)
        headers["Range"] = f"bytes={start}-{stop - 1}"
        response = self._client.get(self._url, headers=headers)
        if response.status_code != 206:
            raise RuntimeError(
                "remote server ignored the HTTP Range request; refusing to "
                f"download the complete HDF5 shard (status={response.status_code})"
            )
        data = response.content
        expected_size = stop - start
        if len(data) != expected_size:
            raise OSError(
                f"incomplete HTTP Range response: expected {expected_size} bytes, "
                f"received {len(data)}"
            )

        self.request_count += 1
        self.bytes_fetched += len(data)
        self._cache[block_index] = data
        while len(self._cache) > self._max_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if size is None or size < 0:
            stop = self._size
        else:
            stop = min(self._position + size, self._size)
        if self._position >= stop:
            return b""

        parts = []
        while self._position < stop:
            block_index, offset = divmod(self._position, self._block_size)
            block = self._fetch_block(block_index)
            take = min(stop - self._position, len(block) - offset)
            if take <= 0:
                raise OSError("remote block ended before the requested byte range")
            parts.append(block[offset : offset + take])
            self._position += take
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def flush(self) -> None:
        self._checkClosed()

    def close(self) -> None:
        if not self.closed:
            self._cache.clear()
            self._client.close()
        super().close()


def _normalize_sample_index(index: int, sample_count: int) -> int:
    normalized = index + sample_count if index < 0 else index
    if normalized < 0 or normalized >= sample_count:
        raise IndexError(
            f"sample index {index} is outside a dataset with {sample_count} samples"
        )
    return normalized


def _validate_h5_file(h5_file) -> int:
    if h5_file.attrs.get("format") != STAGE1_H5_FORMAT:
        raise ValueError("not a cstnet2 Stage 1 HDF5 file")
    if int(h5_file.attrs.get("format_version", -1)) != STAGE1_H5_VERSION:
        raise ValueError("unsupported Stage 1 HDF5 format version")
    missing = [field for field in STAGE1_H5_FIELDS if field not in h5_file]
    if missing:
        raise ValueError(f"missing Stage 1 HDF5 fields: {missing}")
    offsets = h5_file["offsets"]
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("invalid Stage 1 HDF5 offsets")
    return len(offsets) - 1


def _read_h5_sample(h5_source, sample_index: int) -> tuple[tuple[np.ndarray, ...], int, int]:
    h5py = import_h5py()
    with h5py.File(h5_source, "r") as h5_file:
        sample_count = _validate_h5_file(h5_file)
        normalized_index = _normalize_sample_index(sample_index, sample_count)
        start, stop = (
            int(value)
            for value in h5_file["offsets"][normalized_index : normalized_index + 2]
        )
        fields = tuple(np.asarray(h5_file[name][start:stop]) for name in H5_POINT_FIELDS)

    xyz, pmt, direction, dimension, location, face_idx = fields
    fields = (
        xyz.astype(np.float32, copy=False),
        pmt.astype(np.int32, copy=False),
        direction.astype(np.float32, copy=False),
        dimension.astype(np.float32, copy=False),
        location.astype(np.float32, copy=False),
        face_idx.astype(np.int32, copy=False),
    )
    return fields, normalized_index, sample_count


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()


def _hf_resolve_url(repo: str, revision: str, path: str) -> str:
    repo = repo.removeprefix("hf://datasets/").strip("/")
    if repo.count("/") != 1:
        raise ValueError(f"expected Hugging Face repo as owner/name, got: {repo!r}")
    return (
        f"https://huggingface.co/datasets/{repo}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
    )


def _load_hf_manifest(repo: str, revision: str, token: str | None) -> dict:
    try:
        import httpx
    except ImportError as error:
        raise ImportError(
            "Hugging Face streaming requires httpx; install huggingface_hub"
        ) from error
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = _hf_resolve_url(repo, revision, "stage1_manifest.json")
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60.0),
    ) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        manifest = response.json()
    if manifest.get("format") != STAGE1_H5_FORMAT:
        raise ValueError(f"remote manifest has an unexpected format: {manifest.get('format')!r}")
    if int(manifest.get("format_version", -1)) != STAGE1_H5_VERSION:
        raise ValueError("remote manifest has an unsupported format version")
    return manifest


def _load_hf_sample(
    repo: str,
    revision: str,
    sample_index: int,
    *,
    block_size: int,
    max_blocks: int,
) -> tuple[tuple[np.ndarray, ...], str, dict[str, int]]:
    token = _hf_token()
    manifest = _load_hf_manifest(repo, revision, token)
    sample_count = int(manifest["sample_count"])
    samples_per_shard = int(manifest["samples_per_shard"])
    global_index = _normalize_sample_index(sample_index, sample_count)
    shard_index, local_index = divmod(global_index, samples_per_shard)
    shards = manifest["shards"]
    if shard_index >= len(shards):
        raise ValueError("remote manifest does not contain the expected HDF5 shard")

    shard_name = str(shards[shard_index])
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    reader = HttpRangeReader(
        _hf_resolve_url(repo, revision, shard_name),
        headers=headers,
        block_size=block_size,
        max_blocks=max_blocks,
    )
    try:
        fields, normalized_local_index, _ = _read_h5_sample(reader, local_index)
    except BaseException:
        reader.close()
        raise
    stream_stats = {
        "bytes_fetched": reader.bytes_fetched,
        "request_count": reader.request_count,
        "shard_size": reader.size,
    }
    reader.close()
    label = (
        f"{repo} | sample {global_index}/{sample_count - 1} | "
        f"{shard_name}:{normalized_local_index}"
    )
    return fields, label, stream_stats


def _load_local_source(
    source: Path,
    sample_index: int,
) -> tuple[tuple[np.ndarray, ...], str]:
    suffix = source.suffix.lower()
    if suffix == ".txt":
        if sample_index not in (0, -1):
            raise ValueError("--sample-index is only used with HDF5 or Hugging Face sources")
        point_set = load_constraint_point_file(
            source,
            task_name="constraint visualization",
        )
        return split_constraint_columns(point_set), source.name
    if suffix in {".h5", ".hdf5"}:
        fields, normalized_index, sample_count = _read_h5_sample(source, sample_index)
        return fields, f"{source.name} | sample {normalized_index}/{sample_count - 1}"
    raise ValueError(f"expected a .txt, .h5, or .hdf5 source, got: {source}")


def _categorical_colors(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable colors, sorted unique labels, and their palette."""
    labels = np.asarray(labels)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    palette = np.empty((len(unique_labels), 3), dtype=np.float32)
    for index in range(len(unique_labels)):
        # Golden-ratio hue spacing remains distinguishable for a moderate Face count.
        hue = (index * 0.618033988749895) % 1.0
        palette[index] = colorsys.hsv_to_rgb(hue, 0.68, 0.90)
    return palette[inverse], unique_labels, palette


def _vector_colors(vectors: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode a 3D vector as RGB after per-axis normalization."""
    vectors = np.asarray(vectors, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    colors = np.broadcast_to(INVALID_COLOR, vectors.shape).copy()
    if not np.any(valid):
        return colors

    valid_vectors = vectors[valid]
    low = valid_vectors.min(axis=0)
    high = valid_vectors.max(axis=0)
    span = high - low
    normalized = np.full_like(valid_vectors, 0.5)
    varying = span > 1e-8
    normalized[:, varying] = (
        (valid_vectors[:, varying] - low[varying]) / span[varying]
    )
    colors[valid] = np.clip(normalized, 0.0, 1.0)
    return colors


def _direction_colors(direction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Encode canonical direction components from [-1, 1] to RGB."""
    direction = np.asarray(direction, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    colors = np.broadcast_to(INVALID_COLOR, direction.shape).copy()
    colors[valid] = np.clip((direction[valid] + 1.0) * 0.5, 0.0, 1.0)
    return colors


def _sample_indices(point_count: int, max_points: int, seed: int) -> np.ndarray:
    if max_points == 0 or point_count <= max_points:
        return np.arange(point_count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(point_count, size=max_points, replace=False))


def _set_equal_axes(ax, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) * 0.5
    radius = max(float((high - low).max()) * 0.55, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _style_axis(ax, title: str, xyz: np.ndarray, elev: float, azim: float) -> None:
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xlabel("X", labelpad=-6)
    ax.set_ylabel("Y", labelpad=-6)
    ax.set_zlabel("Z", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=elev, azim=azim)
    _set_equal_axes(ax, xyz)


def _scatter(ax, xyz: np.ndarray, colors: np.ndarray, point_size: float) -> None:
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=colors,
        s=point_size,
        marker=".",
        linewidths=0,
        depthshade=False,
    )


def _representatives_by_face(
    xyz: np.ndarray,
    values: np.ndarray,
    face_idx: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a vector constraint and its point positions once per valid Face."""
    centers = []
    representatives = []
    for face in np.unique(face_idx[valid]):
        mask = valid & (face_idx == face)
        centers.append(xyz[mask].mean(axis=0))
        representatives.append(values[mask].mean(axis=0))
    if not centers:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.asarray(centers), np.asarray(representatives)


def create_constraint_figure(
    xyz: np.ndarray,
    primitive_type: np.ndarray,
    direction: np.ndarray,
    dimension: np.ndarray,
    location: np.ndarray,
    face_idx: np.ndarray,
    *,
    title: str,
    point_size: float,
    elev: float,
    azim: float,
):
    """Create the five-panel constraint visualization figure."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    figure = plt.figure(figsize=(17, 10), constrained_layout=True)
    figure.suptitle(title, fontsize=15)
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(5)]

    # 1. Primitive type (the one-hot constraint is stored as its class index on disk).
    primitive_colors = PRIMITIVE_COLORS[np.clip(primitive_type, 0, 4)]
    _scatter(axes[0], xyz, primitive_colors, point_size)
    primitive_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=PRIMITIVE_COLORS[index],
            markeredgecolor="none",
            label=f"{index}: {PRIMITIVE_NAMES[index]}",
        )
        for index in np.unique(primitive_type)
    ]
    axes[0].legend(handles=primitive_handles, loc="upper left", fontsize=7)
    _style_axis(axes[0], "Primitive type", xyz, elev, azim)

    # 2. Direction: RGB encodes (dx, dy, dz); one arrow is drawn for each Face.
    direction_valid = np.isin(primitive_type, (0, 1, 2))
    _scatter(axes[1], xyz, _direction_colors(direction, direction_valid), point_size)
    direction_centers, face_directions = _representatives_by_face(
        xyz, direction, face_idx, direction_valid
    )
    if len(direction_centers):
        norms = np.linalg.norm(face_directions, axis=1, keepdims=True)
        face_directions = face_directions / np.maximum(norms, 1e-8)
        arrow_length = max(float(np.ptp(xyz, axis=0).max()) * 0.12, 1e-6)
        axes[1].quiver(
            direction_centers[:, 0],
            direction_centers[:, 1],
            direction_centers[:, 2],
            face_directions[:, 0],
            face_directions[:, 1],
            face_directions[:, 2],
            length=arrow_length,
            normalize=True,
            color="black",
            linewidth=0.8,
            arrow_length_ratio=0.25,
        )
    _style_axis(axes[1], "Direction  (RGB = dx, dy, dz)", xyz, elev, azim)

    # 3. Dimension: invalid plane/free-form values are gray and excluded from scaling.
    dimension_valid = np.isin(primitive_type, (1, 2, 3)) & np.isfinite(dimension)
    dimension_colors = np.broadcast_to(INVALID_COLOR, (len(xyz), 3)).copy()
    if np.any(dimension_valid):
        valid_dimension = dimension[dimension_valid]
        value_min = float(valid_dimension.min())
        value_max = float(valid_dimension.max())
        if value_max <= value_min:
            value_max = value_min + 1e-8
        dimension_norm = Normalize(vmin=value_min, vmax=value_max)
        dimension_cmap = plt.get_cmap("viridis")
        dimension_colors[dimension_valid] = dimension_cmap(
            dimension_norm(valid_dimension)
        )[:, :3]
        scalar_map = plt.cm.ScalarMappable(norm=dimension_norm, cmap=dimension_cmap)
        scalar_map.set_array([])
        figure.colorbar(
            scalar_map,
            ax=axes[2],
            shrink=0.62,
            pad=0.02,
            label="radius / semi-angle",
        )
    _scatter(axes[2], xyz, dimension_colors, point_size)
    _style_axis(axes[2], "Dimension  (gray = invalid)", xyz, elev, azim)

    # 4. Location: per-axis normalized RGB plus one actual location marker per Face.
    location_valid = np.isin(primitive_type, (0, 1, 2, 3))
    _scatter(axes[3], xyz, _vector_colors(location, location_valid), point_size)
    _, face_locations = _representatives_by_face(xyz, location, face_idx, location_valid)
    if len(face_locations):
        axes[3].scatter(
            face_locations[:, 0],
            face_locations[:, 1],
            face_locations[:, 2],
            c="black",
            s=max(point_size * 12.0, 18.0),
            marker="x",
            linewidths=1.2,
            depthshade=False,
            label="Face location",
        )
        axes[3].legend(loc="upper left", fontsize=7)
    location_extent = np.concatenate((xyz, face_locations), axis=0) if len(face_locations) else xyz
    _style_axis(axes[3], "Location  (RGB = normalized x, y, z)", location_extent, elev, azim)

    # 5. Face index: every Face receives a deterministic categorical color.
    face_colors, unique_faces, face_palette = _categorical_colors(face_idx)
    _scatter(axes[4], xyz, face_colors, point_size)
    for face in unique_faces:
        face_center = xyz[face_idx == face].mean(axis=0)
        axes[4].text(
            face_center[0],
            face_center[1],
            face_center[2],
            str(int(face)),
            fontsize=7,
            ha="center",
            va="center",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
        )
    if len(unique_faces) <= 20:
        face_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5,
                markerfacecolor=color,
                markeredgecolor="none",
                label=str(int(face)),
            )
            for face, color in zip(unique_faces, face_palette)
        ]
        axes[4].legend(
            handles=face_handles,
            title="Face idx",
            loc="upper left",
            fontsize=7,
            title_fontsize=8,
            ncols=2 if len(unique_faces) > 10 else 1,
        )
    _style_axis(axes[4], f"Face index  ({len(unique_faces)} Faces)", xyz, elev, azim)

    # The sixth cell explains the encodings without obscuring any point cloud.
    info_axis = figure.add_subplot(2, 3, 6)
    info_axis.axis("off")
    info_axis.text(
        0.03,
        0.96,
        "Color encoding\n\n"
        "Primitive type: fixed categorical colors\n\n"
        "Direction: vector components mapped from [-1, 1] to RGB;\n"
        "black arrows show the mean direction of each Face\n\n"
        "Dimension: continuous viridis scale\n\n"
        "Location: each axis is normalized to RGB; black crosses\n"
        "show the actual per-Face location coordinates\n\n"
        "Face index: categorical colors with the index written at\n"
        "each Face center; a legend is also shown for up to 20 Faces\n\n"
        "Gray points indicate that a component is invalid for that\n"
        "primitive type.",
        va="top",
        fontsize=11,
        linespacing=1.25,
    )
    return figure


def main(args: argparse.Namespace) -> None:
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    stream_stats = None
    if args.hf_repo is not None:
        fields, source_label, stream_stats = _load_hf_sample(
            args.hf_repo,
            args.revision,
            args.sample_index,
            block_size=args.stream_block_mib * (1 << 20),
            max_blocks=args.stream_cache_blocks,
        )
    else:
        fields, source_label = _load_local_source(args.source, args.sample_index)
    xyz, primitive_type, direction, dimension, location, face_idx = fields

    if np.any((primitive_type < 0) | (primitive_type > 4)):
        invalid = np.unique(primitive_type[(primitive_type < 0) | (primitive_type > 4)])
        raise ValueError(f"primitive type must be in [0, 4], got {invalid.tolist()}")

    indices = _sample_indices(len(xyz), args.max_points, args.seed)
    xyz, primitive_type, direction, dimension, location, face_idx = (
        field[indices]
        for field in (xyz, primitive_type, direction, dimension, location, face_idx)
    )
    print(
        f"Loaded {source_label}: displaying {len(xyz)} points, "
        f"{len(np.unique(face_idx))} Faces"
    )
    if stream_stats is not None:
        print(
            f"Streamed {stream_stats['bytes_fetched'] / (1 << 20):.2f} MiB in "
            f"{stream_stats['request_count']} HTTP Range requests from a "
            f"{stream_stats['shard_size'] / (1 << 20):.2f} MiB shard"
        )

    figure = create_constraint_figure(
        xyz,
        primitive_type,
        direction,
        dimension,
        location,
        face_idx,
        title=source_label,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved visualization to: {args.save}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(figure)


if __name__ == "__main__":
    main(parse_args())
