# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities related to data saving/loading."""

import contextlib
import errno
import getpass
import glob
import hashlib
import importlib
import io
import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import IO, Any, Optional, Union

import fsspec
import fsspec.utils
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from lightning_utilities.core.imports import module_available

from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH

log = logging.getLogger(__name__)

# Streaming a small checkpoint straight from the object store beats paying for a local copy.
_CACHE_MIN_SIZE_BYTES = 128 * 1024 * 1024
_CACHE_DIR_PREFIX = "lightning_cache_"
_CACHE_POLL_INTERVAL_SECONDS = 0.05
_CACHE_WAIT_TIMEOUT_SECONDS = 3600.0
_LOCAL_RANK_KEYS = (
    "LOCAL_RANK",
    "SLURM_LOCALID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "JSM_NAMESPACE_LOCAL_RANK",
)


def _get_cache_roots() -> tuple[str, ...]:
    """Return candidate cache root directories in order of preference."""
    return ("/dev/shm", tempfile.gettempdir())


def _user_cache_prefix() -> str:
    """Namespace cache directories by UID so multi-user nodes never collide in /dev/shm or /tmp."""
    uid = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
    return f"{_CACHE_DIR_PREFIX}{uid}_"


def _get_local_rank() -> int:
    """Return the node-local rank of the current process (0 for single-process execution)."""
    for key in _LOCAL_RANK_KEYS:
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                continue
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return rank % torch.cuda.device_count()
        return rank
    return 0


def _session_token() -> str:
    """Return a token shared by local peer ranks spawned by the same launcher."""
    return f"{os.getppid()}:{os.environ.get('MASTER_PORT', '')}:{os.environ.get('TORCHELASTIC_RUN_ID', '')}"


def _find_cached_checkpoint(cache_subdir: str, file_size: int) -> Optional[str]:
    """Return the full path of an existing valid cached checkpoint across candidate roots, or ``None``."""
    for root in _get_cache_roots():
        candidate = os.path.join(root, cache_subdir, "checkpoint.ckpt")
        with contextlib.suppress(OSError):
            if os.path.exists(candidate) and os.path.getsize(candidate) == file_size:
                return candidate
    return None


def _torch_load(
    path: str,
    map_location: _MAP_LOCATION_TYPE,
    weights_only: Optional[bool],
) -> Any:
    """Load a local checkpoint, memory-mapping it when the file format allows."""
    if sys.platform != "win32":
        try:
            return torch.load(
                path,
                map_location=map_location,  # type: ignore[arg-type]
                weights_only=weights_only,
                mmap=True,
            )
        except RuntimeError as e:
            if "mmap" not in str(e):
                raise
            log.debug(f"Checkpoint {path} cannot be memory-mapped ({e}); loading it normally.")
    return torch.load(
        path,
        map_location=map_location,  # type: ignore[arg-type]
        weights_only=weights_only,
    )


def _remote_version(file_info: dict[str, Any]) -> str:
    """Return a token that changes whenever the remote object's content changes."""
    for key in ("etag", "ETag", "generation", "version_id", "mtime", "LastModified", "last_modified"):
        value = file_info.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return ""


def _reclaim_superseded_entries(path_digest: str, keep: str) -> None:
    """Delete older cache entries for the same remote path across candidate roots."""
    prefix = f"{_user_cache_prefix()}{path_digest}_"
    keep_base = os.path.basename(keep)
    for root in _get_cache_roots():
        for stale in glob.glob(os.path.join(root, f"{prefix}*")):
            if os.path.basename(stale) == keep_base:
                continue
            shutil.rmtree(stale, ignore_errors=True)


def clear_cache() -> None:
    """Remove every local checkpoint cache entry written by :func:`_load` for the current user."""
    prefix = _user_cache_prefix()
    for root in _get_cache_roots():
        for entry in glob.glob(os.path.join(root, f"{prefix}*")):
            shutil.rmtree(entry, ignore_errors=True)


def _load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
    weights_only: Optional[bool] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
        weights_only: If ``True``, restricts loading to ``state_dicts`` of plain ``torch.Tensor`` and other primitive
            types. If loading a checkpoint from a trusted source that contains an ``nn.Module``, use
            ``weights_only=False``. If loading checkpoint from an untrusted source, we recommend using
            ``weights_only=True``. For more information, please refer to the
            `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.

    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(
            path_or_url,
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
            weights_only=weights_only,
        )

    path_str = str(path_or_url)
    if path_str.startswith("http"):
        if weights_only is None:
            weights_only = False
            log.debug(
                f"Defaulting to `weights_only=False` for remote checkpoint: {path_or_url}."
                f" If loading a checkpoint from an untrusted source, we recommend using `weights_only=True`."
            )

        return torch.hub.load_state_dict_from_url(
            path_str,
            map_location=map_location,  # type: ignore[arg-type]
            weights_only=weights_only,
        )

    fs = get_filesystem(path_or_url)

    # 1. Local path optimization: no copy is needed, map the file directly.
    if _is_local_file_protocol(path_str):
        return _torch_load(fs._strip_protocol(path_str), map_location, weights_only)

    # 2. Remote checkpoint fetching via local_rank == 0 + fs.get_file
    try:
        file_info = fs.info(path_str)
        raw_size = file_info.get("size")
        file_size = int(raw_size) if raw_size is not None else 0
    except Exception:
        file_info = {}
        file_size = 0

    version = _remote_version(file_info)

    # Fall back to streaming for small files, unknown size, or a backend that exposes no version
    # token. Caching on size alone would serve stale bytes for an overwritten checkpoint.
    if file_size < _CACHE_MIN_SIZE_BYTES or not version:
        if file_size >= _CACHE_MIN_SIZE_BYTES:
            log.debug(
                f"{path_str} exposes no version token (etag/generation/mtime), so it is streamed"
                f" rather than cached locally."
            )
        with fs.open(path_str, "rb") as f:
            return torch.load(
                f,
                map_location=map_location,  # type: ignore[arg-type]
                weights_only=weights_only,
            )

    path_digest = hashlib.sha256(path_str.encode()).hexdigest()[:16]
    version_digest = hashlib.sha256(f"{version}:{file_size}".encode()).hexdigest()[:16]
    cache_subdir = f"{_user_cache_prefix()}{path_digest}_{version_digest}"

    # Check if already cached in any candidate root before checking free space or waiting.
    cached_path = _find_cached_checkpoint(cache_subdir, file_size)
    if cached_path is not None:
        return _torch_load(cached_path, map_location, weights_only)

    if _get_local_rank() == 0:
        roots = _get_cache_roots()
        shm_dir = roots[0] if roots else "/dev/shm"
        has_shm = os.path.exists(shm_dir) and os.access(shm_dir, os.W_OK)
        if has_shm:
            try:
                if shutil.disk_usage(shm_dir).free < file_size * 1.5:
                    has_shm = False
            except OSError:
                has_shm = False
        selected_root = shm_dir if has_shm else (roots[-1] if roots else tempfile.gettempdir())

        cache_dir = os.path.join(selected_root, cache_subdir)
        os.makedirs(cache_dir, mode=0o700, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(cache_dir, 0o700)

        local_path = os.path.join(cache_dir, "checkpoint.ckpt")
        error_path = os.path.join(cache_dir, "checkpoint.ckpt.err")
        staging_path = os.path.join(cache_dir, f"checkpoint.ckpt.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")

        for root in roots:
            with contextlib.suppress(OSError):
                os.remove(os.path.join(root, cache_subdir, "checkpoint.ckpt.err"))

        if not os.path.exists(local_path) or os.path.getsize(local_path) != file_size:
            size_gb = file_size / (1024**3)
            log.info(f"Fetching {path_str} ({size_gb:.2f} GB) to {staging_path} on local_rank=0...")

            try:
                fs.get_file(path_str, staging_path)
                staged_size = os.path.getsize(staging_path)
                if staged_size != file_size:
                    raise OSError(f"Truncated download of {path_str}: expected {file_size} bytes, got {staged_size}")
                os.chmod(staging_path, 0o600)
                os.replace(staging_path, local_path)
            except BaseException as exc:
                err_tmp = f"{error_path}.tmp.{os.getpid()}"
                with contextlib.suppress(OSError):
                    Path(err_tmp).write_text(f"{_session_token()}\n{type(exc).__name__}: {exc}", encoding="utf-8")
                    os.replace(err_tmp, error_path)
                raise
            finally:
                with contextlib.suppress(OSError):
                    if os.path.exists(staging_path):
                        os.remove(staging_path)

            _reclaim_superseded_entries(path_digest, cache_dir)
    else:
        wait_start = time.time()
        deadline = time.monotonic() + _CACHE_WAIT_TIMEOUT_SECONDS
        while True:
            cached_path = _find_cached_checkpoint(cache_subdir, file_size)
            if cached_path is not None:
                local_path = cached_path
                break
            for root in _get_cache_roots():
                err_candidate = os.path.join(root, cache_subdir, "checkpoint.ckpt.err")
                if os.path.exists(err_candidate):
                    with contextlib.suppress(OSError):
                        lines = Path(err_candidate).read_text(encoding="utf-8").splitlines()
                        err_session = lines[0] if len(lines) > 1 else ""
                        detail = lines[-1] if lines else ""
                        if err_session == _session_token() or os.path.getmtime(err_candidate) >= wait_start:
                            raise RuntimeError(f"local_rank=0 failed to download {path_str}: {detail}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {_CACHE_WAIT_TIMEOUT_SECONDS}s waiting for local_rank=0"
                    f" to download {path_str}."
                )
            time.sleep(_CACHE_POLL_INTERVAL_SECONDS)

    return _torch_load(local_path, map_location, weights_only)


def get_filesystem(path: _PATH, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def _atomic_save(checkpoint: dict[str, Any], filepath: _PATH) -> None:
    """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.

    """
    log.debug(f"Saving checkpoint: {filepath}")

    try:
        # We use a transaction here to avoid file corruption if the save gets interrupted
        fs, urlpath = fsspec.core.url_to_fs(str(filepath))
        with fs.transaction:
            if _is_object_storage(fs):
                is_azure = False
                if module_available("adlfs"):
                    from adlfs import AzureBlobFileSystem

                    is_azure = isinstance(fs, AzureBlobFileSystem)

                # Object storage cannot stream `torch.save`, so build the payload in memory first.
                bytesbuffer = io.BytesIO()
                torch.save(checkpoint, bytesbuffer)
                if is_azure:
                    # Azure uses a plain write because adlfs stages blocks sequentially, making
                    # pipe() slower.
                    with fs.open(urlpath, "wb") as f:
                        f.write(bytesbuffer.getvalue())
                else:
                    # Use fs.pipe() for S3/GCS where it triggers parallel multipart uploads,
                    # giving 4-5x throughput improvement for checkpoints >= 500 MB.
                    fs.pipe(urlpath, bytesbuffer.getvalue())
            else:
                # Stream directly to the file so we never hold a second full copy of the checkpoint
                # in memory. This matters for large FSDP/ModelParallel full state dicts on local disk.
                with fs.open(urlpath, "wb") as f:
                    torch.save(checkpoint, f)
    except PermissionError as e:
        if isinstance(e.__context__, OSError) and getattr(e.__context__, "errno", None) == errno.EXDEV:
            raise RuntimeError(
                'Upgrade fsspec to enable cross-device local checkpoints: pip install "fsspec[http]>=2025.5.0"',
            ) from e


def _is_object_storage(fs: AbstractFileSystem) -> bool:
    if module_available("adlfs"):
        from adlfs import AzureBlobFileSystem

        if isinstance(fs, AzureBlobFileSystem):
            return True

    if module_available("gcsfs"):
        from gcsfs import GCSFileSystem

        if isinstance(fs, GCSFileSystem):
            return True

    if module_available("s3fs"):
        from s3fs import S3FileSystem

        if isinstance(fs, S3FileSystem):
            return True

    return False


def _is_dir(fs: AbstractFileSystem, path: Union[str, Path], strict: bool = False) -> bool:
    """Check if a path is directory-like.

    This function determines if a given path is considered directory-like, taking into account the behavior
    specific to object storage platforms. For other filesystems, it behaves similarly to the standard `fs.isdir`
    method.

    Args:
        fs: The filesystem to check the path against.
        path: The path or URL to be checked.
        strict: A flag specific to Object Storage platforms. If set to ``False``, any non-existing path is considered
            as a valid directory-like path. In such cases, the directory (and any non-existing parent directories)
            will be created on the fly. Defaults to False.

    """
    # Object storage fsspec's are inconsistent with other file systems because they do not have real directories,
    # see for instance https://gcsfs.readthedocs.io/en/latest/api.html?highlight=makedirs#gcsfs.core.GCSFileSystem.mkdir
    # In particular, `fs.makedirs` is a no-op so we use `strict=False` to consider any path as valid, except if the
    # path already exists but is a file
    if _is_object_storage(fs):
        if strict:
            return fs.isdir(path)

        # Check if the path is not already taken by a file. If not, it is considered a valid directory-like path
        # because the directory (and all non-existing parent directories) will be created on the fly.
        return not fs.isfile(path)

    return fs.isdir(path)


def _is_local_file_protocol(path: _PATH) -> bool:
    return fsspec.utils.get_protocol(str(path)) == "file"


def _resolve_path(path: _PATH) -> Union[str, Path]:
    """Return a ``Path`` for local file paths and a plain ``str`` for remote fsspec URLs.

    ``Path()`` collapses the double slash in a URL (e.g. ``gs://bucket`` -> ``gs:/bucket``),
    corrupting it, so remote URLs must be kept as strings.

    """
    if _is_local_file_protocol(str(path)):
        _, urlpath = url_to_fs(str(path))
        return Path(urlpath).expanduser().resolve()
    return str(path)


def _checkpoint_join(path: Union[str, Path], name: str) -> Union[str, Path]:
    """Join ``name`` onto a checkpoint ``path`` without corrupting remote URLs."""
    if isinstance(path, Path):
        return path / name

    # Remote URLs stay as strings because `Path`/`os.path.join` use local
    # filesystem semantics (e.g. '\' on Windows), which can corrupt URLs.
    return str(path).rstrip("/") + "/" + name


def _is_checkpoint_dir(path: Union[str, Path]) -> bool:
    """Return whether ``path`` points to an existing directory, supporting fsspec paths."""
    if isinstance(path, Path):
        return path.is_dir()
    return get_filesystem(path).isdir(str(path))


def _prepare_directory_checkpoint(path: Union[str, Path]) -> None:
    """Ensure ``path`` is a directory for a sharded checkpoint.

    Removes a conflicting file sitting at ``path`` and creates the directory. Creating a
    directory is a no-op on object storage, which has no real directories.

    """
    if isinstance(path, Path):
        if path.is_file():
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        return
    fs = get_filesystem(path)
    if fs.isfile(str(path)):
        fs.rm(str(path))
    if not _is_object_storage(fs):
        fs.makedirs(str(path), exist_ok=True)


def _remove_checkpoint(path: Union[str, Path]) -> None:
    """Remove a checkpoint file or directory (recursively), supporting fsspec paths."""
    if isinstance(path, Path):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        return
    fs = get_filesystem(path)
    if fs.exists(str(path)):
        fs.rm(str(path), recursive=True)


def _get_distributed_checkpoint_writer(path: _PATH) -> Any:
    if _is_local_file_protocol(str(path)):
        from torch.distributed.checkpoint import FileSystemWriter

        # FSDP's FileSystemWriter streams the tensors to disk to minimize memory peaks
        return FileSystemWriter(path=path, single_file_per_rank=True)
    FsspecWriter = _import_fsspec_dcp_filesystem("FsspecWriter")
    return FsspecWriter(path=str(path), single_file_per_rank=True)


def _get_distributed_checkpoint_reader(path: _PATH) -> Any:
    if _is_local_file_protocol(str(path)):
        from torch.distributed.checkpoint import FileSystemReader

        return FileSystemReader(path=path)
    FsspecReader = _import_fsspec_dcp_filesystem("FsspecReader")
    return FsspecReader(path=str(path))


def _import_fsspec_dcp_filesystem(name: str) -> Any:
    """Import ``FsspecReader``/``FsspecWriter`` from torch's private DCP fsspec module.

    These live in a private module that not every PyTorch build ships, so raise an actionable error
    instead of letting a bare ``ImportError`` surface from deep in the call stack.

    """
    try:
        module = importlib.import_module("torch.distributed.checkpoint._fsspec_filesystem")
    except ImportError as e:
        raise ImportError(
            "Remote (fsspec) distributed checkpoints require"
            " `torch.distributed.checkpoint._fsspec_filesystem`, which is not available in this"
            " PyTorch build. Use a local checkpoint path or upgrade PyTorch."
        ) from e
    return getattr(module, name)
