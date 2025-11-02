"""
Data Loading Utilities for DVC Integration
Author: Team 62 (extracted from notebooks/1.0-ERL-Utilidad-DVC-lectura-datasets.ipynb)
Enhanced by: Alexis Alduncin

Robust data loading with MD5 verification and Docker/local path detection.
"""

##Emanuel 10/10/2025 Update of libraries for DVC and MD5 verification.
# ====================================================================================
# ===============================================                                   ==
# Libraries for dataset verification with DVC. ==                                   ==
# ===============================================                                   ==
from pathlib import Path  # Cross-platform path handling                            ==
from typing import Dict, Tuple, Optional  # Optional type hints for better clarity  ==
import os  # File system and environment variable handling                          ==
import yaml  # Read .dvc (YAML) pointer files                                       ==
import hashlib  # Compute MD5 hashes to verify data integrity                       ==
import pandas as pd  # Read and manage tabular data                                 ==
import subprocess    # Execute SO commands                                          ==
# ====================================================================================
#from dvc.repo import Repo
#from dvc.api import open as dvc_open, get_url
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_repo_root() -> str:
    """
    Auto-detect repository root.

    Returns "/work" if running in Docker, otherwise finds Git repo root.

    Returns:
        Absolute path to repository root
    """
    # Check if running in Docker
    if os.path.exists("/work") and os.path.isdir("/work/.git"):
        logger.info("Detected Docker environment: /work")
        return "/work"

    # Find Git root from current directory
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            logger.info(f"Detected local Git repo: {parent}")
            return str(parent)

    # Fallback to current directory
    logger.warning(f"Git repo not found, using current directory: {current}")
    return str(current)

##Emanuel 10/10/2025 Updated error that cause a crash if file was not local at the machine.
def ensure_repo_ready(repo_root: str = "/work") -> None:
    """
    Verifies that:
    - `repo_root` is a valid project folder with Git and DVC.
    - Directory `repo_root` exists.
    - It contains a `.git` subdirectory (it's a Git repo).
    - It contains a `.dvc` subdirectory (it's a DVC repo).

    Raises:
    - FileNotFoundError if `repo_root` does not exist.
    - RuntimeError if `.git` or `.dvc` is missing.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f"Repo root does not exist: {repo_root}")
    if not os.path.isdir(os.path.join(repo_root, ".git")):
        raise RuntimeError(f"Not a Git repo: {repo_root}")
    if not os.path.isdir(os.path.join(repo_root, ".dvc")):
        raise RuntimeError(f"Not a DVC repo: {repo_root} (.dvc not found)")

    logger.info(f"Repository ready: {repo_root}")


def _md5_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Computes the MD5 hash of a file by streaming it from disk to verify integrity
    against the value stored by DVC in the `.dvc` pointer (default md5-based cache).

    Parameters:
    - path: absolute file path.
    - chunk_size: read block size in bytes (default 1 MB).

    Returns:
    - Hex MD5 string of the file content.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_expected_md5_from_dvc(pointer_path: str) -> Optional[str]:
    """
    Reads the expected MD5 from a single-file `.dvc` pointer.

    `.dvc` format:
      - md5: <hash>
      - hash: md5
      - path: <file_name>

    Parameters:
    - pointer_path: absolute path to the `.dvc` file.

    Returns:
    - The MD5 string if present, or None if the pointer does not exist / lacks md5.

    Use:
    - Compare the expected MD5 from `.dvc` with the actual local file MD5.
    """
    if not os.path.exists(pointer_path):
        return None

    with open(pointer_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    outs = data.get("outs") or []
    if not outs:
        return None

    out = outs[0]
    return out.get("md5") or out.get("checksum") or None

##Emanuel 10/10/2025 Obsoleted function
#def dvc_get_resolved_url(path_repo_rel: str, repo_root: str = None) -> str:

def _dvc_pull_target(path_repo_rel: str, repo_root: str = "/work") -> None:
    """
    Runs `dvc pull <path>` or `<path>.dvc` to materialize the correct version from the remote (S3)
    into the local workspace/cache. Raises if it fails (credentials, permissions, etc.).

    Parameters:
    - path_repo_rel: repo-relative path to fetch (e.g., "data/raw/file.csv").
    - repo_root: repo root (e.g., "/work").
    """
# Build possible targets
    dvc_pointer = os.path.join(repo_root, path_repo_rel + ".dvc")
    if os.path.exists(dvc_pointer):
        target = path_repo_rel + ".dvc"
    else:
        target = path_repo_rel

    # Try pulling
    result = subprocess.run(
        ["dvc", "pull", "--quiet", target],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    # Raise if failed
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to run 'dvc pull {target}':\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def dvc_read_csv_verified(
    path_repo_rel: str,
    repo_root: str = None,
    prefer_dvc: bool = False,
    verify_local_md5: bool = True,
    pandas_read_csv_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Read a DVC-versioned CSV ensuring integrity when reading locally.

    Strategy:
    - If `prefer_dvc=True`: force fetching the official version with `dvc pull`
      and then read locally. Returns ("pulled").
    - If `prefer_dvc=False`:
        1) If the local file exists and `verify_local_md5=True`, compare local MD5
           with the expected MD5 from the `.dvc` pointer.
           * If equal -> read local (fast). Returns ("local").
           * If NOT equal -> run `dvc pull` and read the official version. Returns ("pulled").
        2) If the file does NOT exist -> run `dvc pull` and read the official version. Returns ("pulled").

    Parameters:
    - path_repo_rel: repo-relative CSV path (e.g., "data/raw/file.csv").
    - repo_root: repo root (e.g., "/work").
    - prefer_dvc: if True, ignore local state and fetch official version with `dvc pull`.
    - verify_local_md5: if True, validate local MD5 before trusting local read.
    - pandas_read_csv_kwargs: kwargs for `pandas.read_csv()` (sep, encoding, etc.).

    Returns:
    - (df, source) where source ∈ {"local", "pulled"} describing the read source.

    Exceptions:
    - Raises if the file cannot be materialized from the remote (credentials,
      permissions, or missing blob).
    """
    if repo_root is None:
        repo_root = get_repo_root()

    ensure_repo_ready(repo_root)
    if pandas_read_csv_kwargs is None:
        pandas_read_csv_kwargs = {}

    local_path = os.path.join(repo_root, path_repo_rel)
    dvc_pointer = local_path + ".dvc"
    expected_md5 = _read_expected_md5_from_dvc(dvc_pointer)

    # 1) Forzar lectura "oficial"
    if prefer_dvc:
        logger.info(f"Reading via DVC (forced): {path_repo_rel}")
        _dvc_pull_target(path_repo_rel, repo_root)
        df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
        logger.info(f"Loaded via DVC: {df_local.shape}")
        return df_local, "pulled"

    # 2) Usar archivo local si existe
    if os.path.exists(local_path):
        if verify_local_md5 and expected_md5:
            try:
                md5_local = _md5_file(local_path)
                if md5_local == expected_md5:
                    logger.info(f"Reading local file (MD5 verified): {path_repo_rel}")
                    df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
                    logger.info(f"Loaded from local: {df_local.shape}")
                    return df_local, "local"
                else:
                    logger.warning(f"MD5 mismatch: expected {expected_md5}, got {md5_local}")
                    _dvc_pull_target(path_repo_rel, repo_root)
                    df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
                    return df_local, "pulled"
            except Exception as e:
                logger.warning(f"MD5 verification failed: {e}")
                _dvc_pull_target(path_repo_rel, repo_root)
                df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
                return df_local, "pulled"
        else:
            df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
            return df_local, "local"

    # 3) No hay archivo local → traer “oficial”
    _dvc_pull_target(path_repo_rel, repo_root)
    df_local = pd.read_csv(local_path, **pandas_read_csv_kwargs)
    return df_local, "pulled"


# Convenience function with config integration
def load_data(path: str = "data/raw/work_absenteeism_modified.csv", **kwargs) -> pd.DataFrame:
    """
    Simple wrapper for common use case.

    Args:
        path: Repository-relative path to CSV
        **kwargs: Additional arguments for pd.read_csv()

    Returns:
        DataFrame
    """
    df, source = dvc_read_csv_verified(path, pandas_read_csv_kwargs=kwargs)
    logger.info(f"Data loaded from {source}: {df.shape}")
    return df
