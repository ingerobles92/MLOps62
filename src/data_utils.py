"""
Data Loading Utilities for DVC Integration
Author: Team 62 (extracted from notebooks/EDA/eda_V1.ipynb)
Enhanced by: Alexis Alduncin

Robust data loading with MD5 verification and Docker/local path detection.
"""

import os
import yaml
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from dvc.repo import Repo
from dvc.api import open as dvc_open, get_url
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


def ensure_repo_ready(repo_root: str = None) -> None:
    """
    Verify that repo_root is a valid Git + DVC project.

    Args:
        repo_root: Path to repository root (auto-detected if None)

    Raises:
        FileNotFoundError: If repo_root doesn't exist
        RuntimeError: If .git or .dvc is missing
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
    Calculate MD5 hash of a file to verify integrity against DVC metadata.

    Args:
        path: Absolute path to file
        chunk_size: Read chunk size in bytes (default 1 MB)

    Returns:
        MD5 hash hex string
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_expected_md5_from_dvc(pointer_path: str) -> Optional[str]:
    """
    Read expected MD5 from a .dvc pointer file.

    Args:
        pointer_path: Absolute path to .dvc file

    Returns:
        MD5 hash if found, None otherwise
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


def dvc_get_resolved_url(path_repo_rel: str, repo_root: str = None) -> str:
    """
    Resolve remote URL of DVC-tracked output (e.g., s3://.../<hash>).

    Args:
        path_repo_rel: Repository-relative path to file
        repo_root: Repository root (auto-detected if None)

    Returns:
        URL to remote blob location
    """
    if repo_root is None:
        repo_root = get_repo_root()

    with Repo.open(repo_root):
        return get_url(path_repo_rel, repo=repo_root)


def dvc_read_csv_verified(
    path_repo_rel: str,
    repo_root: str = None,
    prefer_dvc: bool = False,
    verify_local_md5: bool = True,
    pandas_read_csv_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Read DVC-tracked CSV with integrity verification.

    Strategy:
    - If prefer_dvc=True: Always read via DVC (max fidelity). Returns ("dvc").
    - If prefer_dvc=False:
        1) If local file exists and verify_local_md5=True: compare MD5
           - Match: read local (fast) -> ("local")
           - No match: fallback to DVC -> ("dvc")
        2) If file doesn't exist: read via DVC -> ("dvc")

    Args:
        path_repo_rel: Repository-relative path to CSV
        repo_root: Repository root (auto-detected if None)
        prefer_dvc: Force DVC reading (ignore local file)
        verify_local_md5: Validate local MD5 before trusting
        pandas_read_csv_kwargs: Additional kwargs for pd.read_csv()

    Returns:
        Tuple of (DataFrame, source) where source in {"local", "dvc"}

    Raises:
        Propagates errors if file not accessible locally or remotely
    """
    if repo_root is None:
        repo_root = get_repo_root()

    ensure_repo_ready(repo_root)

    if pandas_read_csv_kwargs is None:
        pandas_read_csv_kwargs = {}

    local_path = os.path.join(repo_root, path_repo_rel)
    dvc_pointer = local_path + ".dvc"

    # Force DVC reading
    if prefer_dvc:
        logger.info(f"Reading via DVC (forced): {path_repo_rel}")
        with Repo.open(repo_root):
            with dvc_open(path_repo_rel, repo=repo_root, mode="rb") as f:
                df = pd.read_csv(f, **pandas_read_csv_kwargs)
        logger.info(f"Loaded via DVC: {df.shape}")
        return df, "dvc"

    # Try local read with MD5 verification
    if os.path.exists(local_path):
        if verify_local_md5:
            expected = _read_expected_md5_from_dvc(dvc_pointer)
            if expected:
                try:
                    actual = _md5_file(local_path)
                    if actual == expected:
                        logger.info(f"Reading local file (MD5 verified): {path_repo_rel}")
                        df = pd.read_csv(local_path, **pandas_read_csv_kwargs)
                        logger.info(f"Loaded from local: {df.shape}")
                        return df, "local"
                    else:
                        logger.warning(f"MD5 mismatch: expected {expected}, got {actual}")
                except Exception as e:
                    logger.warning(f"MD5 verification failed: {e}")
        else:
            # No verification, read local directly
            logger.info(f"Reading local file (no verification): {path_repo_rel}")
            df = pd.read_csv(local_path, **pandas_read_csv_kwargs)
            logger.info(f"Loaded from local: {df.shape}")
            return df, "local"

    # Fallback: read via DVC
    logger.info(f"Reading via DVC (fallback): {path_repo_rel}")
    with Repo.open(repo_root):
        with dvc_open(path_repo_rel, repo=repo_root, mode="rb") as f:
            df = pd.read_csv(f, **pandas_read_csv_kwargs)
    logger.info(f"Loaded via DVC: {df.shape}")
    return df, "dvc"


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
