"""
Google Cloud Storage utilities for Texas Crash Analysis Dashboard
Handles downloading data and model files from GCS bucket for deployment
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
import streamlit as st

# GCS Configuration
BUCKET_NAME = "mads-team-roadwatch-25"

# Mapping of local paths to GCS blob paths
GCS_PATHS = {
    # Data files
    "kaggle_crashes": "kaggle_us_accidents_texas.csv",
    "work_zones": "texas_wzdx_feed.csv",
    "hpms": "hpms_texas_2023.gpkg",

    # ML datasets - crash level (using timestamped files from bucket)
    "crash_ml_train": "crash_level/train_20251129_085128.csv",
    "crash_ml_val": "crash_level/val_20251129_085128.csv",
    "crash_ml_test": "crash_level/test_20251129_085128.csv",

    # ML datasets - segment level
    "segment_ml_train": "segment_level/train_latest.csv",
    "segment_ml_val": "segment_level/val_latest.csv",
    "segment_ml_test": "segment_level/test_latest.csv",

    # Production models
    "catboost_model": "production/catboost_calibrated_20251129_093533/pipeline.pkl",
    "catboost_metadata": "production/catboost_calibrated_20251129_093533/metadata.json",
    "random_forest_model": "production/random_forest_calibrated_optimized_20251129_093518/pipeline.pkl",
    "random_forest_metadata": "production/random_forest_calibrated_optimized_20251129_093518/metadata.json",
    "lightgbm_model": "production/lightgbm_calibrated_20251129_093553/pipeline.pkl",
    "lightgbm_metadata": "production/lightgbm_calibrated_20251129_093553/metadata.json",
}

# Model name to GCS key mapping
MODEL_NAME_TO_GCS = {
    "catboost_best_balanced": ("catboost_model", "catboost_metadata"),
    "random_forest_best_recall": ("random_forest_model", "random_forest_metadata"),
    "lightgbm_best_auc": ("lightgbm_model", "lightgbm_metadata"),
}

# Cache directory for downloaded files
CACHE_DIR = Path(tempfile.gettempdir()) / "roadwatch_cache"


def is_cloud_deployment() -> bool:
    """
    Check if running in cloud deployment mode (Streamlit Cloud or similar)

    Returns True if:
    - Running on Streamlit Cloud (st.secrets available with GCS creds)
    - GOOGLE_APPLICATION_CREDENTIALS env var is set
    - USE_GCS env var is explicitly set to 'true'
    """
    # Check for explicit GCS mode
    if os.environ.get("USE_GCS", "").lower() == "true":
        return True

    # Check for Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            return True
    except Exception:
        pass

    # Check for Google credentials env var pointing to a file that exists
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).exists():
        return True

    return False


def get_gcs_client():
    """
    Get authenticated GCS client

    Tries multiple authentication methods:
    1. Streamlit secrets (for Streamlit Cloud)
    2. GOOGLE_APPLICATION_CREDENTIALS env var
    3. Default credentials (for local development with gcloud auth)
    """
    from google.cloud import storage
    from google.oauth2 import service_account

    # Method 1: Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return storage.Client(credentials=credentials)
    except Exception:
        pass

    # Method 2 & 3: Use default (env var or ADC)
    return storage.Client()


@st.cache_resource(show_spinner=False)
def _get_cached_client():
    """Cached GCS client to avoid re-authentication"""
    return get_gcs_client()


def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cached_path(gcs_key: str) -> Path:
    """Get the local cache path for a GCS key"""
    blob_path = GCS_PATHS.get(gcs_key, gcs_key)
    # Flatten the path for caching (replace / with _)
    cache_filename = blob_path.replace("/", "_")
    return CACHE_DIR / cache_filename


def download_from_gcs(gcs_key: str, force: bool = False) -> Optional[Path]:
    """
    Download a file from GCS to local cache

    Args:
        gcs_key: Key from GCS_PATHS dict or direct blob path
        force: Force re-download even if cached

    Returns:
        Path to downloaded file, or None if download failed
    """
    ensure_cache_dir()

    blob_path = GCS_PATHS.get(gcs_key, gcs_key)
    local_path = get_cached_path(gcs_key)

    # Check cache first
    if local_path.exists() and not force:
        return local_path

    try:
        with st.spinner(f"Downloading {blob_path}..."):
            client = _get_cached_client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(blob_path)

            # Download to cache
            blob.download_to_filename(str(local_path))

        return local_path

    except Exception as e:
        st.error(f"Failed to download {blob_path} from GCS: {e}")
        return None


def download_model_files(model_name: str) -> tuple[Optional[Path], Optional[Path]]:
    """
    Download model pipeline and metadata files

    Args:
        model_name: Production model name (e.g., 'catboost_best_balanced')

    Returns:
        Tuple of (pipeline_path, metadata_path) or (None, None) if failed
    """
    if model_name not in MODEL_NAME_TO_GCS:
        st.error(f"Unknown model: {model_name}")
        return None, None

    model_key, metadata_key = MODEL_NAME_TO_GCS[model_name]

    pipeline_path = download_from_gcs(model_key)
    metadata_path = download_from_gcs(metadata_key)

    return pipeline_path, metadata_path


def clear_cache():
    """Clear all cached files"""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        st.success("Cache cleared!")
