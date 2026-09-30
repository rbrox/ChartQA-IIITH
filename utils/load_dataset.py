"""Download a Hugging Face dataset with pre-checks and graceful error handling.

Usage:
    export HF_TOKEN="hf_xxx"
    python download_dataset.py
"""
import os
import sys

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

TOKEN_DOCS = "https://huggingface.co/docs/hub/security-tokens"

load_dotenv()


def download_dataset(repo_id: str, local_dir: str | None = None) -> str | None:
    """Download `repo_id` dataset into `local_dir`.

    Returns the local path only if EVERY file was downloaded, else None.
    """
    # ---- Pre-check 0: Set path ------------------------------------
    if local_dir is None:
            local_dir = os.path.join("../datasets/", str(repo_id))
    
    # ---- Pre-check 1: valid HF token ------------------------------------
    token = os.getenv("HF_TOKEN")
    if not token:
        print(f"[ERROR] HF_TOKEN is not set. Create one and export it.\n"
              f"        Docs: {TOKEN_DOCS}")
        return None

    api = HfApi(token=token)
    try:
        user = api.whoami()["name"]
        print(f"[OK] Authenticated as '{user}'")
    except HfHubHTTPError:
        print(f"[ERROR] HF_TOKEN is invalid or expired. Get a new one.\n"
              f"        Docs: {TOKEN_DOCS}")
        return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("[ERROR] Cannot reach huggingface.co to verify the token. "
              "Check your internet connection and retry.")
        return None

    # ---- Pre-check 2: download, handling connection problems -------------
    try:
        expected_files = api.list_repo_files(repo_id, repo_type="dataset")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            token=token,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        print("[ERROR] Connection lost or not established. "
              "Download incomplete - re-run to resume.")
        return None
    except HfHubHTTPError as e:
        print(f"[ERROR] Hugging Face returned an error: {e}")
        return None
    except KeyboardInterrupt:
        print("[ABORTED] Download interrupted by user. Re-run to resume.")
        return None

    # ---- Verify the dataset is entirely downloaded ------------------------
    missing = [f for f in expected_files
               if not os.path.exists(os.path.join(path, f))]
    if missing:
        print(f"[ERROR] Incomplete download, {len(missing)} file(s) missing "
              f"(e.g. {missing[0]}). Re-run to resume.")
        return None

    print(f"[SUCCESS] Dataset fully downloaded to: {path}")
    return path


if __name__ == "__main__":
    result = download_dataset("HuggingFaceM4/ChartQA")  # <-- change this
    sys.exit(0 if result else 1)