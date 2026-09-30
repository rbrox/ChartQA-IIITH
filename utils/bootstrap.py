"""One-command project setup: creates <project root>/.venv (if needed) and
syncs it with requirements.txt. Safe to re-run any time.

Layout expected:
    ChartQA-IIITH/            <- PROJECT_ROOT (.venv and requirements.txt live here)
        .venv/
        requirements.txt
        utils/                <- SCRIPT_DIR (this file lives here)
            bootstrap.py
            venv_health_check.py

Usage (from anywhere, needs only system Python 3.10):
    python3.10 utils/bootstrap.py
"""
import subprocess
import sys
import venv
from pathlib import Path

REQUIRED_PYTHON = (3, 10)
SCRIPT_DIR = Path(__file__).resolve().parent      # .../ChartQA-IIITH/utils
PROJECT_ROOT = SCRIPT_DIR.parent                  # .../ChartQA-IIITH
VENV_DIR = PROJECT_ROOT / ".venv"
HEALTH_CHECK = SCRIPT_DIR / "ensure_env.py"


def venv_python() -> Path:
    sub = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return VENV_DIR / sub


def main() -> int:
    # 1. Right Python to create the venv?
    if sys.version_info[:2] != REQUIRED_PYTHON:
        print(f"[ERROR] Python {'.'.join(map(str, REQUIRED_PYTHON))} required, "
              f"found {sys.version.split()[0]}. Re-run with the correct "
              f"interpreter, e.g. python3.10 utils/bootstrap.py")
        return 1

    if not HEALTH_CHECK.exists():
        print(f"[ERROR] {HEALTH_CHECK} not found.")
        return 1

    # 2. Create the venv at the project root if it doesn't exist
    if not venv_python().exists():
        print(f"[INFO] Creating virtual environment at {VENV_DIR} ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"[OK] Found existing virtual environment at {VENV_DIR}")

    # 3. Upgrade pip inside the venv
    subprocess.run([str(venv_python()), "-m", "pip", "install", "--upgrade", "pip"],
                   check=False)

    # 4. Sync packages using the venv's own interpreter
    result = subprocess.run([str(venv_python()), str(HEALTH_CHECK)],
                            cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("[ERROR] Setup failed. See messages above.")
        return result.returncode

    activate = (r".venv\Scripts\activate" if sys.platform == "win32"
                else "source .venv/bin/activate")
    print(f"\n[DONE] Environment ready. From {PROJECT_ROOT}, activate it with:\n    {activate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())