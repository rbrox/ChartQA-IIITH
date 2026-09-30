"""Verify the running interpreter is the project's .venv and that installed
packages match requirements.txt. Repairs the environment if they don't."""
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def _pip(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", *args],
        capture_output=capture, text=True,
    )


def _parse_requirements(req_file: Path) -> dict[str, str | None]:
    """Return {package: pinned_version_or_None} from a requirements file."""
    reqs = {}
    for line in req_file.read_text().splitlines():
        line = line.split("#")[0].split(";")[0].strip()
        if not line or line.startswith("-"):
            continue
        name, _, ver = line.partition("==")
        reqs[name.split("[")[0].strip()] = ver.strip() or None
    return reqs


def _scan(reqs: dict[str, str | None]) -> tuple[list[str], list[str]]:
    """Return (missing, mismatched) package lists."""
    missing, mismatched = [], []
    for name, pinned in reqs.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            missing.append(name)
            continue
        if pinned and installed != pinned:
            mismatched.append(f"{name} (installed {installed}, need {pinned})")
    return missing, mismatched


def _wipe_environment() -> None:
    """Uninstall every package pip lists (pip/setuptools/wheel are kept)."""
    frozen = _pip("freeze", capture=True).stdout.strip()
    if not frozen:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(frozen)
    _pip("uninstall", "-y", "-r", tmp.name)
    Path(tmp.name).unlink(missing_ok=True)


def ensure_environment(venv_dir: str = ".venv",
                       requirements: str = "requirements.txt") -> bool:
    """Return True if the environment is ready, False otherwise."""
    # ---- 1. Running inside the project's .venv? -------------------------
    venv_path = (BASE / venv_dir).resolve()
    if Path(sys.prefix).resolve() != venv_path:
        print(f"[ERROR] Not running the {venv_dir} interpreter.\n"
              f"        Current: {sys.executable}\n"
              f"        Activate it (source {venv_dir}/bin/activate) or run "
              f"{venv_path / 'bin' / 'python'} directly.")
        return False
    print(f"[OK] Using venv interpreter: {sys.executable}")

    req_file = BASE / requirements
    if not req_file.exists():
        print(f"[ERROR] {req_file} not found.")
        return False
    reqs = _parse_requirements(req_file)

    # ---- 2 & 3. Check packages; install or rebuild as needed -------------
    missing, mismatched = _scan(reqs)
    conflicts = not missing and not mismatched and _pip("check", capture=True).returncode != 0

    if mismatched or conflicts:
        reason = mismatched or ["dependency conflict reported by pip check"]
        print(f"[WARN] Version clash: {reason}. Rebuilding from {requirements}...")
        _wipe_environment()
        install = _pip("install", "-r", str(req_file))
    elif missing:
        print(f"[INFO] Missing packages: {missing}. Installing from {requirements}...")
        install = _pip("install", "-r", str(req_file))
    else:
        print(f"[OK] All packages in {requirements} are installed and consistent.")
        return True

    # ---- Final verification ---------------------------------------------
    missing, mismatched = _scan(reqs)
    if install.returncode != 0 or missing or mismatched:
        print("[ERROR] Environment could not be repaired. Check pip output above "
              "and your internet connection.")
        return False
    print("[SUCCESS] Environment is ready.")
    return True


if __name__ == "__main__":
    sys.exit(0 if ensure_environment() else 1)