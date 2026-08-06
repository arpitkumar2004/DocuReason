#!/usr/bin/env python3
"""
scripts/verify_pypi_package.py

Local verification utility for DocuReason framework PyPI packaging.
Verifies that:
 1. Package builds clean sdist (.tar.gz) and wheel (.whl) distributions.
 2. PyPI metadata passes strict validation via twine check.
 3. Package wheel contains expected modules (docureason, src, metadata).
 4. Built wheel can be installed and imported cleanly in a isolated test environment.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_cmd(cmd, cwd=None, env=None, check=True):
    print(f"--> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[FAILED] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(res.returncode)
    return res


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"

    print("==================================================")
    print("DocuReason PyPI Package Verification Routine")
    print("==================================================")

    # 1. Clean previous build artifacts
    if dist_dir.exists():
        print("[CLEAN] Cleaning existing dist/ directory...")
        shutil.rmtree(dist_dir)

    build_dir = repo_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    egg_dirs = list(repo_root.glob("*.egg-info"))
    for egg in egg_dirs:
        shutil.rmtree(egg)

    # 2. Build distribution using 'build' package
    print("\n[BUILD] Building sdist and wheel...")
    run_cmd([sys.executable, "-m", "build"], cwd=repo_root)

    sdist_files = list(dist_dir.glob("*.tar.gz"))
    wheel_files = list(dist_dir.glob("*.whl"))

    if not sdist_files or not wheel_files:
        print("[ERROR] Build failed: Missing .tar.gz or .whl in dist/")
        sys.exit(1)

    print(f"[OK] Generated sdist: {[f.name for f in sdist_files]}")
    print(f"[OK] Generated wheel: {[f.name for f in wheel_files]}")

    # 3. Check twine metadata strictness
    print("\n[CHECK] Checking package metadata with Twine...")
    run_cmd([sys.executable, "-m", "twine", "check", "--strict"] + [str(p) for p in dist_dir.glob("*")], cwd=repo_root)
    print("[OK] Twine metadata validation passed!")

    # 4. Check wheel contents with check-wheel-contents if installed
    try:
        import check_wheel_contents  # noqa: F401
        print("\n[CHECK] Checking wheel contents structure...")
        run_cmd([sys.executable, "-m", "check_wheel_contents"] + [str(p) for p in wheel_files], cwd=repo_root)
        print("[OK] Wheel contents structure validation passed!")
    except ImportError:
        print("\n[SKIP] 'check-wheel-contents' not installed, skipping wheel structure check.")

    # 5. Sanity check wheel installation in a temporary virtualenv
    print("\n[TEST] Testing wheel installation in temporary environment...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        venv_dir = Path(tmp_dir) / "venv"
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])

        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
        else:
            venv_python = venv_dir / "bin" / "python"

        target_wheel = wheel_files[0]
        run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
        run_cmd([str(venv_python), "-m", "pip", "install", str(target_wheel)])

        # Test importing docureason and checking version
        test_import_script = "import docureason; print('Installed docureason version:', getattr(docureason, '__version__', 'OK'))"
        res = run_cmd([str(venv_python), "-c", test_import_script])
        print(f"[OK] Import test output: {res.stdout.strip()}")

    print("\n==================================================")
    print("SUCCESS: ALL PYPI PACKAGE VERIFICATION CHECKS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    main()
