#!/usr/bin/env python3
"""
Setup script that detects CUDA version and configures dependencies accordingly.
Run this script before `uv sync` to configure the project for your environment.
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Optional

# CUDA version to TensorFlow mapping
CUDA_CONFIG = {
    "11": {
        "python_requires": ">=3.9,<3.11",
        "tensorflow": '"tensorflow>=2.10.0,<2.11.0"',
        "tensorflow_hub": '"tensorflow-hub>=0.12.0"',
        "numpy": '"numpy>=1.23.0,<2.0.0"',
        "keras": None,
    },
    "12": {
        "python_requires": ">=3.9",
        "tensorflow": '"tensorflow[and-cuda]>=2.16.0"',
        "tensorflow_hub": '"tensorflow-hub>=0.16.0"',
        "numpy": '"numpy>=2.0.0"',
        "keras": '"keras>=3.0.0"',
    },
}

PYPROJECT_TEMPLATE = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src", "utils"]

[project]
name = "real-time-action-detection"
version = "0.1.0"
description = "Real-time action detection using pose estimation"
readme = "README.md"
requires-python = "{python_requires}"
dependencies = [
    "matplotlib>=3.5.0",
    {numpy}
    "opencv-python>=4.5.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "scikit-learn>=1.0.0",
    {tensorflow}
    {tensorflow_hub}
    "ultralytics>=8.0.0",
{keras}]
'''


def get_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi or nvcc."""
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            universal_newlines=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Try nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"release (\d+)\.", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return None


def generate_pyproject(cuda_major: str) -> str:
    """Generate pyproject.toml content for the given CUDA version."""
    config = CUDA_CONFIG.get(cuda_major)
    if not config:
        print(f"Warning: Unknown CUDA version {cuda_major}, defaulting to CUDA 11")
        config = CUDA_CONFIG["11"]

    keras_line = f"    {config['keras']},\n" if config["keras"] else ""

    return PYPROJECT_TEMPLATE.format(
        python_requires=config["python_requires"],
        numpy=config["numpy"] + ",",
        tensorflow=config["tensorflow"] + ",",
        tensorflow_hub=config["tensorflow_hub"] + ",",
        keras=keras_line,
    )


def main():
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    print("Detecting CUDA version...")
    cuda_version = get_cuda_version()

    if cuda_version:
        print(f"Detected CUDA {cuda_version}")
    else:
        print("Could not detect CUDA. Defaulting to CUDA 11 (CPU/older GPU).")
        cuda_version = "11"

    cuda_major = cuda_version if cuda_version in CUDA_CONFIG else "11"
    print(f"Configuring for CUDA {cuda_major}.x...")

    content = generate_pyproject(cuda_major)
    pyproject_path.write_text(content)
    print(f"Updated {pyproject_path}")

    print("\nNow run:")
    print("  uv lock && uv sync")


if __name__ == "__main__":
    main()
