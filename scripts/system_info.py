"""
System and Hardware Verification Script for Fovea-LiDAR.
Collects and logs:
  - OS & Kernel version
  - Python version & compiler
  - PyTorch version & build
  - CUDA availability & runtime version
  - GPU Device Name & VRAM (GB)
  - CPU Processor & Logical Core count
  - Total Physical RAM (GB)
Outputs formatted JSON to outputs/logs/system_info.json.
"""

import sys
import json
import platform
from pathlib import Path
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def getSystemInfo() -> dict:
    cudaAvailable = torch.cuda.is_available()
    gpuName = torch.cuda.get_device_name(0) if cudaAvailable else "None (CPU only)"
    gpuMemoryGb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cudaAvailable else 0.0
    cudaVersion = torch.version.cuda if cudaAvailable else "None"

    info = {
        "operatingSystem": platform.platform(),
        "osName": platform.system(),
        "osRelease": platform.release(),
        "osVersion": platform.version(),
        "pythonVersion": sys.version.split()[0],
        "pythonCompiler": platform.python_compiler(),
        "pytorchVersion": torch.__version__,
        "cudaAvailable": cudaAvailable,
        "cudaVersion": cudaVersion,
        "gpuName": gpuName,
        "gpuMemoryGb": gpuMemoryGb,
        "cpuProcessor": platform.processor(),
        "cpuLogicalCores": psutil.cpu_count(logical=True),
        "cpuPhysicalCores": psutil.cpu_count(logical=False),
        "totalRamGb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "availableRamGb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
    }
    return info


def main() -> None:
    info = getSystemInfo()

    outputDir = PROJECT_ROOT / "outputs" / "logs"
    outputDir.mkdir(parents=True, exist_ok=True)
    outputPath = outputDir / "system_info.json"

    with open(outputPath, "w") as f:
        json.dump(info, f, indent=2)

    print("=" * 65)
    print("  FOVEA-LiDAR HARDWARE & ENVIRONMENT AUDIT")
    print("=" * 65)
    print(f"  Operating System : {info['operatingSystem']}")
    print(f"  Python Version   : {info['pythonVersion']} ({info['pythonCompiler']})")
    print(f"  PyTorch Version  : {info['pytorchVersion']}")
    print(f"  CUDA Available   : {info['cudaAvailable']} (Version: {info['cudaVersion']})")
    print(f"  GPU Name         : {info['gpuName']}")
    print(f"  GPU VRAM         : {info['gpuMemoryGb']} GB")
    print(f"  CPU Processor    : {info['cpuProcessor']}")
    print(f"  CPU Cores        : {info['cpuPhysicalCores']} physical, {info['cpuLogicalCores']} logical")
    print(f"  System RAM       : {info['totalRamGb']} GB (Available: {info['availableRamGb']} GB)")
    print("=" * 65)
    print(f"  -> System info successfully logged to: {outputPath}")


if __name__ == "__main__":
    main()
