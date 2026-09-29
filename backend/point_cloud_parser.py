"""
TerraSight / Fovea-LiDAR Point Cloud Ingestion & Parsing Engine.
Supports multiple standard robotics and geospatial LiDAR formats:
  - .bin (SemanticKITTI 4-float x,y,z,intensity binary format)
  - .pcd (Point Cloud Data ASCII and binary headers)
  - .ply (Polygon File Format ASCII and basic binary)
  - .xyz / .txt / .csv (Whitespace or comma delimited coordinate tables)
  - .npy (NumPy array format)
"""

import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from core.point_cloud import PointCloud


def parsePointCloudBytes(filename: str, content: bytes) -> PointCloud:
    """
    Parse uploaded point cloud bytes into a normalized PointCloud container.
    """
    lowerName = filename.lower()

    if lowerName.endswith(".bin"):
        return _parseKittiBin(content)
    elif lowerName.endswith(".pcd"):
        return _parsePcd(content)
    elif lowerName.endswith(".ply"):
        return _parsePly(content)
    elif lowerName.endswith(".npy"):
        return _parseNpy(content)
    elif lowerName.endswith((".xyz", ".txt", ".csv")):
        return _parseDelimitedText(content)
    else:
        # Default attempt: binary float32 4-tuple or text fallback
        try:
            return _parseKittiBin(content)
        except Exception:
            return _parseDelimitedText(content)


def _parseKittiBin(content: bytes) -> PointCloud:
    """Parse raw IEEE-754 float32 4-tuple binary (x, y, z, intensity)."""
    rawArray = np.frombuffer(content, dtype=np.float32)
    if len(rawArray) % 4 != 0:
        # Truncate to nearest multiple of 4
        validLength = (len(rawArray) // 4) * 4
        rawArray = rawArray[:validLength]

    reshaped = rawArray.reshape(-1, 4)
    points = reshaped[:, :3].copy()
    intensity = reshaped[:, 3].copy()

    # Filter invalid NaNs / Infs
    validMask = np.isfinite(points).all(axis=1) & np.isfinite(intensity)
    points = points[validMask]
    intensity = intensity[validMask]

    return PointCloud(
        points=points.astype(np.float32),
        intensity=intensity.astype(np.float32),
    )


def _parsePcd(content: bytes) -> PointCloud:
    """Parse PCD (Point Cloud Data) file format."""
    lines = content.split(b"\n")
    headerEnd = 0
    fields = []
    numPoints = 0
    dataType = "ascii"

    for idx, rawLine in enumerate(lines):
        line = rawLine.decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0].upper()

        if tag == "FIELDS":
            fields = [p.lower() for p in parts[1:]]
        elif tag == "POINTS":
            numPoints = int(parts[1])
        elif tag == "DATA":
            dataType = parts[1].lower()
            headerEnd = idx + 1
            break

    if dataType == "ascii":
        dataLines = lines[headerEnd:]
        rows = []
        for line in dataLines:
            sLine = line.decode("utf-8", errors="ignore").strip()
            if not sLine:
                continue
            parts = [float(p) for p in sLine.split()]
            rows.append(parts)
        data = np.array(rows, dtype=np.float32)
    else:
        # Binary data starts after header
        rawBytes = b"\n".join(lines[headerEnd:])
        numFields = len(fields) if fields else 4
        data = np.frombuffer(rawBytes, dtype=np.float32)
        if len(data) % numFields != 0:
            data = data[: (len(data) // numFields) * numFields]
        data = data.reshape(-1, numFields)

    # Map fields
    points = data[:, :3].astype(np.float32)
    intensity = None
    if fields and "intensity" in fields:
        intIdx = fields.index("intensity")
        if intIdx < data.shape[1]:
            intensity = data[:, intIdx].astype(np.float32)
    elif data.shape[1] >= 4:
        intensity = data[:, 3].astype(np.float32)

    return PointCloud(points=points, intensity=intensity)


def _parsePly(content: bytes) -> PointCloud:
    """Parse PLY (Stanford Polygon File Format) point elements."""
    lines = content.split(b"\n")
    headerEnd = 0
    numVertices = 0
    properties = []
    isBinary = False

    for idx, rawLine in enumerate(lines):
        line = rawLine.decode("utf-8", errors="ignore").strip()
        if line.startswith("format binary"):
            isBinary = True
        elif line.startswith("element vertex"):
            numVertices = int(line.split()[-1])
        elif line.startswith("property"):
            properties.append(line.split()[-1].lower())
        elif line == "end_header":
            headerEnd = idx + 1
            break

    if not isBinary:
        dataLines = lines[headerEnd:]
        rows = []
        for line in dataLines:
            sLine = line.decode("utf-8", errors="ignore").strip()
            if not sLine:
                continue
            rows.append([float(p) for p in sLine.split()])
        data = np.array(rows, dtype=np.float32)
        points = data[:, :3]
        intensity = data[:, 3] if data.shape[1] > 3 else None
    else:
        rawBytes = b"\n".join(lines[headerEnd:])
        data = np.frombuffer(rawBytes, dtype=np.float32)
        numProps = len(properties) if properties else 3
        if len(data) % numProps != 0:
            data = data[: (len(data) // numProps) * numProps]
        data = data.reshape(-1, numProps)
        points = data[:, :3]
        intensity = data[:, 3] if data.shape[1] > 3 else None

    return PointCloud(points=points.astype(np.float32), intensity=intensity)


def _parseDelimitedText(content: bytes) -> PointCloud:
    """Parse delimited text (.xyz, .txt, .csv)."""
    text = content.decode("utf-8", errors="ignore")
    lines = text.strip().splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        # Handle commas, tabs, spaces
        cleaned = line.replace(",", " ").replace(";", " ")
        try:
            parts = [float(v) for v in cleaned.split()]
            if len(parts) >= 3:
                rows.append(parts[:4] if len(parts) >= 4 else parts[:3])
        except ValueError:
            continue

    if not rows:
        raise ValueError("Could not parse numeric point cloud from text file.")

    arr = np.array(rows, dtype=np.float32)
    points = arr[:, :3]
    intensity = arr[:, 3] if arr.shape[1] >= 4 else None
    return PointCloud(points=points, intensity=intensity)


def _parseNpy(content: bytes) -> PointCloud:
    """Parse NumPy .npy array bytes."""
    import io
    arr = np.load(io.BytesIO(content))
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected 2D array with >= 3 columns, got shape {arr.shape}")
    points = arr[:, :3].astype(np.float32)
    intensity = arr[:, 3].astype(np.float32) if arr.shape[1] >= 4 else None
    return PointCloud(points=points, intensity=intensity)
