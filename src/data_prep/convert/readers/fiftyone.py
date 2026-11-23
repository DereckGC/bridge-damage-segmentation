"""Lector del formato FiftyOne: un solo samples.json con todas las fotos y sus anotaciones.

    <src>/samples.json          {"samples": [{"filepath", "tags", "metadata", "semseg_ground_truth_filled"}, ...]}
    <src>/data/data_N/*.jpg     las imágenes

Los polígonos vienen con coordenadas NORMALIZADAS (0 a 1) y aquí se pasan a píxeles.
Ejemplo: la copia de dacl10k en Hugging Face (Voxel51).
"""
import json
from pathlib import Path

from data_prep.convert.sample import Sample
from utils.labels import normalize_split


def matches(src: Path) -> bool:
    """¿La carpeta tiene este formato?"""
    return (src / "samples.json").is_file()


def read_fiftyone(src: Path) -> list[Sample]:
    """samples.json -> Sample. Las coordenadas vienen normalizadas (0-1) y se pasan a píxeles."""
    with open(src / "samples.json", "r", encoding="utf-8") as f:
        raw_samples = json.load(f)["samples"]

    samples = []
    for s in raw_samples:
        split = next((normalize_split(t) for t in s.get("tags", [])
                      if normalize_split(t) in ("train", "val", "test", "testdev")), "sin_split")
        polylines = (s.get("semseg_ground_truth_filled") or {}).get("polylines")
        meta = s.get("metadata") or {}
        w, h = meta.get("width"), meta.get("height")
        sample = Sample(image=src / s["filepath"], split=split, labeled=polylines is not None, width=w, height=h)
        for poly in polylines or []:
            for figure in poly["points"]:  # un polyline puede tener varias figuras
                # recortar a [0, 1]: algunos vértices quedaron un poco fuera de la foto
                pts = [[min(max(x, 0.0), 1.0) * w, min(max(y, 0.0), 1.0) * h] for x, y in figure]
                sample.shapes.append((poly["label"], pts))
        samples.append(sample)
    return samples
