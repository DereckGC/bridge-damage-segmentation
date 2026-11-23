"""Lector del formato Supervisely: carpetas por grupo con un JSON por foto.

    <src>/.../<grupo>/img/<foto>.jpg
    <src>/.../<grupo>/ann/<foto>.jpg.json    {"size": {...}, "objects": [{"classTitle", "points": {"exterior"}}]}

Las coordenadas ya vienen en píxeles. Ejemplo: la copia de dacl10k en Dataset Ninja.
"""
import json
from pathlib import Path

from data_prep.convert.sample import Sample
from utils.files import is_image
from utils.labels import normalize_split


def matches(src: Path) -> bool:
    """¿La carpeta tiene este formato?"""
    return any(p.is_dir() for p in src.rglob("ann"))


def read_supervisely(src: Path) -> list[Sample]:
    """<split>/img + <split>/ann/<foto>.json -> Sample. Las coordenadas ya vienen en píxeles."""
    samples = []
    for img_dir in sorted(p for p in src.rglob("img") if p.is_dir()):
        split = normalize_split(img_dir.parent.name)
        group = []
        for img in sorted(p for p in img_dir.iterdir() if is_image(p)):
            ann_path = img_dir.parent / "ann" / (img.name + ".json")
            ann = json.load(open(ann_path, encoding="utf-8")) if ann_path.is_file() else {}
            sample = Sample(image=img, split=split, labeled=ann_path.is_file(),
                            width=ann.get("size", {}).get("width"), height=ann.get("size", {}).get("height"))
            for obj in ann.get("objects", []):
                if obj.get("geometryType", "polygon") == "polygon":
                    sample.shapes.append((obj.get("classTitle", "?"), obj["points"]["exterior"]))
            group.append(sample)
        # Supervisely crea un JSON vacío también para las fotos sin anotar (p. ej. el test oculto).
        # Regla: si NINGUNA foto de la carpeta tiene polígonos, la carpeta entera está sin anotar;
        # si alguna tiene, las vacías son ejemplos negativos (concreto sano) y se conservan.
        if not any(s.shapes for s in group):
            for s in group:
                s.labeled = False
        samples.extend(group)
    return samples
