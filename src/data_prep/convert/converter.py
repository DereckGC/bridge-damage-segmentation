"""Convierte un dataset descargado a la estructura del proyecto (data/raw).

    1. LEER      readers/ transforma lo descargado en una lista de Sample (polígonos en PÍXELES).
    2. REPARTIR  assign_splits decide el grupo de cada foto (si no hay test anotado, divide val).
    3. ESCRIBIR  por cada foto anotada:
                     data/raw/images/<grupo>/<nombre>.jpg
                     data/raw/annotations/<grupo>/<nombre>.json   (formato que lee data_prep/prepare.py)
                 las fotos sin anotaciones van a inputs/sin_anotar.

Uso: opciones 1 y 2 del menú (python src/main.py) o python src/main.py convert --src ...
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from data_prep.convert.readers import READERS, detect_format, read_samples
from data_prep.convert.sample import Sample
from utils.labels import resolve_label


def to_annotation(sample: Sample, unknown: Counter) -> dict:
    """Sample -> JSON de anotación del proyecto (el que lee prepare_data.py; compatible con labelme)."""
    shapes = []
    for label, points in sample.shapes:
        cls = resolve_label(label)
        if cls is None:
            unknown[label] += 1
            continue
        if len(points) >= 3:  # un polígono necesita al menos 3 vértices
            shapes.append({"label": cls, "points": points, "shape_type": "polygon"})
    return {"imageWidth": sample.width, "imageHeight": sample.height,
            "imagePath": sample.image.name, "shapes": shapes}


def assign_splits(samples: list[Sample], test_frac: float, seed: int) -> dict[Path, str]:
    """Decide el destino de cada foto anotada. Si no hay split de test con anotaciones,
    se separa una fracción de val como test (siempre la misma, gracias a la semilla)."""
    labeled = [s for s in samples if s.labeled]
    destino = {s.image: s.split for s in labeled}
    if not any(s.split == "test" for s in labeled):
        val = sorted((s for s in labeled if s.split == "val"), key=lambda s: s.image.name)
        random.Random(seed).shuffle(val)
        for s in val[:int(len(val) * test_frac)]:
            destino[s.image] = "test"
    return destino


def inspect(samples: list[Sample], fmt: str) -> None:
    print(f"Formato: {fmt}  |  muestras: {len(samples)}")
    print("Por split (anotadas / sin anotar):")
    for split in sorted({s.split for s in samples}):
        group = [s for s in samples if s.split == split]
        print(f"  {split:<10} {sum(s.labeled for s in group):>6} / {sum(not s.labeled for s in group):>6}")
    print(f"Imágenes que todavía no están en disco: {sum(not s.image.is_file() for s in samples)}")
    print("\nPolígonos por etiqueta de origen -> clase del proyecto:")
    for label, n in Counter(lab for s in samples for lab, _ in s.shapes).most_common():
        cls = resolve_label(label)
        print(f"  {label:<30} {n:>7}  ->  {cls or '¡SIN EQUIVALENTE! (agregar a ALIASES en utils/labels.py)'}")


def run(src: str, fmt: str = "auto", raw: str = "data/raw", inputs: str = "inputs/sin_anotar",
        test_frac: float = 0.5, seed: int = 42, copy: bool = False, inspect_only: bool = False) -> None:
    """Punto de entrada: lo llama el orquestador (app/orchestrator.py)."""
    src, raw, inputs = Path(src), Path(raw), Path(inputs)
    if fmt == "auto":
        fmt = detect_format(src)
        if fmt is None:
            raise ValueError(f"No reconozco el formato de {src}. Indícalo con --format ({', '.join(READERS)}).")
    samples = read_samples(src, fmt)

    if inspect_only:
        inspect(samples, fmt)
        return

    if (raw / "images").is_dir() and any(p.is_file() for p in (raw / "images").rglob("*")):
        print("AVISO: data/raw/images ya contiene archivos. Si no son de este dataset, muévelos fuera "
              "para que prepare_data no los reporte como errores.\n")

    destino = assign_splits(samples, test_frac, seed)
    move_or_copy = shutil.copy2 if copy else shutil.move
    counts, unknown, missing = Counter(), Counter(), 0

    for s in samples:
        if not s.image.is_file():
            missing += 1
            continue
        if not s.labeled:  # sin respuestas: solo sirve para probar la detección al final
            inputs.mkdir(parents=True, exist_ok=True)
            move_or_copy(str(s.image), str(inputs / s.image.name))
            counts["inputs"] += 1
            continue
        split = destino[s.image]
        (raw / "images" / split).mkdir(parents=True, exist_ok=True)
        (raw / "annotations" / split).mkdir(parents=True, exist_ok=True)
        move_or_copy(str(s.image), str(raw / "images" / split / s.image.name))
        with open(raw / "annotations" / split / (s.image.stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(to_annotation(s, unknown), f)
        counts[split] += 1

    print(f"Formato: {fmt}\nResultado:")
    for k, n in sorted(counts.items()):
        print(f"  {k:<10} {n}")
    if missing:
        print(f"\nAVISO: {missing} imágenes no están en disco (¿la descarga no terminó?). "
              f"Termínala y vuelve a ejecutar este comando.")
    if unknown:
        print("\nAVISO: etiquetas sin equivalente en core/classes.py (se omitieron):", dict(unknown))
