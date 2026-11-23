"""Convierte data/raw (fotos + JSON de polígonos) en data/processed (fotos reducidas + máscaras).

Por cada foto de data/raw/images/<grupo>/:
    1. lee su JSON de data/raw/annotations/<grupo>/
    2. reduce la foto (lado largo = data.long_side del config)
    3. convierte los polígonos en una máscara (19, alto, ancho)
    4. guarda data/processed/<grupo>/images/<foto>.jpg y .../masks/<foto>.npz

Las operaciones genéricas (abrir/reducir/guardar fotos, crear/guardar máscaras) viven en utils/.

Uso: opción 3 del menú (python src/main.py) o python src/main.py prepare
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from core.classes import CLASS_TO_IDX, CLASSES, NUM_CLASSES
from utils.constants import SPLITS
from utils.files import ensure_dirs, list_images
from utils.images import load_image, resize_long_side, save_image
from utils.masks import pixels_per_class, polygons_to_mask, save_mask


def process_one(img_name: str, cfg: dict, split: str) -> dict:
    """Procesa UNA foto. Devuelve un pequeño reporte para las estadísticas (no la foto ni la máscara)."""
    raw, out = cfg["paths"]["raw"], cfg["paths"]["processed"]
    stem = os.path.splitext(img_name)[0]
    img_path = os.path.join(raw, "images", split, img_name)
    ann_path = os.path.join(raw, "annotations", split, stem + ".json")

    if not os.path.exists(ann_path):
        return {"name": img_name, "error": "does not have JSON data"}
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        img = load_image(img_path, (ann.get("imageWidth"), ann.get("imageHeight")))
    except Exception as e:  # imagen corrupta, JSON mal formado, etc.
        return {"name": img_name, "error": f"{type(e).__name__}: {e}"}

    # 1) Reducir la foto: lado largo = long_side, manteniendo la proporción.
    img_small, scale = resize_long_side(img, cfg["data"]["long_side"])
    h, w = img_small.shape[:2]

    # 2) Polígonos -> máscara (C, h, w) a la MISMA escala que la foto.
    mask, unknown = polygons_to_mask(ann.get("shapes", []), h, w, scale, CLASS_TO_IDX, NUM_CLASSES)

    # 3) Guardar foto y máscara con el mismo nombre, en el mismo grupo.
    save_image(os.path.join(out, split, "images", stem + ".jpg"), img_small)
    save_mask(os.path.join(out, split, "masks", stem + ".npz"), mask)

    return {"name": img_name, "error": None, "unknown": sorted(unknown),
            "pixels": pixels_per_class(mask), "total": h * w}


def print_stats(split: str, results: list) -> None:
    """Resumen de un grupo: procesadas, errores, etiquetas desconocidas y desbalance de clases."""
    ok = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    unknown = sorted({lab for r in ok for lab in r["unknown"]})

    print(f"\n=== {split}: {len(ok)} procesadas, {len(errors)} con error ===")
    for r in errors[:10]:
        print(f"  ERROR {r['name']}: {r['error']}")
    if len(errors) > 10:
        print(f"  ... y {len(errors) - 10} errores más")
    if unknown:
        print(f"  AVISO: etiquetas que no están en core/classes.py (se ignoraron): {unknown}")
    if not ok:
        return

    # Desbalance: qué fracción de fotos contiene cada clase y qué % de píxeles ocupa.
    pixels = np.stack([r["pixels"] for r in ok])                 # (N, C)
    total = sum(r["total"] for r in ok)
    print(f"  {'Clase':<15}{'% fotos':>9}{'% píxeles':>11}")
    for i, name in enumerate(CLASSES):
        pct_img = 100 * (pixels[:, i] > 0).mean()
        pct_px = 100 * pixels[:, i].sum() / total
        print(f"  {name:<15}{pct_img:>8.1f}%{pct_px:>10.3f}%")


def run(cfg: dict, workers: int, limit: int | None = None) -> None:
    """Punto de entrada: lo llama el orquestador (app/orchestrator.py).
    `limit` = procesar solo las N primeras fotos de cada grupo (None = todas); sirve para pruebas."""
    for split in SPLITS:
        img_dir = os.path.join(cfg["paths"]["raw"], "images", split)
        if not os.path.isdir(img_dir):
            print(f"[{split}] no existe {img_dir}, se omite.")
            continue
        names = list_images(img_dir)
        if limit:
            names = names[:limit]
        ensure_dirs(*(os.path.join(cfg["paths"]["processed"], split, sub) for sub in ("images", "masks")))

        # Varias fotos en paralelo (un proceso por núcleo de CPU).
        work = partial(process_one, cfg=cfg, split=split)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(work, names, chunksize=8), total=len(names), desc=split))
        print_stats(split, results)
