"""Revisión visual de las máscaras: dibuja la máscara encima de su foto para comprobar a ojo
que los daños quedaron en el lugar correcto, ANTES de entrenar.

Dos modos:
    normal      foto completa | foto con la máscara pintada encima
    aumentado   la misma foto con 3 aumentaciones distintas (recorte, giro, color...), cada una
                con su máscara: así se comprueba que training/dataset.py mantiene foto y máscara alineadas

Resultado: <paths.outputs>/check_masks/<grupo>/<foto>_check.jpg  (o _aumentada.jpg)

Uso: opción 4 del menú (python src/main.py) o python src/main.py visualize --split train --n 10
"""
import random
from pathlib import Path

import numpy as np

from core.classes import CLASSES, COLORS
from utils.drawing import overlay_mask, side_by_side
from utils.files import ensure_dirs, list_images
from utils.images import read_image_rgb, save_image
from utils.masks import load_mask

AUGMENTED_VARIANTS = 3


def run(cfg: dict, split: str = "train", n: int = 10, name: str | None = None, augmented: bool = False) -> None:
    """Punto de entrada reutilizable: lo llama el orquestador."""
    base = Path(cfg["paths"]["processed"]) / split
    out_dir = Path(cfg["paths"].get("outputs", "outputs")) / "check_masks" / split
    ensure_dirs(out_dir)

    if name:
        names = [name]
    else:
        available = [Path(f).stem for f in list_images(base / "images") if (base / "masks" / f"{Path(f).stem}.npz").is_file()]
        names = random.Random(0).sample(available, min(n, len(available)))   # semilla fija: siempre las mismas

    transform = None
    if augmented:
        from training.augmentations import build_train_transforms   # solo se carga si se pide este modo
        transform = build_train_transforms(cfg["data"].get("crop", 512), normalize=False)

    for stem in names:
        img = read_image_rgb(base / "images" / f"{stem}.jpg")
        mask = load_mask(base / "masks" / f"{stem}.npz")
        if transform is None:
            result = side_by_side(img, overlay_mask(img, mask, COLORS, CLASSES))
            path = out_dir / f"{stem}_check.jpg"
        else:
            variants = []
            for _ in range(AUGMENTED_VARIANTS):
                out = transform(image=img, mask=mask.transpose(1, 2, 0).astype(np.uint8))
                aug_mask = out["mask"].transpose(2, 0, 1).astype(bool)
                variants.append(overlay_mask(out["image"], aug_mask, COLORS, CLASSES))
            result = side_by_side(*variants)
            path = out_dir / f"{stem}_aumentada.jpg"
        save_image(path, result)
        classes_found = [CLASSES[c] for c in range(mask.shape[0]) if mask[c].any()]
        print(f"{path}  ->  {', '.join(classes_found) or 'sin daños'}")

    print(f"\nAbre las imágenes de {out_dir} y comprueba que cada color cae sobre el daño que corresponde.")
