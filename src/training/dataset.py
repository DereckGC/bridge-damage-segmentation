"""Dataset de PyTorch: entrega al modelo los pares foto + máscara de data/processed.

Por cada foto:
    1. carga data/processed/<grupo>/images/<foto>.jpg   (RGB, alto x ancho x 3)
       y     data/processed/<grupo>/masks/<foto>.npz    (19 x alto x ancho, True/False)
    2. aplica las transformaciones (augmentations.py):
         train      -> aumentaciones aleatorias + recorte crop x crop + normalización
         val / test -> solo normalización, foto completa
    3. devuelve tensores de PyTorch:
         foto     float32 (3, alto, ancho)      valores normalizados
         máscara  float32 (19, alto, ancho)     0 o 1 en cada píxel y clase
         nombre   str                           para identificar la foto (p. ej. en reportes)

Uso (lo hace training/trainer.py):
    from training.dataset import BridgeDamageDataset
    ds = BridgeDamageDataset("data/processed", "train", crop=512)
    foto, mascara, nombre = ds[0]
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.augmentations import build_eval_transforms, build_train_transforms
from core.classes import NUM_CLASSES
from utils.files import list_images
from utils.images import read_image_rgb
from utils.masks import load_mask


class BridgeDamageDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, crop: int = 512, augment: bool | None = None):
        """
        processed_dir: carpeta data/processed (paths.processed del config)
        split:         "train", "val" o "test"
        crop:          tamaño del recorte de entrenamiento (data.crop del config)
        augment:       aplicar aumentaciones. Por defecto: sí en train, no en val/test.
        """
        self.split = split
        self.img_dir = Path(processed_dir) / split / "images"
        self.mask_dir = Path(processed_dir) / split / "masks"
        # Solo fotos que tienen su máscara: si prepare_data se interrumpió, las incompletas se ignoran.
        self.names = [Path(n).stem for n in list_images(self.img_dir) if (self.mask_dir / f"{Path(n).stem}.npz").is_file()]
        if not self.names:
            raise FileNotFoundError(f"No hay pares foto + máscara en {self.img_dir} y {self.mask_dir}. "
                                    f"¿Generaste las máscaras (opción 3 del menú)?")
        augment = (split == "train") if augment is None else augment
        self.transforms = build_train_transforms(crop) if augment else build_eval_transforms()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        name = self.names[i]
        img = read_image_rgb(self.img_dir / f"{name}.jpg")                 # (H, W, 3) uint8
        mask = load_mask(self.mask_dir / f"{name}.npz")                     # (19, H, W) bool
        if mask.shape[0] != NUM_CLASSES:
            raise ValueError(f"{name}: la máscara tiene {mask.shape[0]} capas y core/classes.py tiene "
                             f"{NUM_CLASSES}. Vuelve a generar las máscaras (opción 3 del menú).")

        # Albumentations espera la máscara como (H, W, capas); la nuestra es (capas, H, W).
        out = self.transforms(image=img, mask=mask.transpose(1, 2, 0).astype(np.uint8))

        # PyTorch espera (canales, H, W): se reordenan los ejes de la foto y de la máscara.
        image = torch.from_numpy(np.ascontiguousarray(out["image"].transpose(2, 0, 1))).float()
        target = torch.from_numpy(np.ascontiguousarray(out["mask"].transpose(2, 0, 1))).float()
        return image, target, name

    def __repr__(self) -> str:
        return f"BridgeDamageDataset(split={self.split!r}, fotos={len(self)})"
