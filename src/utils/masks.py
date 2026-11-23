"""Máscaras multi-etiqueta: una matriz (clases, alto, ancho) de True/False, una capa por clase."""
import cv2
import numpy as np


def polygons_to_mask(shapes, height: int, width: int, scale: float,
                     class_to_idx: dict[str, int], num_classes: int):
    """Pinta cada polígono en la capa de su clase.

    shapes: lista de {"label": str, "points": [[x, y], ...]} en píxeles de la imagen ORIGINAL.
    scale:  factor para llevar esos puntos al tamaño (height, width) de la máscara.
    Devuelve (mascara bool (num_classes, height, width), set de etiquetas desconocidas).
    """
    mask = np.zeros((num_classes, height, width), dtype=np.uint8)   # OpenCV no dibuja sobre bool
    unknown = set()
    for shape in shapes:
        label = shape["label"]
        if label not in class_to_idx:
            unknown.add(label)
            continue
        pts = np.asarray(shape["points"], dtype=np.float64) * scale
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)       # formato que exige fillPoly
        if len(pts) < 3:                                              # un polígono necesita 3 vértices
            continue
        # Polígonos de la misma clase que se solapan simplemente se unen.
        cv2.fillPoly(mask[class_to_idx[label]], [pts], 1)
    return mask.astype(bool), unknown


def save_mask(path, mask: np.ndarray) -> None:
    """Guarda una máscara (C, H, W) bool comprimida: packbits mete 8 píxeles en 1 byte."""
    c, h, w = mask.shape
    np.savez_compressed(path, bits=np.packbits(mask, axis=-1), shape=np.array([c, h, w]))


def load_mask(path) -> np.ndarray:
    """Inverso de save_mask. Devuelve (C, H, W) bool."""
    data = np.load(path)
    c, h, w = data["shape"].tolist()
    return np.unpackbits(data["bits"], axis=-1, count=w).astype(bool).reshape(c, h, w)


def pixels_per_class(mask: np.ndarray) -> np.ndarray:
    """Cuántos píxeles tiene cada clase: vector de largo C."""
    return mask.sum(axis=(1, 2))
