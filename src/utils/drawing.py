"""Dibujar máscaras sobre fotos (para revisar datos y, más adelante, mostrar predicciones)."""
import cv2
import numpy as np


def overlay_mask(img: np.ndarray, mask: np.ndarray, colors, class_names, alpha: float = 0.5) -> np.ndarray:
    """Pinta cada capa de la máscara con el color de su clase, con su contorno y una leyenda.

    img:   foto RGB (H, W, 3) uint8
    mask:  máscara (C, H, W) bool, del mismo alto y ancho que la foto
    Devuelve una copia de la foto con los daños coloreados.
    """
    out = img.astype(np.float32)
    present = []
    for c in range(mask.shape[0]):
        m = mask[c]
        if not m.any():
            continue
        present.append(c)
        color = np.array(colors[c], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * color             # mezcla solo los píxeles de la clase
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, colors[c], 2)
    out = out.clip(0, 255).astype(np.uint8)
    draw_legend(out, [(class_names[c], colors[c]) for c in present])
    return out


def draw_legend(img: np.ndarray, items: list[tuple[str, tuple]], start_y: int = 28) -> None:
    """Escribe una leyenda (cuadrito de color + nombre) arriba a la izquierda. Modifica `img`."""
    y = start_y
    for name, color in items:
        cv2.rectangle(img, (10, y - 16), (30, y + 4), color, -1)
        cv2.putText(img, name, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, name, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def side_by_side(*images: np.ndarray, gap: int = 8) -> np.ndarray:
    """Une imágenes del mismo alto en una fila, separadas por una franja blanca."""
    h = images[0].shape[0]
    sep = np.full((h, gap, 3), 255, dtype=np.uint8)
    parts = []
    for i, im in enumerate(images):
        parts.extend([sep, im] if i else [im])
    return np.concatenate(parts, axis=1)
