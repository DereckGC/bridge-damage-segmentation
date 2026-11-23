"""Abrir, reducir y guardar fotos. Todas las imágenes se manejan como arreglos RGB (alto, ancho, 3)."""
import cv2
import numpy as np
from PIL import Image, ImageOps


def load_image(path, expected_size=None) -> np.ndarray:
    """Lee la imagen como RGB. Si su tamaño no coincide con `expected_size` (ancho, alto), prueba
    aplicando la rotación EXIF (fotos de celular giradas): las anotaciones se refieren a la imagen
    tal como la vio el anotador."""
    img = Image.open(path)
    if expected_size is not None and img.size != tuple(expected_size):
        rotated = ImageOps.exif_transpose(img)
        if rotated.size == tuple(expected_size):
            img = rotated
    return np.array(img.convert("RGB"))


def resize_long_side(img: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """Reduce la imagen para que su lado más largo mida `long_side`, sin deformarla.
    Devuelve (imagen_reducida, escala). La escala sirve para ajustar coordenadas (p. ej. polígonos)."""
    h0, w0 = img.shape[:2]
    scale = long_side / max(h0, w0)
    h, w = round(h0 * scale), round(w0 * scale)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), scale  # OpenCV recibe (ancho, alto)


def save_image(path, img: np.ndarray, quality: int = 95) -> None:
    """Guarda una imagen RGB como JPEG. OpenCV trabaja en BGR, por eso se invierte el orden de colores."""
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])


def read_image_rgb(path) -> np.ndarray:
    """Lectura rápida con OpenCV (sin revisar EXIF), devolviendo RGB. Para fotos ya procesadas."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
