"""Listar y contar archivos e imágenes en carpetas."""
from pathlib import Path

from utils.constants import IMG_EXT


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMG_EXT


def list_images(folder: str | Path) -> list[str]:
    """Nombres (no rutas) de las imágenes de una carpeta, ordenados. Lista vacía si no existe."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(f.name for f in folder.iterdir() if f.is_file() and is_image(f))


def count_images(folder: str | Path) -> int:
    return len(list_images(folder))


def has_images(folder: str | Path) -> bool:
    folder = Path(folder)
    return folder.is_dir() and any(is_image(f) for f in folder.iterdir())


def count_files(folder: str | Path, ext: str) -> int:
    """Cuántos archivos con esa extensión hay en la carpeta (p. ej. ext=".npz")."""
    folder = Path(folder)
    return sum(1 for _ in folder.glob(f"*{ext}")) if folder.is_dir() else 0


def ensure_dirs(*folders: str | Path) -> None:
    """Crea las carpetas (y sus padres) si no existen."""
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
