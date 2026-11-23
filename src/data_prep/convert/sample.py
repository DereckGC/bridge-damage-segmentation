"""Estructura común: una foto con sus anotaciones, ya independiente del formato de origen.

Cada lector (readers/) transforma su formato en una lista de Sample, y el convertidor
(converter.py) solo trabaja con Sample. Así, agregar un formato no toca el resto del código.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    """Una foto con sus anotaciones, ya independiente del formato de origen."""
    image: Path                       # ruta de la imagen en la descarga
    split: str                        # "train", "val", "test"... tal como venga del origen (normalizado)
    labeled: bool                     # False si la foto no tiene anotaciones (p. ej. testdev)
    width: int | None = None          # tamaño con el que se anotó (sirve para detectar rotaciones EXIF)
    height: int | None = None
    shapes: list[tuple[str, list[list[float]]]] = field(default_factory=list)  # (etiqueta, puntos en píxeles)


