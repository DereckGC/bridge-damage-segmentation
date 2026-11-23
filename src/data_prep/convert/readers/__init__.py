"""Lectores de formatos de dataset. Cada módulo de esta carpeta sabe leer UN formato y tiene:

    matches(src) -> bool          ¿la carpeta tiene este formato?
    read_<formato>(src) -> list[Sample]

Para soportar un formato nuevo (COCO, Label Studio...): crea readers/<formato>.py con esas dos
funciones y regístralo en READERS. Nada más cambia.
"""
from pathlib import Path

from data_prep.convert.readers import fiftyone, supervisely

# Orden de detección: se usa el primer formato cuyo matches() sea verdadero.
READERS = {
    "fiftyone": fiftyone,
    "supervisely": supervisely,
}


def detect_format(src: Path) -> str | None:
    """Nombre del formato de la carpeta, o None si no se reconoce."""
    src = Path(src)
    return next((name for name, module in READERS.items() if module.matches(src)), None)


def read_samples(src: Path, fmt: str):
    """Lee la carpeta con el lector del formato indicado."""
    module = READERS[fmt]
    return getattr(module, f"read_{fmt}")(Path(src))
