"""Traducir nombres de etiquetas de cualquier dataset a las clases del proyecto (core/classes.py)."""
import re

from core.classes import CLASSES

# Nombres alternativos con los que una clase puede venir escrita en otros datasets.
# Se comparan normalizados (minúsculas, sin espacios, guiones, barras ni "_").
# Si un dataset trae una etiqueta "SIN EQUIVALENTE", agrégala aquí.
ALIASES = {
    "alligatorcrack": "ACrack",
    "expansionjoint": "EJoint",
    "protectiveequipment": "PEquipment",
    "jointtape": "JTape",
    "washoutsconcretecorrosion": "WConccor",
    "exposedrebar": "ExposedRebars",
}


def normalize_name(name: str) -> str:
    """'Alligator Crack' -> 'alligatorcrack'."""
    return re.sub(r"[\s_\-/]+", "", name).lower()


_NAME_TO_CLASS = {normalize_name(c): c for c in CLASSES} | ALIASES


def resolve_label(name: str) -> str | None:
    """Clase de core/classes.py que corresponde a `name`, o None si no tiene equivalente."""
    return _NAME_TO_CLASS.get(normalize_name(name))


def normalize_split(name: str) -> str:
    """Nombres de grupo de otros datasets -> los del proyecto ('validation' -> 'val')."""
    name = name.lower()
    return {"validation": "val", "valid": "val"}.get(name, name)
