"""Leer el archivo de configuración (YAML) y consultar sus claves."""
from pathlib import Path

import yaml

from utils.constants import DEFAULT_CONFIG, DEFAULT_PATHS


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Lee un YAML y lo devuelve como diccionario. Lanza OSError / yaml.YAMLError si falla."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: dict, dotted_key: str, default=None):
    """Lee una clave anidada con puntos: get_nested(cfg, "data.long_side") == cfg["data"]["long_side"].
    Devuelve `default` si alguna parte del camino no existe."""
    node = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def missing_keys(cfg: dict, required: list[str]) -> list[str]:
    """Claves (con puntos) que faltan en el config. Lista vacía = está todo."""
    return [k for k in required if get_nested(cfg, k) is None]


def project_paths(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Path]:
    """Rutas del proyecto (raw, processed, models...) leídas del config, con valores por defecto
    si el archivo no existe o no las define. Nunca falla: sirve para mostrar el estado."""
    paths = dict(DEFAULT_PATHS)
    try:
        paths.update(load_config(config_path).get("paths", {}) or {})
    except (OSError, yaml.YAMLError):
        pass
    return {k: Path(v) for k, v in paths.items()}
