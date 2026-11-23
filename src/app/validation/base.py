"""Base de la validación: el mecanismo general y las reglas reutilizables.

Es mejor fallar en 1 segundo con un mensaje claro ("no existe data/raw/images") que fallar
a los 20 minutos de un preprocesado con un error críptico.

Cada acción tiene su método `_validate_<accion>`. Los métodos NO se detienen en el primer error:
lo acumulan con `self._error(...)`, de modo que el usuario ve TODOS los problemas juntos.
Al final, `validate()` lanza ValidationError si hubo alguno, o devuelve los parámetros limpios
(p. ej. con la config ya cargada) listos para pasárselos a la acción.
"""
import os

import yaml

from utils.config import load_config, missing_keys


class ValidationError(Exception):
    """Uno o más parámetros no son válidos. `errors` es la lista de mensajes."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))



class BaseValidator:
    """Mecanismo común. Las reglas de cada acción están en data_rules.py y training_rules.py."""

    def __init__(self):
        self._errors: list[str] = []

    # ------------------------------------------------------------------ API pública
    def validate(self, action: str, params: dict) -> dict:
        """Valida los parámetros de `action`. Devuelve un dict limpio o lanza ValidationError."""
        method = getattr(self, f"_validate_{action}", None)
        if method is None:
            raise ValidationError([f"No hay reglas de validación para la acción '{action}'."])
        self._errors = []
        clean = method(dict(params))
        if self._errors:
            raise ValidationError(self._errors)
        return clean


    # ------------------------------------------------------------------ reglas reutilizables
    def _error(self, msg: str) -> None:
        self._errors.append(msg)

    def _load_config(self, path: str, required_keys: list[str]) -> dict | None:
        """Carga el YAML y comprueba que existan las claves anidadas, p. ej. 'paths.raw'."""
        if not os.path.isfile(path):
            self._error(f"--config: no existe el archivo '{path}'.")
            return None
        try:
            cfg = load_config(path)
        except yaml.YAMLError as e:
            self._error(f"--config: '{path}' no es un YAML válido ({e}).")
            return None
        missing = missing_keys(cfg, required_keys)
        for k in missing:
            self._error(f"Falta la clave '{k}' en {path}.")
        return None if missing else cfg

    def _choice(self, value, label: str, options) -> None:
        if value not in options:
            self._error(f"{label}: '{value}' no es válido. Opciones: {', '.join(options)}.")

    def _int_at_least(self, value, label: str, low: int, high: int | None = None) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            self._error(f"{label}: debe ser un número entero (recibido: {value!r}).")
        elif value < low or (high is not None and value > high):
            rango = f"entre {low} y {high}" if high is not None else f">= {low}"
            self._error(f"{label}: debe ser {rango} (recibido: {value}).")

    def _in_range(self, value, label: str, low: float, high: float, inclusive: bool = True) -> None:
        ok = (low <= value <= high) if inclusive else (low < value < high)
        if not ok:
            signo = "≤" if inclusive else "<"
            self._error(f"{label}: debe cumplir {low} {signo} valor {signo} {high} (recibido: {value}).")

