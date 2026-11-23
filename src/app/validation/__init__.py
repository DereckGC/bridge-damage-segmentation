"""Validación de parámetros antes de ejecutar cualquier acción.

    base.py             mecanismo común (acumular errores, cargar config, reglas de rango)
    data_rules.py       reglas de convert, prepare y visualize
    training_rules.py   reglas de train

ParameterValidator reúne todas las reglas. Para una acción nueva, agrega su _validate_<accion>
en el archivo de reglas que corresponda (o en uno nuevo, y súmalo aquí).
"""
from app.validation.base import ValidationError
from app.validation.data_rules import DataRules
from app.validation.training_rules import TrainingRules


class ParameterValidator(DataRules, TrainingRules):
    """Valida cualquier acción: busca el método _validate_<accion> en las reglas de arriba."""


__all__ = ["ParameterValidator", "ValidationError"]
