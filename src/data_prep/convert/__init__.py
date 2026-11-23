"""Conversión de datasets descargados al formato del proyecto.

    sample.py      Sample: la estructura común (foto + polígonos en píxeles)
    readers/       un lector por formato de origen (FiftyOne, Supervisely...)
    converter.py   reparte en grupos y escribe data/raw
"""
from data_prep.convert.converter import run
from data_prep.convert.readers import READERS, detect_format

__all__ = ["run", "detect_format", "READERS"]
