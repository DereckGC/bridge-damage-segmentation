"""Reglas de validación de las acciones de DATOS: convert, prepare y visualize."""
import os
from pathlib import Path

from app.validation.base import BaseValidator
from utils.constants import SPLITS
from utils.files import count_files, count_images, has_images

FORMATS = ("auto", "fiftyone", "supervisely")


class DataRules(BaseValidator):
    # ------------------------------------------------------------------ reglas por acción
    def _validate_convert(self, p: dict) -> dict:
        from data_prep.convert import detect_format   # misma regla que usa el convertidor
        src = Path(p["src"])
        if not src.is_dir():
            self._error(f"--src: no existe la carpeta '{src}'.")
        elif p["format"] == "auto" and detect_format(src) is None:
            self._error(f"--src: '{src}' no parece un dataset conocido (no tiene samples.json ni "
                        f"carpetas ann/). Revisa la ruta o indica el formato con --format.")
        self._choice(p["format"], "--format", FORMATS)
        self._in_range(p["test_frac"], "--test-frac", 0.0, 1.0, inclusive=False)
        self._int_at_least(p["seed"], "--seed", 0)
        return p

    def _validate_prepare(self, p: dict) -> dict:
        cfg = self._load_config(p["config"], ["paths.raw", "paths.processed", "data.long_side"])
        self._int_at_least(p["workers"], "--workers", 1, high=os.cpu_count() or 1)
        if p["limit"] is not None:
            self._int_at_least(p["limit"], "--limit", 1)
        if cfg:
            self._int_at_least(cfg["data"]["long_side"], "data.long_side (config)", 256, high=4096)
            raw = Path(cfg["paths"]["raw"])
            if not any(has_images(raw / "images" / s) for s in SPLITS):
                self._error(f"No hay imágenes en {raw / 'images'}/{{train,val,test}}. "
                            f"¿Ejecutaste antes la conversión (opción 2 del menú)?")
            elif not any((raw / "annotations" / s).is_dir() for s in SPLITS):
                self._error(f"No existe {raw / 'annotations'}/{{train,val,test}}: sin anotaciones "
                            f"no se pueden generar máscaras.")
        p["cfg"] = cfg
        return p

    def _validate_visualize(self, p: dict) -> dict:
        cfg = self._load_config(p["config"], ["paths.processed"])
        self._choice(p["split"], "--split", SPLITS)
        self._int_at_least(p["n"], "--n", 1)
        if cfg and p["split"] in SPLITS:
            base = Path(cfg["paths"]["processed"]) / p["split"]
            if not count_images(base / "images") or not count_files(base / "masks", ".npz"):
                self._error(f"No hay fotos con máscara en {base}. "
                            f"¿Generaste las máscaras (opción 3 del menú)?")
            elif p["name"]:
                for sub, ext in (("images", ".jpg"), ("masks", ".npz")):
                    if not (base / sub / f"{p['name']}{ext}").is_file():
                        self._error(f"--name: no existe {base / sub / (p['name'] + ext)} "
                                    f"(escribe el nombre sin extensión).")
            if p["augmented"]:
                self._int_at_least(cfg.get("data", {}).get("crop", 512), "data.crop (config)", 64, high=4096)
        p["cfg"] = cfg
        return p
