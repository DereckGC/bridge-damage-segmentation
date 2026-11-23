"""Valores fijos compartidos por todo el proyecto. Cambiarlos aquí los cambia en todas partes."""

# Grupos de datos con los que trabaja el proyecto (carpetas de data/raw y data/processed).
SPLITS = ("train", "val", "test")

# Extensiones de imagen aceptadas (se comparan en minúsculas).
IMG_EXT = (".jpg", ".jpeg", ".png")

# Archivo de configuración por defecto (ruta relativa a la raíz del proyecto).
DEFAULT_CONFIG = "config/config.yaml"

# Rutas por defecto si el config no las define.
DEFAULT_PATHS = {
    "raw": "data/raw",
    "processed": "data/processed",
    "models": "models",
    "inputs": "inputs",
    "outputs": "outputs",
    "downloads": "dataset",   # donde se guardan los datasets descargados (sin convertir)
}
