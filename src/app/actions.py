"""Catálogo de acciones: el ÚNICO lugar donde se declara qué acciones existen y qué parámetros usan.

De aquí salen la ayuda de la terminal (-h) y las preguntas del menú interactivo.
Para agregar una acción: añade su entrada en ACTIONS, su método _run_<accion> en el orquestador
y su regla _validate_<accion> en app/validation/.
"""
import os

from utils.constants import DEFAULT_CONFIG, SPLITS

# ----------------------------------------------------------------------------------------------
# Catálogo de acciones: el ÚNICO lugar donde se declaran las acciones y sus parámetros.
# De aquí salen la ayuda de la terminal (-h) y las preguntas del menú interactivo.
#   name     nombre interno del parámetro        flag     cómo se escribe en la terminal
#   type     str | int | float | bool            default  valor si el usuario no indica nada
#   help     explicación corta (para -h)         prompt   pregunta que muestra el menú
#   choices  (opcional) valores permitidos       required (opcional) obligatorio en modo comando
# ----------------------------------------------------------------------------------------------
ACTIONS = {
    "convert": {
        "help": "Convierte un dataset descargado (FiftyOne, Supervisely) a data/raw.",
        "args": [
            dict(name="src", flag="--src", type=str, default=None, required=True,
                 help="carpeta descargada, p. ej. dataset/voxel51",
                 prompt="Carpeta del dataset descargado"),
            dict(name="format", flag="--format", type=str, default="auto",
                 choices=["auto", "fiftyone", "supervisely"],
                 help="formato del dataset (auto = detectarlo solo)",
                 prompt="Formato del dataset (auto lo detecta solo)"),
            dict(name="raw", flag="--raw", type=str, default="data/raw",
                 help="destino de imágenes y anotaciones",
                 prompt="Carpeta destino de imágenes y anotaciones"),
            dict(name="inputs", flag="--inputs", type=str, default="inputs/sin_anotar",
                 help="destino de las fotos sin anotar",
                 prompt="Carpeta para las fotos sin anotaciones"),
            dict(name="test_frac", flag="--test-frac", type=float, default=0.5,
                 help="fracción de val que pasa a test",
                 prompt="Fracción de validación que se usará como test (entre 0 y 1)"),
            dict(name="seed", flag="--seed", type=int, default=42,
                 help="semilla de la división val/test",
                 prompt="Semilla de la división val/test"),
            dict(name="copy", flag="--copy", type=bool, default=False,
                 help="copiar en vez de mover las imágenes",
                 prompt="¿Copiar las imágenes en vez de moverlas? (así la descarga queda intacta)"),
            dict(name="inspect", flag="--inspect", type=bool, default=False,
                 help="solo mostrar el contenido, sin convertir",
                 prompt="¿Solo revisar, sin convertir?"),
        ],
    },
    "prepare": {
        "help": "Genera data/processed: fotos reducidas + máscaras (19, alto, ancho).",
        "args": [
            dict(name="config", flag="--config", type=str, default=DEFAULT_CONFIG,
                 help="archivo de configuración",
                 prompt="Archivo de configuración"),
            dict(name="workers", flag="--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                 help="procesos en paralelo",
                 prompt="Procesos en paralelo (más = más rápido, pero usa más CPU)"),
            dict(name="limit", flag="--limit", type=int, default=None,
                 help="procesar solo las N primeras fotos de cada grupo (para pruebas)",
                 prompt="Procesar solo N fotos por grupo (Enter = todas)"),
        ],
    },
    "visualize": {
        "help": "Dibuja las máscaras sobre sus fotos para revisarlas a ojo (outputs/check_masks).",
        "args": [
            dict(name="config", flag="--config", type=str, default=DEFAULT_CONFIG,
                 help="archivo de configuración",
                 prompt="Archivo de configuración"),
            dict(name="split", flag="--split", type=str, default="train", choices=list(SPLITS),
                 help="grupo a revisar: train | val | test",
                 prompt="Grupo a revisar"),
            dict(name="n", flag="--n", type=int, default=10,
                 help="cuántas fotos al azar",
                 prompt="Cuántas fotos revisar"),
            dict(name="name", flag="--name", type=str, default=None,
                 help="una foto concreta (nombre sin extensión)",
                 prompt="Una foto concreta (nombre sin extensión; Enter = al azar)"),
            dict(name="augmented", flag="--augmented", type=bool, default=False,
                 help="mostrar 3 aumentaciones de entrenamiento por foto",
                 prompt="¿Ver también cómo quedan con las aumentaciones del entrenamiento?"),
        ],
    },
    "train": {
        "help": "Entrena el modelo con data/processed y guarda models/best.pt.",
        "args": [
            dict(name="config", flag="--config", type=str, default=DEFAULT_CONFIG,
                 help="archivo de configuración",
                 prompt="Archivo de configuración"),
            dict(name="epochs", flag="--epochs", type=int, default=None,
                 help="número de épocas (por defecto, train.epochs del config)",
                 prompt="Número de épocas (Enter = las del config)"),
            dict(name="resume", flag="--resume", type=bool, default=False,
                 help="retomar desde models/last.pt",
                 prompt="¿Retomar un entrenamiento interrumpido (desde models/last.pt)?"),
            dict(name="allow_cpu", flag="--allow-cpu", type=bool, default=False,
                 help="permitir entrenar sin GPU (solo para pruebas: es muy lento)",
                 prompt="¿Permitir entrenar sin GPU? (solo para pruebas, es muy lento)"),
        ],
    },
}


def arg_spec(action: str, name: str) -> dict:
    return next(a for a in ACTIONS[action]["args"] if a["name"] == name)


def default_params(action: str) -> dict:
    """Todos los parámetros de una acción con su valor por defecto."""
    return {a["name"]: a["default"] for a in ACTIONS[action]["args"]}

