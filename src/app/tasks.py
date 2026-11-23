"""Tareas del menú interactivo, descritas para el usuario.

Cada TAREA usa una ACCIÓN del orquestador (app/actions.py) con algunos parámetros ya fijados.
Por ejemplo, "Revisar un dataset" y "Convertir un dataset" usan la misma acción 'convert',
la primera con inspect=True. Así el menú habla en términos de lo que el usuario quiere hacer.
Para que una acción nueva aparezca en el menú, agrégala aquí.
"""

# ----------------------------------------------------------------------------------------------
# Tareas del menú. Para cada una:
#   action    acción del orquestador que ejecuta        fixed     parámetros ya decididos (no se preguntan)
#   ask       parámetros que siempre se preguntan        advanced  se preguntan solo si el usuario quiere
#   defaults  (opcional) valores sugeridos distintos de los del modo comando
#   note      (opcional) aclaración que se muestra en el resumen antes de confirmar
# ----------------------------------------------------------------------------------------------
TASKS = [
    {
        "title": "Revisar un dataset descargado",
        "description": [
            "Muestra qué contiene un dataset descargado: cuántas fotos hay en cada grupo",
            "(train, val, testdev), cuántas faltan por descargar y si cada etiqueta",
            "corresponde a una clase del proyecto.",
            "No copia ni modifica nada. Úsalo antes de convertir.",
        ],
        "action": "convert",
        "fixed": {"inspect": True, "copy": False},
        "note": "Solo lectura: no copia ni modifica nada.",
        "ask": ["src"],
        "advanced": ["format"],
    },
    {
        "title": "Convertir un dataset al formato del proyecto",
        "description": [
            "Copia las fotos del dataset descargado a data/raw/images/<grupo> y escribe",
            "sus polígonos en data/raw/annotations/<grupo>, en el formato que entiende",
            "el proyecto. Si el dataset no trae un test anotado, separa la mitad de",
            "validación como test. Las fotos sin anotar van a inputs/sin_anotar.",
        ],
        "action": "convert",
        "fixed": {"inspect": False},
        "defaults": {"copy": True},  # en el menú, copiar es lo seguro: la descarga queda intacta
        "ask": ["src", "copy"],
        "advanced": ["format", "raw", "inputs", "test_frac", "seed"],
    },
    {
        "title": "Generar las máscaras (preparar datos)",
        "description": [
            "Reduce cada foto de data/raw (lado largo según la config) y convierte sus",
            "polígonos en una máscara de 19 capas. Guarda ambas en data/processed,",
            "que es lo que usará el entrenamiento. Al final muestra cuántas fotos",
            "procesó, los errores y el desbalance de clases. Puede tardar varios minutos.",
        ],
        "action": "prepare",
        "fixed": {},
        "ask": [],
        "advanced": ["limit", "config", "workers"],
    },
    {
        "title": "Revisar las máscaras (a ojo)",
        "description": [
            "Dibuja la máscara encima de su foto (foto | foto con los daños coloreados)",
            "y guarda las imágenes en outputs/check_masks/<grupo>. Ábrelas y comprueba",
            "que cada color cae sobre el daño que corresponde, antes de entrenar.",
            "Opcional: ver cómo quedan con las aumentaciones del entrenamiento.",
        ],
        "action": "visualize",
        "fixed": {},
        "note": "Solo lectura: no modifica los datos, solo crea imágenes de revisión.",
        "ask": ["split", "n", "augmented"],
        "advanced": ["name", "config"],
    },
    {
        "title": "Entrenar el modelo",
        "description": [
            "Entrena UPerNet con las fotos y máscaras de data/processed. En cada época:",
            "aprende con train, se mide con val (IoU) y guarda models/best.pt si mejoró.",
            "Se detiene solo cuando val deja de mejorar. Puede tardar varias horas: puedes",
            "interrumpirlo con Ctrl+C y retomarlo después con la opción de retomar.",
        ],
        "action": "train",
        "fixed": {},
        "note": "Usa la GPU. Los ajustes (épocas, lote, encoder...) están en config/config.yaml.",
        "ask": ["resume"],
        "advanced": ["epochs", "config", "allow_cpu"],
    },
]
