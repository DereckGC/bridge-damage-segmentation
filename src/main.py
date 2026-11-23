"""Punto de entrada del proyecto. Ejecútalo SIEMPRE desde la raíz del proyecto
(las rutas como config/config.yaml, data/raw y dataset/ son relativas a la raíz):

    python src/main.py                                              # menú interactivo (recomendado)
    python src/main.py -h                                           # lista de acciones
    python src/main.py <accion> -h                                  # parámetros de una acción
    python src/main.py convert --src dataset/voxel51 --inspect
    python src/main.py convert --src dataset/voxel51 --copy
    python src/main.py prepare
    python src/main.py visualize --split train --n 10
    python src/main.py train [--epochs N] [--resume]
"""
import sys
from pathlib import Path

# Los módulos se importan entre sí por nombre (from classes import ...), así que la carpeta
# donde viven debe estar en el path. Funciona tanto si este archivo está en src/ como en la raíz.
# Va FUERA del `if __name__ == "__main__"` a propósito: en Windows, los procesos en paralelo de
# prepare_data vuelven a importar este archivo y también necesitan encontrar los módulos.
_here = Path(__file__).resolve().parent
_src = _here if (_here / "app").is_dir() else _here / "src"
sys.path.insert(0, str(_src))

from app.orchestrator import Orchestrator  # noqa: E402

if __name__ == "__main__":
    sys.exit(Orchestrator().run(sys.argv[1:]))
