"""Orquestador: un único punto de entrada para todas las acciones del proyecto.

Dos formas de usarlo:
    python src/main.py                  -> menú interactivo (menu.py): guía paso a paso
    python src/main.py <accion> [...]    -> modo comando, p. ej. python src/main.py prepare --workers 4

Las dos terminan en Orchestrator.execute(accion, parametros), que hace:
    1. VALIDAR    ParameterValidator revisa que todo tenga sentido (rutas, rangos, config).
    2. DESPACHAR  se llama al método _run_<accion>, que a su vez llama al run() del módulo.

Para agregar una acción nueva (p. ej. 'train'):
    a) añadir su entrada en app/actions.py (descripción + parámetros)
    b) escribir el método Orchestrator._run_train(...)
    c) escribir la regla _validate_train(...) en app/validation/
    d) (opcional) añadir una tarea en app/tasks.py para que aparezca en el menú
"""
import argparse
import sys
import time
import traceback

from app.actions import ACTIONS
from app.validation import ParameterValidator, ValidationError

# Códigos de salida: 0 = bien, 1 = parámetros inválidos, 2 = falló la ejecución, 130 = Ctrl+C
EXIT_OK, EXIT_INVALID, EXIT_FAILED, EXIT_INTERRUPTED = 0, 1, 2, 130


class Orchestrator:
    def __init__(self, validator: ParameterValidator | None = None):
        self.validator = validator or ParameterValidator()
        self.parser = self._build_parser()

    # ------------------------------------------------------------------ API pública
    def run(self, argv: list[str]) -> int:
        """Sin argumentos abre el menú; con argumentos funciona en modo comando."""
        if not argv:
            from app.menu import InteractiveMenu
            return InteractiveMenu(self).loop()

        args = self.parser.parse_args(argv)
        if args.action is None:
            self.parser.print_help()
            return EXIT_INVALID
        params = vars(args)
        action = params.pop("action")
        code = self.execute(action, params)
        if code == EXIT_INVALID:
            print(f"\nAyuda: {self.parser.prog} {action} -h")
        return code

    def execute(self, action: str, params: dict, label: str | None = None) -> int:
        """Valida y ejecuta una acción. Lo usan tanto el modo comando como el menú.
        `label` es el nombre que se muestra al usuario (el menú pasa el título de la tarea)."""
        try:
            clean = self.validator.validate(action, params)
        except ValidationError as e:
            print(f"\n[ERROR] Parámetros no válidos para '{action}':")
            for msg in e.errors:
                print(f"  - {msg}")
            return EXIT_INVALID

        handler = getattr(self, f"_run_{action}")
        name = label or action
        print(f"\n==> {name}")
        start = time.time()
        try:
            handler(**clean)
        except KeyboardInterrupt:
            print("\n[INTERRUMPIDO] Cancelado por el usuario.")
            return EXIT_INTERRUPTED
        except Exception:
            print(f"\n[FALLO] '{name}' terminó con un error inesperado:\n")
            traceback.print_exc()
            return EXIT_FAILED
        print(f"==> {name}: terminado en {time.time() - start:.1f} s")
        return EXIT_OK

    # ------------------------------------------------------------------ construcción del CLI
    def _build_parser(self) -> argparse.ArgumentParser:
        # El nombre del comando se toma de cómo se ejecutó (p. ej. "python src/main.py").
        script = sys.argv[0].replace("\\", "/") if sys.argv and sys.argv[0] else "main.py"
        parser = argparse.ArgumentParser(
            prog=f"python {script}",
            description="Detección de daños en puentes (UPerNet). "
                        "Sin argumentos se abre el menú interactivo.",
        )
        sub = parser.add_subparsers(dest="action", metavar="<accion>")
        for name, spec in ACTIONS.items():
            p = sub.add_parser(name, help=spec["help"], description=spec["help"])
            for a in spec["args"]:
                if a["type"] is bool:
                    p.add_argument(a["flag"], dest=a["name"], action="store_true", help=a["help"])
                else:
                    p.add_argument(a["flag"], dest=a["name"], type=a["type"], default=a["default"],
                                   required=a.get("required", False), help=a["help"])
        return parser

    # ------------------------------------------------------------------ acciones
    # Los imports van DENTRO de cada método: cada acción carga solo lo que necesita
    # (más adelante, solo 'train' y 'predict' cargarán PyTorch).

    def _run_convert(self, src, format, raw, inputs, test_frac, seed, copy, inspect):
        from data_prep import convert
        convert.run(src=src, fmt=format, raw=raw, inputs=inputs, test_frac=test_frac,
                            seed=seed, copy=copy, inspect_only=inspect)

    def _run_prepare(self, cfg, workers, config):
        from data_prep import prepare
        print(f"Config: {config} | workers: {workers}")
        prepare.run(cfg, workers)

    def _run_visualize(self, cfg, split, n, name, augmented, config):
        from data_prep import visualize
        visualize.run(cfg, split=split, n=n, name=name, augmented=augmented)

    def _run_train(self, cfg, config, epochs, resume, allow_cpu):
        from training import trainer
        print(f"Config: {config}")
        trainer.run(cfg, epochs=epochs, resume=resume, allow_cpu=allow_cpu)

