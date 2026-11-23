"""Menú interactivo: se abre al ejecutar `python src/main.py` sin argumentos.

Muestra el estado del proyecto, sugiere el siguiente paso, describe cada tarea en lenguaje
sencillo y pide los datos uno por uno (Enter = valor por defecto). Al final ejecuta la tarea
con Orchestrator.execute(), exactamente igual que el modo comando, validación incluida.

Las tareas están en app/tasks.py y el cálculo del estado en app/status.py.
"""
from app.actions import arg_spec, default_params
from app.orchestrator import EXIT_OK
from app.status import find_downloaded_datasets, project_status
from app.tasks import TASKS
from utils.console import ask_value, ask_yes_no, format_value, pause

LINE = "=" * 64


class InteractiveMenu:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    def loop(self) -> int:
        try:
            while True:
                self._show_header()
                choice = input("Elige una opción: ").strip()
                if choice in ("0", "q", "salir"):
                    print("Hasta luego.")
                    return EXIT_OK
                if not choice.isdigit() or not 1 <= int(choice) <= len(TASKS):
                    print(f"\n'{choice}' no es una opción válida.")
                    pause("Presiona Enter para volver al menú...")
                    continue
                self._run_task(TASKS[int(choice) - 1])
                pause("Presiona Enter para volver al menú...")
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            return EXIT_OK

    # ------------------------------------------------------------------ pantallas
    def _show_header(self) -> None:
        lines, hint = project_status()
        print(f"\n{LINE}\n  Detección de daños en puentes (UPerNet)\n{LINE}")
        print("Estado del proyecto:   [x] completo   [~] incompleto   [ ] pendiente")
        for mark, title, detail in lines:
            print(f"  [{mark}] {title:<20} {detail}")
        print(f"\nSugerencia: {hint}\n")
        print("¿Qué quieres hacer?")
        for i, task in enumerate(TASKS, start=1):
            print(f"  {i}) {task['title']}")
        print("  0) Salir\n")

    def _run_task(self, task: dict) -> None:
        action = task["action"]
        print(f"\n--- {task['title']} ---")
        for line in task["description"]:
            print(f"  {line}")
        print("\n(Enter = usar el valor entre corchetes)\n")

        params = default_params(action) | task.get("defaults", {}) | task["fixed"]
        if action == "convert":
            found = find_downloaded_datasets()
            if found:
                params["src"] = str(found[0])

        for name in task["ask"]:
            params[name] = self._ask(action, name, params[name])
        if task["advanced"] and ask_yes_no("¿Quieres cambiar opciones avanzadas?", False):
            for name in task["advanced"]:
                params[name] = self._ask(action, name, params[name])

        print(f"\nSe va a ejecutar: {task['title']}")
        if task.get("note"):
            print(f"  ({task['note']})")
        for name in task["ask"] + task["advanced"]:
            print(f"  - {arg_spec(action, name)['prompt']}: {format_value(params[name])}")
        if not ask_yes_no("¿Continuar?", True):
            print("Cancelado.")
            return
        code = self.orch.execute(action, params, label=task["title"])
        if code != EXIT_OK:
            print("\nNo se completó. Revisa los mensajes de arriba y vuelve a intentarlo.")

    @staticmethod
    def _ask(action: str, name: str, current):
        """Pregunta un parámetro usando su descripción en el catálogo ACTIONS del orquestador."""
        spec = arg_spec(action, name)
        if spec["type"] is bool:
            return ask_yes_no(spec["prompt"], bool(current))
        return ask_value(spec["prompt"], spec["type"], current, spec.get("choices"),
                         optional=not spec.get("required", False) and spec["default"] is None)
