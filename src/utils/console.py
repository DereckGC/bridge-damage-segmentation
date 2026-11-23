"""Preguntar datos al usuario en la terminal, con valores por defecto y validación de tipo."""

YES = ("s", "si", "sí", "y", "yes")
NO = ("n", "no")


def ask_yes_no(question: str, default: bool) -> bool:
    """Pregunta sí/no. Enter = default. Repite hasta recibir una respuesta válida."""
    options = "S/n" if default else "s/N"
    while True:
        answer = input(f"{question} ({options}): ").strip().lower()
        if not answer:
            return default
        if answer in YES:
            return True
        if answer in NO:
            return False
        print("  Responde s (sí) o n (no).")


def ask_value(question: str, value_type=str, default=None, choices=None, optional: bool = False):
    """Pregunta un valor de tipo str/int/float. Enter = default (si hay). Repite si el valor no es válido.
    optional=True permite dejarlo vacío (devuelve None)."""
    hint = f" ({' / '.join(map(str, choices))})" if choices else ""
    while True:
        shown = f" [{default}]" if default not in (None, "") else ""
        answer = input(f"{question}{hint}{shown}: ").strip()
        if not answer:
            if default in (None, ""):
                if optional:
                    return None
                print("  Este dato es obligatorio.")
                continue
            return default
        try:
            return value_type(answer.replace(",", ".") if value_type is float else answer)
        except ValueError:
            kind = {int: "un número entero", float: "un número (p. ej. 0.5)"}.get(value_type, "un valor")
            print(f"  Debe ser {kind}.")


def format_value(value) -> str:
    """True/False -> sí/no; None -> (ninguno); lo demás, tal cual."""
    if value is None:
        return "(ninguno)"
    return "sí" if value is True else "no" if value is False else str(value)


def pause(message: str = "Presiona Enter para continuar...") -> None:
    input(f"\n{message}")
