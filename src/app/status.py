"""Estado del proyecto que muestra el menú: qué pasos están hechos y cuál sigue.
Solo lee carpetas y archivos (no carga PyTorch), para que el menú abra rápido."""
from pathlib import Path

from utils.config import project_paths
from utils.constants import SPLITS
from utils.files import count_files, count_images

DONE, PARTIAL, TODO = "x", "~", " "   # marcas del estado: completo, incompleto, pendiente


def find_downloaded_datasets(root: Path | None = None) -> list[Path]:
    """Carpetas dentro de la carpeta de descargas (paths.downloads) con un formato de dataset reconocido."""
    root = root or project_paths()["downloads"]
    from data_prep.convert import detect_format   # misma regla que usa el convertidor
    if not root.is_dir():
        return []
    return [d for d in sorted(p for p in root.iterdir() if p.is_dir()) if detect_format(d)]


def training_summary(models_dir: Path) -> str:
    """Resumen del entrenamiento leyendo models/log.csv (sin cargar PyTorch, para que sea rápido)."""
    import csv
    log = models_dir / "log.csv"
    try:
        with open(log, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("val_mIoU")]
    except OSError:
        rows = []
    if not rows:
        return "models/best.pt" if (models_dir / "best.pt").is_file() else "todavía no"
    best = max(rows, key=lambda r: float(r["val_mIoU"]))
    return f"mejor val mIoU {float(best['val_mIoU']):.3f} (época {best['epoch']} de {rows[-1]['epoch']} entrenadas)"


def project_status() -> tuple[list[tuple[str, str, str]], str]:
    """Devuelve las líneas de estado [(marca, título, detalle)] y una sugerencia de siguiente paso."""
    paths = project_paths()
    raw, processed = paths["raw"], paths["processed"]
    datasets = find_downloaded_datasets()
    raw_counts = {s: count_images(raw / "images" / s) for s in SPLITS}
    mask_counts = {s: count_files(processed / s / "masks", ".npz") for s in SPLITS}
    has_model = (paths["models"] / "best.pt").is_file()
    has_last = (paths["models"] / "last.pt").is_file()
    total_raw, total_masks = sum(raw_counts.values()), sum(mask_counts.values())

    # Máscaras: completas solo si cada grupo tiene tantas máscaras como fotos en data/raw.
    masks_complete = total_raw > 0 and all(mask_counts[s] >= raw_counts[s] for s in SPLITS)
    if masks_complete:
        mask_mark, mask_detail = DONE, " | ".join(f"{s} {n}" for s, n in mask_counts.items())
    elif total_masks:
        mask_mark = PARTIAL
        mask_detail = "INCOMPLETAS: " + " | ".join(f"{s} {mask_counts[s]} de {raw_counts[s]}" for s in SPLITS)
    else:
        mask_mark, mask_detail = TODO, "ninguna todavía"

    lines = [
        (DONE if datasets else TODO, "Dataset descargado",
         ", ".join(str(d) for d in datasets) if datasets else f"no hay nada en {paths['downloads']}/"),
        (DONE if total_raw else TODO, "Datos convertidos",
         " | ".join(f"{s} {n}" for s, n in raw_counts.items())),
        (mask_mark, "Máscaras generadas", mask_detail),
        (DONE if has_model else (PARTIAL if has_last else TODO), "Modelo entrenado", training_summary(paths["models"])),
    ]
    if not datasets and not total_raw:
        hint = f"Descarga un dataset en {paths['downloads']}/ (ver README)."
    elif not total_raw:
        hint = "Opción 1 para revisar el dataset y luego la 2 para convertirlo."
    elif not masks_complete:
        hint = ("Las máscaras no están completas (¿se interrumpió la generación?). "
                "Opción 3 para generarlas." if total_masks else "Opción 3 para generar las máscaras.")
    elif not has_model:
        hint = ("Hay un entrenamiento sin terminar: opción 5 y responde sí a retomar." if has_last else
                "Opción 4 para revisar las máscaras a ojo y luego la 5 para entrenar.")
    else:
        hint = ("Modelo entrenado (si se interrumpió, opción 5 y retomar). "
                "Siguiente: detectar daños en fotos nuevas (próximamente).")
    return lines, hint
