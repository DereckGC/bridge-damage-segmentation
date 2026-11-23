"""Reglas de validación del ENTRENAMIENTO: config, máscaras completas, PyTorch y GPU."""
from pathlib import Path

from app.validation.base import BaseValidator
from utils.files import count_files, count_images


class TrainingRules(BaseValidator):
    def _validate_train(self, p: dict) -> dict:
        cfg = self._load_config(p["config"], [
            "paths.processed", "paths.models", "data.crop", "model.encoder", "model.pretrained",
            "train.epochs", "train.batch_size", "train.lr"])
        if p["epochs"] is not None:
            self._int_at_least(p["epochs"], "--epochs", 1)
        if cfg:
            t = cfg["train"]
            self._int_at_least(t["epochs"], "train.epochs (config)", 1)
            self._int_at_least(t["batch_size"], "train.batch_size (config)", 1)
            self._int_at_least(t.get("num_workers", 4), "train.num_workers (config)", 0)
            crop = cfg["data"]["crop"]
            self._int_at_least(crop, "data.crop (config)", 64, high=4096)
            if isinstance(crop, int) and crop % 32:
                self._error(f"data.crop (config): debe ser múltiplo de 32 (recibido: {crop}).")
            self._check_training_data(cfg)
            if p["resume"] and not (Path(cfg["paths"]["models"]) / "last.pt").is_file():
                self._error(f"--resume: no existe {Path(cfg['paths']['models']) / 'last.pt'}; "
                            f"no hay un entrenamiento anterior que retomar.")
        self._check_torch(p["allow_cpu"])
        p["cfg"] = cfg
        return p

    def _check_training_data(self, cfg: dict) -> None:
        """train y val deben tener todas sus máscaras (si se interrumpió prepare, faltarían fotos)."""
        processed, raw = Path(cfg["paths"]["processed"]), Path(cfg["paths"].get("raw", "data/raw"))
        for split in ("train", "val"):
            masks = count_files(processed / split / "masks", ".npz")
            expected = count_images(raw / "images" / split)
            if not masks:
                self._error(f"No hay máscaras en {processed / split / 'masks'}. "
                            f"Genéralas primero (opción 3 del menú).")
            elif expected and masks < expected:
                self._error(f"Las máscaras de '{split}' están incompletas ({masks} de {expected}). "
                            f"Vuelve a generarlas (opción 3 del menú) y déjalas terminar.")

    def _check_torch(self, allow_cpu: bool) -> None:
        """PyTorch debe poder cargarse y, salvo para pruebas, ver la GPU."""
        try:
            import torch  # noqa: F401
            import segmentation_models_pytorch  # noqa: F401
        except Exception as e:  # p. ej. OSError por DLL bloqueadas en Windows
            msg = f"No se pudo cargar PyTorch: {type(e).__name__}: {e}"
            if "4551" in str(e) or "Control de aplicaciones" in str(e) or "Application Control" in str(e):
                msg += ("\n    Windows (Smart App Control) está bloqueando las DLL de PyTorch. Desactívalo en "
                        "Seguridad de Windows > Control de aplicaciones y navegador > Smart App Control.")
            self._error(msg)
            return
        if not allow_cpu and not torch.cuda.is_available():
            self._error(f"PyTorch ({torch.__version__}) no detecta la GPU. Si la versión termina en '+cpu', "
                        f"reinstálalo con: pip install torch torchvision --index-url "
                        f"https://download.pytorch.org/whl/cu132   (o usa --allow-cpu solo para pruebas)")
