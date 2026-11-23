"""Entrenamiento del modelo.

Ciclo de cada época:
    1. ENTRENAR  por cada lote de train: predecir -> comparar con la máscara (pérdida) ->
                 ajustar un poco los pesos para reducir el error.
    2. VALIDAR   predecir las fotos completas de val y calcular el IoU de cada clase (no se
                 ajusta nada: solo se mide).
    3. GUARDAR   last.pt siempre (para poder retomar); best.pt si el mIoU de val mejoró.
    4. PARAR     si el mIoU de val no mejora en `patience` épocas seguidas.

Resultados en <paths.models>/:
    best.pt    el mejor modelo según val (el que se usará para predecir)
    last.pt    el estado de la última época (pesos + optimizador), para retomar
    log.csv    una fila por época: pérdida, mIoU, IoU por clase, tiempo

Uso: opción 5 del menú (python src/main.py) o python src/main.py train [--epochs N] [--resume]
"""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.classes import CLASSES, NUM_CLASSES
from training.dataset import BridgeDamageDataset
from modeling.inference import sliding_window_probs
from training.losses import build_loss
from modeling.model import build_model, count_parameters
from utils.files import ensure_dirs
from utils.metrics import MultilabelIoU


# ==============================================================================================
# Piezas
# ==============================================================================================
def seed_everything(seed: int) -> None:
    """Fija las semillas para que dos entrenamientos con la misma config sean comparables."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(allow_cpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("No se detecta una GPU con CUDA. Entrenar en CPU sería extremadamente lento; "
                       "usa --allow-cpu solo para pruebas.")


def build_optimizer(model, t: dict) -> torch.optim.Optimizer:
    """AdamW con dos velocidades: el encoder ya viene preentrenado y aprende más despacio
    (encoder_lr_mult veces el lr), para no "olvidar" lo que aprendió con ImageNet."""
    encoder, rest = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (encoder if name.startswith("encoder.") else rest).append(p)
    lr = t["lr"]
    return torch.optim.AdamW([
        {"params": encoder, "lr": lr * t.get("encoder_lr_mult", 0.1)},
        {"params": rest, "lr": lr},
    ], weight_decay=t.get("weight_decay", 0.01))


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Velocidad de aprendizaje: sube poco a poco durante el calentamiento (warmup) y después
    baja suavemente siguiendo un coseno hasta casi cero al final del entrenamiento."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, amp_dtype,
                    use_amp, grad_accum: int, grad_clip: float, epoch: int, epochs: int) -> float:
    """Una pasada por todas las fotos de train. Devuelve la pérdida promedio."""
    model.train()
    total, steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, desc=f"época {epoch}/{epochs}", leave=False)
    for i, (images, targets, _) in enumerate(bar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # 1) predecir (con precisión mixta en GPU: más rápido y usa menos memoria)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
        # 2) comparar con la máscara
        loss = loss_fn(logits.float(), targets)
        # 3) calcular hacia dónde mover cada peso (retropropagación)
        scaler.scale(loss / grad_accum).backward()
        # 4) mover los pesos (cada grad_accum lotes)
        if (i + 1) % grad_accum == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # evita saltos bruscos
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        total += loss.item()
        steps += 1
        bar.set_postfix(pérdida=f"{total / steps:.4f}")
    return total / max(1, steps)


@torch.no_grad()
def evaluate(model, dataset, tile: int, device, use_amp: bool, max_images: int | None = None) -> MultilabelIoU:
    """Predice las fotos completas de un grupo (val/test) y acumula el IoU. No ajusta nada."""
    metric = MultilabelIoU(NUM_CLASSES)
    n = len(dataset) if not max_images else min(max_images, len(dataset))
    for i in tqdm(range(n), desc="validando", leave=False):
        image, target, _ = dataset[i]
        probs = sliding_window_probs(model, image, tile=tile, device=device, amp=use_amp)
        metric.update(probs.unsqueeze(0), target.unsqueeze(0))
    return metric


def print_iou_table(per_class: np.ndarray, title: str) -> None:
    print(f"\n{title}")
    print(f"  {'Clase':<15}{'IoU':>8}")
    for name, value in zip(CLASSES, per_class):
        print(f"  {name:<15}{'   —' if np.isnan(value) else f'{value:8.3f}'}")
    print(f"  {'mIoU':<15}{np.nanmean(per_class) if not np.all(np.isnan(per_class)) else 0:8.3f}")


# ==============================================================================================
# Entrenamiento completo
# ==============================================================================================
def run(cfg: dict, epochs: int | None = None, resume: bool = False, allow_cpu: bool = False) -> None:
    """Punto de entrada reutilizable: lo llama el orquestador."""
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    seed_everything(t.get("seed", 42))
    device = select_device(allow_cpu)
    use_amp = t.get("amp", True) and device.type == "cuda"
    # bfloat16 si la GPU lo soporta (RTX serie 30/40 en adelante); si no, float16 con GradScaler.
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # ----- datos
    processed, crop = cfg["paths"]["processed"], cfg["data"]["crop"]
    train_ds = BridgeDamageDataset(processed, "train", crop=crop)
    val_ds = BridgeDamageDataset(processed, "val")
    workers = t.get("num_workers", 4)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          drop_last=len(train_ds) > t["batch_size"], num_workers=workers,
                          pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    # ----- modelo, pérdida, optimizador
    model = build_model(cfg).to(device)
    params = count_parameters(model)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(model, t)
    grad_accum = max(1, t.get("grad_accum", 1))
    steps_per_epoch = max(1, math.ceil(len(train_dl) / grad_accum))
    scheduler = build_scheduler(optimizer, steps_per_epoch * epochs, steps_per_epoch * t.get("warmup_epochs", 1))

    # ----- salida y (opcional) retomar
    out_dir = Path(cfg["paths"]["models"])
    ensure_dirs(out_dir)
    log_path = out_dir / "log.csv"
    start_epoch, best_miou = 1, -1.0
    patience = t.get("patience", 12)
    patience_left = patience
    if resume:
        state = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_miou, patience_left = state["best_miou"], state["patience_left"]
        print(f"Retomando desde la época {state['epoch']} (mejor mIoU hasta ahora: {best_miou:.4f})")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_mIoU", "lr", "seconds", *CLASSES])

    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Modelo: UPerNet + {cfg['model']['encoder']}  |  {params['total']:.1f} M parámetros "
          f"(encoder {params['encoder']:.1f} M, decoder {params['decoder']:.1f} M)")
    print(f"Datos: {len(train_ds)} fotos de train, {len(val_ds)} de val  |  lote {t['batch_size']}, "
          f"recorte {crop}, épocas {start_epoch}-{epochs}, precisión mixta: {'sí' if use_amp else 'no'}\n")

    val_every, max_val = t.get("val_every", 1), t.get("val_max_images")
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, scheduler, scaler, device,
                                     amp_dtype, use_amp, grad_accum, t.get("grad_clip", 1.0), epoch, epochs)

        miou, per_class = float("nan"), np.full(NUM_CLASSES, np.nan)
        validated = epoch % val_every == 0 or epoch == epochs
        if validated:
            metric = evaluate(model, val_ds, crop, device, use_amp, max_val)
            miou, per_class = metric.mean(), metric.per_class()

        improved = validated and miou > best_miou
        if improved:
            best_miou, patience_left = miou, patience
            torch.save({"model": model.state_dict(), "cfg": cfg, "classes": CLASSES, "epoch": epoch,
                        "val_miou": miou, "per_class_iou": dict(zip(CLASSES, per_class.tolist()))},
                       out_dir / "best.pt")
        elif validated:
            patience_left -= 1

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "cfg": cfg,
                    "classes": CLASSES, "epoch": epoch, "best_miou": best_miou,
                    "patience_left": patience_left}, out_dir / "last.pt")

        seconds, lr = time.time() - t0, optimizer.param_groups[1]["lr"]
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", "" if np.isnan(miou) else f"{miou:.5f}",
                                    f"{lr:.2e}", f"{seconds:.0f}",
                                    *["" if np.isnan(v) else f"{v:.4f}" for v in per_class]])
        mark = "  <- mejor, guardado best.pt" if improved else ""
        val_txt = f"val mIoU {miou:.4f}" if validated else "val —"
        print(f"[época {epoch}/{epochs}] pérdida {train_loss:.4f} | {val_txt} | "
              f"mejor {max(best_miou, 0):.4f} | {seconds:.0f} s{mark}")

        if patience_left <= 0:
            print(f"\nParada temprana: val no mejoró en {patience} validaciones seguidas.")
            break

    best_path = out_dir / "best.pt"
    if best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        print_iou_table(np.array(list(best["per_class_iou"].values()), dtype=float),
                        f"Mejor modelo: época {best['epoch']} (guardado en {best_path})")
    print(f"\nHistorial completo en {log_path}")
