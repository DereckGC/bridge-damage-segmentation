"""Métricas de segmentación multi-etiqueta: IoU por clase y su promedio (mIoU).

IoU de una clase = píxeles donde predicción Y realidad dicen "sí" / píxeles donde CUALQUIERA dice "sí"
                 = TP / (TP + FP + FN)
Se acumula sobre TODO el grupo (no foto por foto) y luego se promedia entre clases.
Una clase que no aparece ni se predice en ningún píxel no tiene IoU definido (NaN) y no cuenta
para el promedio.
"""
import numpy as np
import torch


class MultilabelIoU:
    def __init__(self, num_classes: int, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = np.zeros(self.num_classes, dtype=np.float64)   # aciertos: dijo sí y era sí
        self.fp = np.zeros(self.num_classes, dtype=np.float64)   # falsas alarmas: dijo sí y era no
        self.fn = np.zeros(self.num_classes, dtype=np.float64)   # omisiones: dijo no y era sí

    @torch.no_grad()
    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        """probs y target con forma (B, C, H, W). probs en [0, 1]; target con 0/1."""
        pred = probs >= self.threshold
        real = target > 0.5
        dims = (0, 2, 3)  # sumar sobre fotos, alto y ancho: queda un número por clase
        self.tp += (pred & real).sum(dim=dims).double().cpu().numpy()
        self.fp += (pred & ~real).sum(dim=dims).double().cpu().numpy()
        self.fn += (~pred & real).sum(dim=dims).double().cpu().numpy()

    def per_class(self) -> np.ndarray:
        """IoU de cada clase (NaN si la clase no aparece ni se predijo)."""
        union = self.tp + self.fp + self.fn
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(union > 0, self.tp / union, np.nan)

    def mean(self) -> float:
        """mIoU: promedio del IoU de las clases que tienen IoU definido."""
        values = self.per_class()
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else 0.0
