"""Función de pérdida: el número que mide "qué tan equivocado está el modelo" y que el
entrenamiento intenta reducir ajustando los pesos.

Combina dos pérdidas porque cada una cubre una debilidad de la otra:
  - BCE (entropía cruzada binaria): compara píxel a píxel la probabilidad con la respuesta.
    Es estable y aprende bien desde el principio, pero la dominan los píxeles de fondo
    (la gran mayoría), así que por sí sola descuida los daños pequeños o finos (grietas).
  - Dice: mide el solapamiento entre la zona predicha y la real de cada clase, sin importar
    cuántos píxeles de fondo haya. Está muy alineada con el IoU y es robusta al desbalance,
    pero sola es inestable al inicio.

Ambas tratan cada clase como un problema de sí/no independiente (multi-etiqueta), igual que
la sigmoide del modelo: un píxel puede ser Rust y Drainage a la vez.
"""
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.bce = nn.BCEWithLogitsLoss()                            # aplica la sigmoide internamente
        self.dice = DiceLoss(mode="multilabel", from_logits=True)    # idem

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: salida cruda del modelo (B, C, H, W). target: máscara (B, C, H, W) con 0/1."""
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(cfg: dict) -> SegmentationLoss:
    t = cfg.get("train", {})
    return SegmentationLoss(bce_weight=t.get("bce_weight", 1.0), dice_weight=t.get("dice_weight", 1.0))
