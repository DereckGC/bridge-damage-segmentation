"""El modelo: UPerNet (decoder) sobre un encoder de timm (p. ej. ConvNeXt), vía segmentation-models-pytorch.

Entrada:  foto normalizada (B, 3, H, W), con H y W múltiplos de 32.
Salida:   logits (B, 19, H, W): un número por píxel y clase. La sigmoide los convierte en
          probabilidades entre 0 y 1 (se aplica en la pérdida y en la predicción, no aquí).
"""
import segmentation_models_pytorch as smp
import torch

from core.classes import CLASSES


def build_model(cfg):
    return smp.UPerNet(
        encoder_name=cfg["model"]["encoder"],                      # p. ej. "tu-convnext_tiny"
        encoder_weights="imagenet" if cfg["model"]["pretrained"] else None,
        classes=len(CLASSES),                                      # 19 canales de salida
        activation=None,                                           # devuelve logits
    )


def count_parameters(model) -> dict[str, float]:
    """Millones de parámetros del encoder, del resto (decoder + cabeza) y total."""
    total = sum(p.numel() for p in model.parameters())
    encoder = sum(p.numel() for p in model.encoder.parameters())
    return {"encoder": encoder / 1e6, "decoder": (total - encoder) / 1e6, "total": total / 1e6}


def load_trained_model(checkpoint_path, device="cpu"):
    """Reconstruye el modelo de un checkpoint (best.pt / last.pt) con sus pesos entrenados.
    Devuelve (modelo en modo evaluación, contenido del checkpoint)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    cfg["model"]["pretrained"] = False          # los pesos vienen del checkpoint, no de ImageNet
    if ckpt.get("classes") and list(ckpt["classes"]) != list(CLASSES):
        raise ValueError("El checkpoint se entrenó con otra lista de clases que la de core/classes.py.")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
