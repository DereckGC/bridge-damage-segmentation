"""Predicción sobre una foto completa con "ventana deslizante".

El modelo se entrena con recortes de crop x crop (p. ej. 512). Para predecir una foto entera
(p. ej. 1024 x 683) se recorre con ventanas de ese tamaño que se solapan un poco; en las zonas
solapadas se promedian las probabilidades, para que no se noten los bordes de cada ventana.

Lo usa el entrenamiento (para validar con las fotos completas de val) y lo usará la predicción.
"""
import torch
import torch.nn.functional as F


def _window_starts(length: int, tile: int, stride: int) -> list[int]:
    """Posiciones de inicio de las ventanas a lo largo de un eje, cubriendo todo el largo."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)   # última ventana pegada al borde
    return starts


@torch.no_grad()
def sliding_window_probs(model, image: torch.Tensor, tile: int, overlap: float = 0.25,
                         device=None, amp: bool = True, batch: int = 4) -> torch.Tensor:
    """image: foto normalizada (3, H, W). Devuelve probabilidades (C, H, W) en CPU, entre 0 y 1."""
    device = device or next(model.parameters()).device
    model.eval()
    _, h, w = image.shape
    # Relleno: al menos tile x tile y múltiplo de 32 (el encoder reduce el tamaño a la mitad 5 veces).
    hp, wp = max(h, tile), max(w, tile)
    hp, wp = (hp + 31) // 32 * 32, (wp + 31) // 32 * 32
    x = F.pad(image.unsqueeze(0), (0, wp - w, 0, hp - h))

    stride = max(1, int(tile * (1 - overlap)))
    coords = [(y, xx) for y in _window_starts(hp, tile, stride) for xx in _window_starts(wp, tile, stride)]
    acc, count = None, torch.zeros(1, hp, wp)
    use_amp = amp and device.type == "cuda"

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        tiles = torch.cat([x[:, :, y:y + tile, xx:xx + tile] for y, xx in chunk]).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            probs = torch.sigmoid(model(tiles).float()).cpu()
        if acc is None:
            acc = torch.zeros(probs.shape[1], hp, wp)
        for (y, xx), p in zip(chunk, probs):
            acc[:, y:y + tile, xx:xx + tile] += p
            count[:, y:y + tile, xx:xx + tile] += 1
    return (acc / count.clamp(min=1))[:, :h, :w]   # promediar solapes y quitar el relleno
