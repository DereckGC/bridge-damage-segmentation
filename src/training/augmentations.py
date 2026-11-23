"""Transformaciones que se aplican a cada par foto + máscara antes de dárselo al modelo.

Usa Albumentations (>= 2.0), que aplica EXACTAMENTE la misma transformación geométrica
(recorte, volteo, giro) a la foto y a su máscara, para que sigan alineadas píxel a píxel.
Las transformaciones de color (brillo, ruido, desenfoque) solo se aplican a la foto.

Está separado de dataset.py para que la revisión visual (data_prep/visualize.py) pueda usarlo sin
necesidad de cargar PyTorch.
"""
import albumentations as A
import cv2

# Media y desviación de ImageNet: el encoder se preentrenó con fotos normalizadas así,
# por lo que hay que normalizar nuestras fotos de la misma forma.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transforms(crop: int, normalize: bool = True) -> A.Compose:
    """Aumentaciones de ENTRENAMIENTO: cambian en cada época, así el modelo nunca ve dos veces
    exactamente la misma imagen y aprende a generalizar en vez de memorizar.

    crop:      tamaño del recorte cuadrado que recibe el modelo (data.crop del config, p. ej. 512)
    normalize: False solo para visualizar (deja la foto en colores normales)
    """
    steps = [
        # Si la foto es más chica que el recorte, se rellena con negro (y la máscara con 0).
        A.PadIfNeeded(min_height=crop, min_width=crop, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        # Recorte aleatorio: toma entre el 20 % y el 100 % de la foto y lo lleva a crop x crop.
        A.RandomResizedCrop(size=(crop, crop), scale=(0.2, 1.0), ratio=(0.75, 1.333)),
        # Geometría: espejo horizontal y giros/zoom leves.
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), border_mode=cv2.BORDER_CONSTANT,
                 fill=0, fill_mask=0, p=0.3),
        # Color: las fotos de inspección varían mucho en luz, sombra y cámara (solo afecta a la foto).
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            A.RandomGamma(),
        ], p=0.7),
        # Calidad: ruido, movimiento y compresión JPEG, como en fotos reales de celular.
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08)),
            A.MotionBlur(blur_limit=5),
            A.ImageCompression(quality_range=(50, 95)),
        ], p=0.3),
    ]
    if normalize:
        steps.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(steps)


def build_eval_transforms(normalize: bool = True) -> A.Compose:
    """Validación y test: SIN cambios aleatorios (siempre la misma foto completa), solo normalizar."""
    return A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)] if normalize else [])
