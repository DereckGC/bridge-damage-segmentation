"""Funciones y constantes genéricas, reutilizables en todo el proyecto.

Cada módulo agrupa un tema. Ninguno depende de una acción concreta (convertir, preparar,
entrenar...), así que cualquier parte del proyecto puede usarlos:

    utils.constants   valores fijos compartidos (grupos, extensiones, config por defecto)
    utils.config      leer el config.yaml y consultar sus claves
    utils.files       listar y contar archivos e imágenes en carpetas
    utils.images      abrir, reducir y guardar fotos
    utils.masks       crear, guardar y cargar máscaras multi-etiqueta
    utils.drawing     pintar máscaras sobre fotos, leyendas y composiciones lado a lado
    utils.metrics     IoU por clase y mIoU para evaluar predicciones
    utils.labels      traducir nombres de etiquetas a las clases del proyecto
    utils.console     preguntar datos al usuario en la terminal

Uso:  from utils.masks import load_mask
"""
