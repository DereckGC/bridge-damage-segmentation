
CLASSES = [
    "Crack",          # 0  grieta
    "ACrack",         # 1  grieta tipo piel de cocodrilo (alligator crack)
    "Wetspot",        # 2  mancha de humedad
    "Efflorescence",  # 3  eflorescencia
    "Rust",           # 4  óxido
    "Rockpocket",     # 5  coquera / nido de grava
    "Hollowareas",    # 6  zonas huecas (delaminación marcada)
    "Cavity",         # 7  cavidad / poro
    "Spalling",       # 8  desprendimiento de recubrimiento
    "Graffiti",       # 9  grafiti
    "Weathering",     # 10 meteorización
    "Restformwork",   # 11 restos de encofrado
    "ExposedRebars",  # 12 armadura expuesta
    "Bearing",        # 13 apoyo (objeto)
    "EJoint",         # 14 junta de dilatación (objeto)
    "Drainage",       # 15 drenaje (objeto)
    "PEquipment",     # 16 equipo de protección / barandas (objeto)
    "JTape",          # 17 cinta de junta (objeto)
    "WConccor",       # 18 lavado / corrosión del concreto
]

NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Colores RGB para visualizar (arbitrarios, solo para overlays).
COLORS = [
    (230, 25, 75),    # 0  Crack
    (245, 130, 48),   # 1  ACrack
    (0, 130, 200),    # 2  Wetspot
    (255, 225, 25),   # 3  Efflorescence
    (170, 110, 40),   # 4  Rust
    (145, 30, 180),   # 5  Rockpocket
    (70, 240, 240),   # 6  Hollowareas
    (240, 50, 230),   # 7  Cavity
    (210, 245, 60),   # 8  Spalling
    (250, 190, 212),  # 9  Graffiti
    (0, 128, 128),    # 10 Weathering
    (220, 190, 255),  # 11 Restformwork
    (128, 0, 0),      # 12 ExposedRebars
    (170, 255, 195),  # 13 Bearing
    (128, 128, 0),    # 14 EJoint
    (255, 215, 180),  # 15 Drainage
    (0, 0, 128),      # 16 PEquipment
    (128, 128, 128),  # 17 JTape
    (60, 180, 75),    # 18 WConccor
]

assert len(COLORS) == NUM_CLASSES, "Debe haber un color por clase"