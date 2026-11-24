# Detección de daños en puentes con segmentación semántica (UPerNet)

Modelo que detecta, **píxel por píxel**, daños y elementos en fotos de inspección de puentes de
concreto: grietas, eflorescencias, óxido, desprendimientos, armadura expuesta, entre otros
(19 clases).

- **Tarea:** segmentación semántica **multi‑etiqueta**: un mismo píxel puede tener varias clases
  a la vez (p. ej. óxido sobre un drenaje).
- **Modelo:** UPerNet (decoder) sobre un encoder ConvNeXt preentrenado en ImageNet,
  con [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch).
- **Datos:** [dacl10k](https://arxiv.org/abs/2309.00460), 9 920 fotos de inspecciones reales
  anotadas con polígonos.
- **Métrica:** IoU por clase y su promedio (mIoU).

## Estructura

```
.
├── config/
│   └── config.yaml              # rutas, tamaños e hiperparámetros (todo lo ajustable)
├── src/
│   ├── main.py                  # punto de entrada: menú interactivo o comandos
│   ├── app/                     # interfaz con el usuario
│   │   ├── actions.py           #   catálogo de acciones y sus parámetros
│   │   ├── orchestrator.py      #   interpreta el pedido, valida y ejecuta
│   │   ├── menu.py              #   menú interactivo
│   │   ├── tasks.py             #   tareas que ofrece el menú
│   │   ├── status.py            #   estado del proyecto que muestra el menú
│   │   └── validation/          #   reglas que se revisan antes de ejecutar
│   ├── core/
│   │   └── classes.py           # las 19 clases y sus colores
│   ├── data_prep/               # de la descarga a los pares foto + máscara
│   │   ├── convert/             #   dataset descargado -> data/raw
│   │   │   ├── readers/         #     un lector por formato (fiftyone, supervisely)
│   │   │   ├── sample.py        #     estructura común
│   │   │   └── converter.py     #     reparte en grupos y escribe
│   │   ├── prepare.py           #   data/raw -> data/processed (fotos + máscaras)
│   │   └── visualize.py         #   revisión visual de las máscaras
│   ├── modeling/
│   │   ├── model.py             # construir UPerNet / cargar un modelo entrenado
│   │   └── inference.py         # predicción con ventana deslizante
│   ├── training/
│   │   ├── dataset.py           # pares foto + máscara como tensores
│   │   ├── augmentations.py     # aumentaciones y normalización
│   │   ├── losses.py            # pérdida BCE + Dice
│   │   └── trainer.py           # ciclo de entrenamiento y validación
│   └── utils/                   # funciones genéricas (config, archivos, imágenes,
│                                #   máscaras, etiquetas, dibujo, métricas, consola)
├── requirements.txt
└── README.MD
```

Carpetas que se crean al usar el proyecto (**no se suben a git**, ver `.gitignore`):

| Carpeta | Contenido |
|---|---|
| `dataset/` | Datasets descargados, sin convertir |
| `data/raw/` | Fotos y anotaciones (JSON) ordenadas en `train/`, `val/`, `test/` |
| `data/processed/` | Fotos reducidas y máscaras `.npz`, listas para entrenar |
| `models/` | `best.pt`, `last.pt` y `log.csv` |
| `inputs/` | Fotos a analizar (incluye `sin_anotar/`) |
| `outputs/` | Imágenes de revisión y resultados |

Flujo: `dataset/` → **convertir** → `data/raw/` → **generar máscaras** → `data/processed/` → **entrenar** → `models/best.pt`

---

## Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| Sistema | Windows 10/11 o Linux | — |
| Python | 3.10 – 3.12 | 3.12 |
| GPU | NVIDIA con 6 GB (con `tu-convnext_tiny`, lote 4) | NVIDIA con 12 GB o más |
| RAM | 16 GB | 32 GB |
| Disco | 20 GB libres | 40 GB (SSD) |

---

## Instalación

Todos los comandos se ejecutan **desde la raíz del proyecto** (PowerShell en Windows).

```powershell
# 1. Entorno virtual (con Python de la Microsoft Store usa "python"; con python.org, "py -3.12")
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip

# 2. PyTorch con GPU (SIEMPRE aparte y antes que el resto)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Resto de dependencias
pip install -r requirements.txt

# 4. Verificar: debe terminar en "True" y mostrar el nombre de la GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- `cu132` corresponde a CUDA 13.2. Si tu driver es más antiguo (mira "CUDA Version" en `nvidia-smi`),
  usa `cu130` o `cu128`.
- **Windows 11:** si aparece `WinError 4551 ... directiva de Control de aplicaciones`, *Smart App
  Control* está bloqueando las DLL de PyTorch. Desactívalo en *Seguridad de Windows > Control de
  aplicaciones y navegador > Smart App Control*.
- Activa el entorno (`.venv\Scripts\activate`) en cada terminal nueva.

---

## Dataset

El enlace oficial de dacl10k ya no está disponible. Se usa la copia publicada en Hugging Face
([Voxel51/dacl10k](https://huggingface.co/datasets/Voxel51/dacl10k), ~5.2 GB). Se descarga con un
entorno aparte, para no mezclar dependencias con el proyecto:

```powershell
python -m venv .venv-descarga
.venv-descarga\Scripts\activate
pip install huggingface_hub
python -X utf8 -c "from huggingface_hub import snapshot_download; snapshot_download('Voxel51/dacl10k', repo_type='dataset', local_dir='dataset/voxel51', max_workers=4)"
deactivate
```

- `-X utf8` evita errores de codificación en Windows.
- Si se interrumpe, repite el mismo comando: continúa donde quedó.
- Al terminar, `dataset/voxel51/data` debe tener 8 922 imágenes.

> **Licencia de los datos:** dacl10k se distribuye bajo **CC BY‑NC 4.0** (uso no comercial).
> Los datos **no** forman parte de este repositorio y no deben subirse.

---

## Uso

```powershell
python src/main.py
```

Abre un **menú** que muestra el estado del proyecto, sugiere el siguiente paso y pide los datos
paso a paso (Enter = valor sugerido):

| Opción | Qué hace |
|---|---|
| 1. Revisar un dataset descargado | Muestra fotos por grupo y cómo se traduce cada etiqueta. No modifica nada |
| 2. Convertir un dataset | Copia fotos y anotaciones a `data/raw/<grupo>`. Divide `validation` en `val` y `test` |
| 3. Generar las máscaras | Reduce las fotos y crea las máscaras de 19 capas en `data/processed` |
| 4. Revisar las máscaras | Dibuja las máscaras sobre las fotos en `outputs/check_masks` (opcional: con aumentaciones) |
| 5. Entrenar el modelo | Entrena, valida cada época y guarda `models/best.pt`. Se puede interrumpir y retomar |

También se puede usar con comandos (`python src/main.py <accion> -h` muestra los parámetros):

```powershell
python src/main.py convert --src dataset/voxel51 --inspect
python src/main.py convert --src dataset/voxel51 --copy
python src/main.py prepare
python src/main.py visualize --split train --n 10 [--augmented]
python src/main.py train [--epochs N] [--resume]
```

Antes de ejecutar, cada acción se **valida** (rutas, config, máscaras completas, PyTorch y GPU) y
se muestran todos los problemas juntos.

### División de los datos

| Grupo | Fotos | Uso |
|---|---|---|
| train | 6 935 | Aprender |
| val | 488 | Medir cada época, elegir `best.pt` y decidir cuándo parar |
| test | 487 | Evaluación final (una sola vez) |
| sin anotar | 1 012 | Probar la detección (`inputs/sin_anotar`) |

`val` y `test` salen de dividir por la mitad el `validation` original (semilla fija), porque el
test de dacl10k no publica sus anotaciones.

---

## Configuración (`config/config.yaml`)

| Clave | Para qué |
|---|---|
| `paths.*` | Carpetas del proyecto (`downloads`, `raw`, `processed`, `models`, ...) |
| `data.long_side` | Lado largo de las fotos procesadas (1024). Si lo cambias, regenera las máscaras |
| `data.crop` | Recorte de entrenamiento y ventana de predicción (512, múltiplo de 32) |
| `model.encoder` | `tu-convnext_tiny` (6 GB), `tu-convnext_small` / `tu-convnext_base` (más GPU) |
| `train.batch_size` | Fotos por lote; bájalo si aparece `CUDA out of memory` |
| `train.epochs`, `train.patience` | Máximo de épocas y épocas sin mejorar antes de parar |
| `train.lr`, `train.encoder_lr_mult` | Velocidad de aprendizaje (el encoder aprende 10 veces más lento) |

El resto de claves está comentado en el propio archivo.

### Resultados del entrenamiento (`models/`)

- `best.pt`: el modelo con mejor mIoU en val (unos 150 MB con `convnext_tiny`).
- `last.pt`: estado completo de la última época, para retomar (unos 450 MB).
- `log.csv`: una fila por época con pérdida, mIoU, IoU por clase y tiempo.

---

## Clases

| # | Clase | Descripción | # | Clase | Descripción |
|---|---|---|---|---|---|
| 0 | Crack | Grieta | 10 | Weathering | Meteorización / desgaste |
| 1 | ACrack | Grieta tipo piel de cocodrilo | 11 | Restformwork | Restos de encofrado |
| 2 | Wetspot | Mancha de humedad | 12 | ExposedRebars | Armadura expuesta |
| 3 | Efflorescence | Eflorescencia | 13 | Bearing | Apoyo |
| 4 | Rust | Óxido | 14 | EJoint | Junta de dilatación |
| 5 | Rockpocket | Coquera / nido de grava | 15 | Drainage | Drenaje |
| 6 | Hollowareas | Zonas huecas | 16 | PEquipment | Equipo de protección |
| 7 | Cavity | Cavidad / poro | 17 | JTape | Cinta de junta |
| 8 | Spalling | Desprendimiento | 18 | WConccor | Lavado / corrosión del concreto |
| 9 | Graffiti | Grafiti | | | |

El orden de `src/core/classes.py` define el número de capa de cada máscara. Si se cambia, hay que
regenerar las máscaras y volver a entrenar.

---

## Cómo extender el proyecto

- **Un formato de dataset nuevo** (COCO, Label Studio...): crea `src/data_prep/convert/readers/<formato>.py`
  con `matches(src)` y `read_<formato>(src)` y regístralo en `readers/__init__.py`.
- **Una acción nueva:** su entrada en `src/app/actions.py`, su método `_run_<accion>` en
  `src/app/orchestrator.py`, su regla `_validate_<accion>` en `src/app/validation/` y, si debe
  aparecer en el menú, su tarea en `src/app/tasks.py`.

---

## Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `py` no se reconoce | Python de la Microsoft Store no trae el lanzador `py` | Usa `python -m venv .venv` |
| Error de "scripts deshabilitados" al activar el entorno | Política de PowerShell | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `WinError 4551` al importar torch | Smart App Control bloquea las DLL | Desactivar Smart App Control (ver Instalación) |
| La versión de torch termina en `+cpu` | Se instaló desde PyPI | `pip uninstall -y torch torchvision` y reinstalar con `--index-url .../cu132` |
| `UnicodeDecodeError ... cp1252` al descargar | Codificación por defecto de Windows | Ejecutar Python con `-X utf8` |
| La descarga se corta o da `429` | Límite de Hugging Face | Repetir el comando (continúa) o iniciar sesión con `hf auth login` |
| El menú marca las máscaras como incompletas | Se interrumpió la generación | Opción 3 de nuevo, hasta que termine |
| `CUDA out of memory` al entrenar | Poca memoria de GPU | Bajar `train.batch_size` o usar un encoder más pequeño |

---