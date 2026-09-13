"""Configuración centralizada de SIGNIA.

Toda constante de pipeline, entrenamiento e inferencia vive aquí.
La misma semántica debe replicarse en el frontend (config.js).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Clases
# ---------------------------------------------------------------------------
# Orden fijo: es el orden de las neuronas de salida del modelo.
CLASES = [
    "HOLA",
    "ADIOS",
    "GRACIAS",
    "POR_FAVOR",
    "SI",
    "NO",
    "AYUDA",
    "BIEN",
    "MAL",
    "AMIGO",
]

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "ai" / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
METADATA_DIR = DATASET_DIR / "metadata"
MODELS_DIR = PROJECT_ROOT / "ai" / "models"
FRONTEND_MODEL_DIR = PROJECT_ROOT / "frontend" / "assets" / "model"
REPORTS_DIR = PROJECT_ROOT / "ai" / "training" / "reports"

METADATA_CSV = METADATA_DIR / "metadata.csv"
PREPROCESSED_FEATURES = PROCESSED_DIR / "features.npy"
PREPROCESSED_LABELS = PROCESSED_DIR / "labels.npy"
PREPROCESSED_SPLITS = PROCESSED_DIR / "splits.json"

MODEL_FILE = MODELS_DIR / "signia_model.keras"
LABELS_FILE = MODELS_DIR / "labels.json"
TRAIN_CONFIG_FILE = MODELS_DIR / "train_config.json"

# ---------------------------------------------------------------------------
# Exportación a TensorFlow.js (Fase 8)
# ---------------------------------------------------------------------------
MODEL_NAME = "SIGNIA"
MODEL_VERSION = 1                 # incrementar manualmente al cambiar el modelo
FRONTEND_MODELS_DIR = PROJECT_ROOT / "frontend" / "models" / "signia"
FRONTEND_METADATA_FILE = FRONTEND_MODELS_DIR / "metadata.json"

# Tolerancias de paridad Python <-> JavaScript (Fase 8)
# ~1e-3 absorbida porque los runtimes (TF CPU vs TF.js CPU/WASM) calculan
# operaciones en el MISMO orden float32 pero con diferentes vectorizaciones;
# la diferencia típica observada es <= 1e-6.
PARITY_MAX_ABS_DIFF_TOLERANCE = 1e-3
PARITY_MAE_TOLERANCE = 1e-4

# ---------------------------------------------------------------------------
# Formato de los datos
# ---------------------------------------------------------------------------
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # x, y, z
HANDS = 2
FEATURES_SIZE = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK * HANDS  # 126

# Posición de los bloques dentro del vector de features.
# Bloque 0 (índices 0..62): mano etiquetada "Left" por MediaPipe Tasks Vision
#   sobre el frame SIN espejar (misma convención que la inferencia del frontend).
# Bloque 1 (índices 63..125): mano etiquetada "Right".
# La mano ausente se rellena con ceros.
LEFT_HAND_SLICE = slice(0, LANDMARKS_PER_HAND * COORDS_PER_LANDMARK)
RIGHT_HAND_SLICE = slice(
    LANDMARKS_PER_HAND * COORDS_PER_LANDMARK,
    LANDMARKS_PER_HAND * COORDS_PER_LANDMARK * HANDS,
)

# ---------------------------------------------------------------------------
# Normalización (DEBE ser idéntica en entrenamiento e inferencia)
# ---------------------------------------------------------------------------
# 1. Referencia (origen): muñeca = landmark 0.
# 2. Escala: distancia muñeca (0) -> MCP del dedo medio (9), por mano.
# 3. Las coordenadas de cada landmark se reexpresan como
#    (landmark - muñeca) / escala.
ANCHOR_LANDMARK = 0        # muñeca
SCALE_LANDMARK = 9         # MCP medio (tamaño de palma)
EPSILON = 1e-6             # evita división por cero

# ---------------------------------------------------------------------------
# Entrenamiento (MLP sobre el vector de 126 dimensiones)
# ---------------------------------------------------------------------------
SEED = 42
TEST_PERSONS = []          # personas reservadas para TEST (revisar requisito 9)
VALIDATION_PERSONS = []    # personas reservadas para VALIDATION
TRAIN_PERSONS = []         # resto por defecto

TARGET_SAMPLES_PER_CLASS = 500   # objetivo mínimo por clase (requisito 8)
AUGMENTATION = True              # ruido/pequeñas perturbaciones si está activado

MLP_LAYERS = [128, 64]           # neuronas por capa oculta
DROPOUT_RATE = 0.3
ACTIVATION = "relu"
EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE_EARLY_STOPPING = 20
PATIENCE_LR_REDUCE = 8
VALIDATION_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Inferencia (umbrales configurables)
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.70   # requisito 14
SMOOTHING_WINDOW = 5          # últimas N predicciones (requisito 15)
CONFIRMATION_MIN_VOTES = 4    # mínimo de coincidencias en la ventana
COOLDOWN_SAMPLES = 30         # frames de enfriamiento tras confirmar (requisito 16)

# ---------------------------------------------------------------------------
# Recolección
# ---------------------------------------------------------------------------
PERSON_ID_DEFAULT = "P1"       # identificar quién hizo cada muestra
CAMERA_INDEX = 0

# F14: sesiones de recolección (A/B/C en días distintos) y tipología de señas.
SESSIONS = ("A", "B", "C")
SESSION_DEFAULT = "A"

# Clasificación F13 (docs/clases_lsc.md): estáticas compatibles con el MLP del
# MVP; dinámicas requieren secuencia (TEMPORAL_MODEL_REQUIRED a futuro).
SIGN_TYPES = {
    "HOLA": "dynamic",
    "ADIOS": "dynamic",
    "GRACIAS": "dynamic",
    "POR_FAVOR": "dynamic",
    "SI": "dynamic",
    "NO": "dynamic",
    "AYUDA": "dynamic",
    "BIEN": "static",
    "MAL": "static",
    "AMIGO": "dynamic",
}

# Manos esperadas en el fotograma guía de cada clase (F13).
EXPECTED_HANDS = {
    "HOLA": 1,
    "ADIOS": 1,
    "GRACIAS": 2,
    "POR_FAVOR": 1,
    "SI": 1,
    "NO": 1,
    "AYUDA": 2,
    "BIEN": 1,
    "MAL": 1,
    "AMIGO": 2,
}

# F14: retención breve de pose en estáticas y límite de grabación en
# dinámicas (valores iniciales documentados, NO óptimos asumidos).
STATIC_HOLD_FRAMES = 25
MAX_SEQUENCE_SECONDS = 8.0

# F14: muestras descartadas por el operador (tecla D) se mueven aquí; el
# validador las cuenta como "descartadas" sin borrarlas silenciosamente.
DISCARDED_DIRNAME = "_discarded"

# F14: el recolector usa el MISMO archivo .task de MediaPipe Tasks Vision que
# el frontend, para que entrenamiento e inferencia compartan modelo y
# convención de handedness (frame sin espejar).
HAND_LANDMARKER_MODEL = (
    PROJECT_ROOT / "frontend" / "vendor" / "mediapipe" / "hand_landmarker.task"
)
REPORTS_DATASET_DIR = DATASET_DIR / "reports"