# SIGNIA — Traductor inteligente de Lengua de Señas Colombiana

MVP funcional que reconoce con la webcam 10 señas de la **Lengua de Señas
Colombiana (LSC)** y las traduce a palabras en español con voz.

Webcam → MediaPipe → landmarks → normalización → modelo ML → suavizado
temporal → texto + voz. Toda la inferencia ocurre **localmente en el
navegador** (TensorFlow.js); no hay backend ni envío de imágenes.

> Estado del proyecto: EN DESARROLLO POR FASES. Ver el README completo
> (estructura, instalación, recolección, entrenamiento, evaluación,
> ejecución, clases, limitaciones, privacidad y roadmap) más abajo.

## Estado por fase

| Fase | Descripción | Estado |
|------|-------------|--------|
| F0 | Documentación LSC (docs/señas_lsc.md) | Fichas con fuentes oficiales; validación de configuraciones pendiente |
| F1 | Estructura + entorno | COMPLETADA (venv + deps + shim Windows documentado) |
| F2 | Detector de manos | Código listo; verificación pendiente (sin cámara en el equipo de desarrollo) |
| F3 | Normalización (features.py) | COMPLETADA — 9 tests |
| F4 | Recolector | Código listo (webcam + URL de celular); verificación pendiente (sin cámara) |
| F5 | Dataset + preprocess + split por persona | COMPLETADA en código — 18 tests; recolecta el dataset real cuando haya cámara |
| F6 | Entrenamiento (train.py) | COMPLETADA en código — EarlyStopping, ModelCheckpoint best, artefacto de reproducibilidad |
| F7 | Evaluación (evaluate.py) | COMPLETADA en código — F1/confusión/PNG + etiqueta por persona vs fallback |
| F8 | Conversión TF.js + paridad | COMPLETADA en código — export_tfjs.py con metadata.json; espejo JS de normalización (features.js); paridad Python ↔ JS (parity.mjs) con fixtures compartidos |
| F9 | Frontend base (cámara + MediaPipe + modelo + estados) | COMPLETADA en código — app local sin CDN; verificación de cámara pendiente (sin webcam en el equipo) |
| F10 | Estabilidad temporal (smoothing) | COMPLETADA en código — EMA + N frames consecutivos; capa independiente, NO modifica el modelo |
| F11 | Historial + voz | COMPLETADA en código — historial de señas confirmadas, SpeechSynthesis local es-CO→es-ES, persistencia opcional; verificación de cámara/voz pendiente (sin webcam en el equipo) |
| F12 | UI final del MVP | COMPLETADA en código — jerarquía visual, responsive 320–1440px, accesibilidad (aria-live solo eventos, focus-visible, reduced-motion), diagnóstico oculto Alt+D, metadata/favicon/manifest PWA-prep; verificación visual pendiente (sin webcam) |
| F13 | Validación científica de las 10 señas + diseño del dataset | COMPLETADA en documentación — 10 clases VALIDADA con citas textuales del DBLSC (docs/clases_lsc.md), STATIC/DYNAMIC identificada (4 compatibles con el MLP, 6 requieren modelo temporal), docs/protocolo_dataset.md (personas, sesiones, splits por persona, consentimiento); sin recolección ni entrenamiento todavía |

## Estructura

```
SIGNIA/
├── frontend/            # Aplicación web (inferencia local en navegador)
│   ├── js/              # features.js (normalización, espejo JS de Python)
│   └── models/signia/   # model.json + shards + metadata.json (TF.js)
├── ai/
│   ├── dataset/         # raw/P/S/CLASE/SAMPLE (esquema F14), _discarded/, processed/, reports/
│   ├── collection/      # collect_data.py, hand_tracker.py
│   ├── preprocessing/   # samples.py, features.py, validate.py, preprocess.py, report_dataset.py
│   ├── training/        # config.py, train.py, evaluate.py, export_tfjs.py
│   ├── models/          # signia_model.keras, labels.json, config
│   └── notebooks/
├── docs/                # señas_lsc.md, clases_lsc.md, protocolo_dataset.md,
│                        # dataset_schema.md, consentimiento.md, piloto_dataset_report.md
├── tests/               # pytest + tests/js_parity/ (paridad Python ↔ JS)
└── requirements.txt
```

## Instalación (Windows)

```powershell
cd SIGNIA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1) Núcleo + entrenamiento
python -m pip install mediapipe opencv-python "numpy>=1.26" pandas scikit-learn matplotlib pytest
python -m pip install "tensorflow==2.19.0" "tf-keras==2.19.0"
# 2) Conversión TF.js (requiere shim en Windows, ver nota)
python -m pip install "tensorflowjs==4.20.0" --no-deps
python -m pip install tensorflow_hub
python -m pip install jax jaxlib flax --only-binary=:all:
```

**Nota Windows (shim):** `tensorflow_decision_forests` no tiene wheels para
Windows, pero tensorflowjs lo importa para *detectar* modelos TFDF. Crear un
módulo vacío (no se usa en la conversión de modelos Keras):

```powershell
New-Item -ItemType Directory -Path ".venv\Lib\site-packages\tensorflow_decision_forests" -Force
Set-Content -Path ".venv\Lib\site-packages\tensorflow_decision_forests\__init__.py" -Value ""
```

Verificar:

```powershell
python -c "import tensorflow as tf, tensorflowjs as tfjs, mediapipe as mp; print('OK')"
python -m pytest tests/ -v
```

## Uso rápido

```powershell
# Verificación de cámara + landmarks (Fase 2)
python ai/collection/check_camera.py
# Validación del dataset (funciona sin dataset: avisa claramente)
python ai/preprocessing/validate_dataset.py
# Recolección de datos (F14; requiere cámara o celular). Sesiones A/B/C, sample_id automático:
python ai/collection/collect_data.py --person P01 --session A --target 100
python ai/collection/collect_data.py --person P01 --session A --target 100 `
    --camera-source http://192.168.1.10:8080/video --source smartphone
# Validación/QC del dataset (marca INVALID sin borrar; funciona sin dataset):
python ai/preprocessing/validate_dataset.py
# Reporte del estado del dataset real (JSON + docs/piloto_dataset_report.md):
python ai/preprocessing/report_dataset.py
# Preprocesado + split por persona (F5; requiere dataset):
python ai/preprocessing/preprocess.py
# Entrenamiento (Fase 6; requiere dataset). Overrides opcionales:
python ai/training/train.py
python ai/training/train.py --epochs 300 --units 128,64 --dropout 0.3 --lr 0.001
# Evaluación sobre el TEST set por persona (Fase 7):
python ai/training/evaluate.py
# Conversión a TensorFlow.js (Fase 8): Keras -> frontend/models/signia/
python ai/training/export_tfjs.py
# Paridad Python <-> TF.js (Fase 8; antes: npm install en tests/js_parity/)
python -m pytest tests/ -v
# Frontend (Fase 9): servidor local, app en http://localhost:8000
node frontend/server.js
# Tests de lógica frontend (Node, sin cámara):
node --test "tests/frontend/**/*.test.mjs"   # 108 tests (F9–F12)
```

## Frontend (Fase 9)

La app se sirve desde `frontend/` como raíz (`node frontend/server.js`).
Todo es LOCAL: MediaPipe (WASM), TensorFlow.js y el modelo de landmarks
están en `frontend/vendor/` — no hay CDN ni envío de frames a servidores.

| Pieza | Archivo | Notas |
|-------|---------|-------|
| Estados del sistema | `frontend/js/states.js` | 10 estados explícitos, mensajes + iconos |
| Cámara | `frontend/js/camera.js` | getUserMedia, 5 clases de error |
| Detección | `frontend/js/handTracking.js` | MediaPipe tasks-vision local, Left/Right |
| Features | `frontend/js/features.js` | espejo exacto de features.py (F8) |
| Modelo | `frontend/js/predictor.js` | carga model.json + metadata.json, valida shapes/clases |
| Smoothing (F10) | `frontend/js/temporalSmoother.js` | EMA + N frames consecutivos; capa independiente |
| Historial (F11) | `frontend/js/history.js` | solo señas CONFIRMADAS; sin duplicados continuos; mano ausente permite repetir; máx. 50 |
| Confirmación (F11) | `frontend/js/flow.js` | puente F10→F11: historial + voz solo con `isNewConfirmation` |
| UI (F12) | `frontend/js/app.js` + css/ | UI final: header con chips de estado, cámara | traducción, controles, historial, footer |
| Lógica UI (F12) | `frontend/js/ui.js` | decisiones de presentación puras y testeables (placeholders, estado del modelo, voz, historial, errores) |
| Voz (F11) | `frontend/js/speech.js` | SpeechSynthesis local es-CO → es-ES → es-*; reemplaza la locución previa |

### Estabilidad temporal (F10)

Pipeline: cámara → MediaPipe → landmarks → features → modelo TF.js →
probabilidades RAW por frame → **smoothing (capa independiente)** →
predicción mostrada.

`temporalSmoother.js` NO modifica el modelo ni sus probabilidades: las
RAW se comparan y se muestran en el panel de diagnóstico (RAW vs
SMOOTHED); la interfaz principal muestra solo el resultado final.
Es **ESTABILIDAD TEMPORAL**, no una mejora de precisión: no se reportan
métricas de accuracy del smoothing hasta una evaluación formal con
dataset real.

Reglas (configurables en `SMOOTHING_CONFIG`:
`alpha=0.35`, `minConfidence=0.70`, `stableFrames=5` — valores iniciales,
no óptimos):

1. EMA por clase sobre las probabilidades del modelo.
2. La clase dominante es el argmax RAW del modelo.
3. Una clase solo se confirma tras `stableFrames` frames consecutivos
   dominantes con evidencia ≥ `minConfidence`.
4. `maxProbability < minConfidence` → "Confianza insuficiente" (nunca
   se inventa una clase).
5. Sin manos: se limpia el estado; al volver no se arrastra la seña
   anterior.

Prueba manual: `docs/prueba_manual_f10.md`.

### Historial + voz (F11)

La voz y el historial funcionan SOLO sobre **señas confirmadas** por F10
(`isNewConfirmation`), nunca fotograma a fotograma.

Flujo: cámara → MediaPipe → features → modelo TF.js → RAW → smoothing
(F10) → SEÑA CONFIRMADA → `flow.js` → `history.js` + `speech.js`.

- `history.js`: una entrada `{label, confidence, timestamp}` por
  confirmación; **sin duplicados continuos** (mantener la seña no repite);
  **mano ausente = separación** → permite repetir la misma seña para
  construir frases ("HOY, GRACIAS"); máximo 50 (`HISTORY_CONFIG.maxEntries`).
- `speech.js`: `window.speechSynthesis` **local** (sin servidores de voz);
  prioridad de voz `es-CO → es-ES → es-*`; política anti-cola
  `replaceOnNew=true` (la nueva locución cancela la anterior); sin
  SpeechSynthesis → botón deshabilitado, la app sigue funcionando.
- La preferencia de voz y el toggle de persistencia se guardan en
  localStorage; el historial persiste solo si el usuario lo activa y
  guarda **únicamente etiquetas y horas** (nunca imágenes ni video).
- El modelo de prueba (no de 10 clases) **nunca se habla** como LSC: la
  voz y "Reproducir última seña" quedan bloqueados.

Prueba manual: `docs/prueba_manual_f11.md`.

### UI final del MVP (F12)

Interfaz de producto terminado, sin nuevas funcionalidades de IA:

- **Estructura:** HEADER (SIGNIA + chips compactos de estado + Diagnóstico
  `Alt+D`) → MAIN (Cámara | Traducción) → CONTROLES (voz, confianza,
  persistencia, reproducir última) → HISTORIAL. En móvil el orden es
  CÁMARA → TRADUCCIÓN → CONTROLES → HISTORIAL.
- **Predicción:** la seña confirmada domina (`HOLA` en grande + barra de
  confianza); placeholders honestos: "Coloca tu mano frente a la cámara",
  "Esperando una seña...", "Confianza insuficiente". La UI nunca inventa
  etiquetas.
- **Modelo:** tarjeta según estado — "Modelo no disponible" + Reintentar,
  badge "MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC", o
  "Modelo SIGNIA versión X" (sin declarar precisión).
- **Privacidad/honestidad:** footer con "Tu cámara se procesa localmente..."
  y "Reconocimiento LSC real: PENDIENTE DE VALIDACIÓN."
- **Diagnóstico** (RAW/SMOOTHED/FPS/shapes/versión...) oculto por defecto,
  abrible con `Alt+D` o el botón "Diagnóstico".
- **Accesibilidad:** `aria-live` solo anuncia la seña confirmada (nunca
  frames/RAW/FPS), `aria-pressed`, `focus-visible`, skip-link,
  `prefers-reduced-motion`, contraste alto.
- **Responsive:** 320 / 375 / 390 / 430 / 768 / 1024 / 1440 px, sin scroll
  horizontal, controles táctiles ≥ 48px.
- **Metadata:** title, description, theme-color, favicon.svg, Open Graph
  básico y `manifest.webmanifest` (preparación PWA — sin service worker).

Prueba manual: `docs/prueba_manual_f12.md`.

Sin modelo exportado la app muestra honestamente "Modelo no cargado"
y un badge de "MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC"
cuando el modelo presente no es el de 10 clases.
Prueba manual (webcam): `docs/prueba_manual_f9.md`.

| Fase | Descripción | Estado |
|------|-------------|--------|
| F0 | Documentación LSC (docs/señas_lsc.md) | Fichas con fuentes oficiales; validación de configuraciones pendiente |
| F1 | Estructura + entorno | COMPLETADA (venv + deps + shim Windows documentado) |
| F2 | Detector de manos | Código listo; verificación pendiente (sin cámara en el equipo de desarrollo) |
| F3 | Normalización (features.py) | COMPLETADA — tests |
| F4 | Recolector | Código listo (webcam + URL de celular); verificación pendiente (sin cámara) |
| F5 | Dataset + preprocess + split por persona | COMPLETADA en código; 18 tests de pipeline |
| F6 | Entrenamiento (train.py) | COMPLETADA en código — EarlyStopping, ModelCheckpoint best, artefacto de reproducibilidad |
| F7 | Evaluación (evaluate.py) | COMPLETADA en código — F1/confusión/PNG + etiqueta por persona vs fallback |
| F8 | Conversión TF.js + paridad | COMPLETADA en código — export_tfjs.py con metadata.json; espejo JS de normalización (features.js); paridad Python ↔ JS (parity.mjs) con fixtures compartidos |
| F9 | Frontend base (cámara + MediaPipe + modelo + estados) | COMPLETADA en código — app local sin CDN; verificación de cámara pendiente (sin webcam en el equipo) |
| F10 | Estabilidad temporal (smoothing) | COMPLETADA en código — EMA + N frames consecutivos; capa independiente, NO modifica el modelo |
| F11 | Historial + voz | COMPLETADA en código — historial de señas confirmadas, SpeechSynthesis local es-CO→es-ES, persistencia opcional; verificación de cámara/voz pendiente (sin webcam en el equipo) |
| F12 | UI final del MVP | COMPLETADA en código — jerarquía visual, responsive 320–1440px, accesibilidad (aria-live solo eventos, focus-visible, reduced-motion), diagnóstico oculto Alt+D, metadata/favicon/manifest PWA-prep; verificación visual pendiente (sin webcam) |
| F13 | Validación científica de las 10 señas + diseño del dataset | COMPLETADA en documentación — 10 clases VALIDADA con citas textuales del DBLSC (docs/clases_lsc.md), STATIC/DYNAMIC identificada (4 compatibles con el MLP, 6 requieren modelo temporal), docs/protocolo_dataset.md (personas, sesiones, splits por persona, consentimiento); sin recolección ni entrenamiento todavía |
| F14 | Infraestructura del dataset real (sin entrenar) | COMPLETADA en código — esquema por muestras (1 muestra = 1 ejecución con secuencia completa para dinámicas), recolector con sesiones A/B/C + descartes (D), QC que marca INVALID sin borrar, anti-leakage persona/ejecución, reportes (dataset_report.json + docs/piloto_dataset_report.md), consentimiento; 99 pytest + 119 Node en verde; recolección física del PILOTO (2 personas) pendiente del usuario |

> **Honestidad (requisito 26):** hasta disponer de un dataset real de LSC
> (recolectado con cámara), NO existen métricas de precisión de SIGNIA.
> `train.py` y `evaluate.py` se verifican con mini-datos sintéticos en
> carpetas temporales de pytest exclusivamente para comprobar el pipeline.

### Validación científica y diseño del dataset (F13)

La F13 no añade funcionalidades de IA ni recolecta datos: fija la **base
científica** del dataset real.

- **10 clases VALIDADA** contra el texto íntegro del **DBLSC** (INSOR /
  Instituto Caro y Cuervo, 2006; PDF oficial de colombiaaprende.edu.co).
  Tabla maestra con cita por clase (página impresa + página del PDF):
  `docs/clases_lsc.md`. Se corrigieron descripciones erróneas de la
  investigación preliminar (GRACIAS era DE-NADA; AYUDA era ¿QUIÉN?; AMIGO era
  la seña de ASL) — `docs/señas_lsc.md` actualizado.
- **STATIC vs DYNAMIC:** BIEN, MAL son `STATIC_COMPATIBLE_WITH_MLP` (2).
  HOLA, ADIOS, GRACIAS, POR_FAVOR, SI, NO, AYUDA, AMIGO son
  `DYNAMIC_REQUIRES_SEQUENCE` (8) — reconocimiento fiel requiere un modelo
  temporal (`TEMPORAL_MODEL_REQUIRED`); el MVP las captura en un **fotograma
  guía** documentado, con riesgo de confusión explícito. (HOLA y NO usan la
  variante canónica del signante: saludo informal con mano hacia la sien y
  mano en '1' oscilando, respectivamente.)
- **LSC ≠ ASL ≠ español signado:** verificado con el ejemplo de AMIGO (la
  versión de "círculos entrelazados" es ASL, no LSC).
- **Protocolo de dataset:** `docs/protocolo_dataset.md` — mínimo 5 personas
  (ideal 10+), IDs P01…, ~100 muestras por persona × clase (ejemplo, no
  óptimo asumido), 3 sesiones A/B/C en días distintos, variabilidad controlada
  (distancia, posición, orientación, iluminación, fondo, velocidad, tamaño),
  **splits por persona** (TRAIN/VALIDATION/TEST con personas disjuntas),
  validación lingüística por instructor sordo antes de cada persona,
  consentimiento y privacidad (los landmarks NO son anónimos), QC con
  `hand_count` esperado por clase.
- Sin recolección ni entrenamiento en F13: se ejecutará tras confirmación
  (F14 en adelante) cuando el usuario disponga de webcam y participantes.

Pruebas tras F13: 35 pytest + 108 Node (sin cambios de código).

### Infraestructura del dataset real (F14)

La F14 **NO entrena el modelo ni declara precisión**: construye la
infraestructura del dataset real y el flujo de recolección−QC−reporte.

- **Esquema por muestras** (`docs/dataset_schema.md`): 1 **muestra = 1
  ejecución** en `ai/dataset/raw/{P}/{SESION}/{CLASE}/{sample_id}/` con
  `metadata.json`, `guide.json` (fotograma guía, referencia del MVP) y
  `frames/frame_NNNN.json`. ID `P01_A_GRACIAS_0003` único e
  inmutable (nunca se sobrescribe). Las dinámicas (ADIOS, GRACIAS,
  POR_FAVOR, SI, AYUDA, AMIGO) guardan la **secuencia completa**;
  los frames **nunca** se usan como muestras independientes (leakage).
- **Recolector** (`ai/collection/collect_data.py`): `--person P01
  --session A --target 100 [--camera-source URL --source smartphone]`.
  Estáticas con ESPACIO (25 frames de retención), dinámicas con R
  (inicio/fin, auto-stop 8 s), descarte con D (mueve a `raw/_discarded/`
  con `issues=["operator_discard"]`). No guarda si la mano esperada por
  la clase no aparece. UI con PERSON/SESSION/CLASS/Samples/Hands/Quality.
- **QC** (`ai/preprocessing/validate_dataset.py`): chequeos F14 §10
  (clase, persona P\d+, sesión A/B/C, timestamp ISO, features 126-d sin
  NaN/Inf, rango <100, hand_count vs mano esperada por clase, dinámicas
  con ≥2 frames + sequence_id, frame_count == archivos, guide_frame en
  rango). Las fallas se **marcan `quality='invalid'` + `issues[]`** en el
  metadata (nunca se borran silenciosamente). Detecta huérfanos y
  desbalance >1.5:1; funciona con dataset vacío.
- **Anti-leakage** (`preprocess.py`): splits por persona completa
  (70/15/15 semilla fija), `verify_no_leakage()` verifica que cada muestra
  esté en un único split, cada persona en un único split y las secuencias
  viajen juntas.
- **Reportes** (`ai/preprocessing/report_dataset.py`): genera
  `ai/dataset/reports/dataset_report.json` (totales, %, válidas/inválidas/
  descartadas, static/dynamic, manos por guía, balance) y
  `docs/piloto_dataset_report.md`. Con dataset vacío NO fabrica datos
  (exit 1 con mensaje).
- **Privacidad** (`docs/consentimiento.md`): consentimiento informado
  antes de recolectar; los landmarks **NO son anónimos** (Ley 1581/2012);
  todo se almacena localmente.
- **Piloto primero**: 2 personas × 10 clases × 10 ejecuciones ≈ 200
  muestras; tras recolectarlo se valida (QC), se revisa el reporte y se
  corrigen problemas ANTES de dar por terminado F14.

Pruebas tras F14: 71 pytest (incluye ~36 nuevos de F14 con fixtures
sintéticos en temporales) + 108 Node. Recolección física pendiente del
usuario (webcam/celular + participantes).

## Cámaras: cómo usar SIGNIA

SIGNIA acepta dos tipos de fuente de video para la recolección y la
verificación: **webcam del computador** o **cámara del teléfono celular**.
En ambos casos el flujo es idéntico:

```
cámara → MediaPipe → landmarks → normalización → modelo → predicción
```

(En producción, la inferencia corre en el navegador con TensorFlow.js:
la cámara del navegador —desktop o móvil— alimenta directamente a
MediaPipe Hands del navegador.)

### A. Webcam del computador (recomendada para desarrollo)

Conecta la webcam y úsala con el índice de cámara local:

```powershell
python ai/collection/check_camera.py                    # cámara 0
python ai/collection/check_camera.py --camera 1         # si hay varias
python ai/collection/collect_data.py --person P01 --session A --target 100
```

### B. Cámara del teléfono como cámara del sistema

Sirve cuando el PC no tiene webcam. Dos opciones:

**B1. Streaming HTTP (recomendado para recolectar):** instala en el
celular la app **"IP Webcam" (Android)** o **"DroidCam"**. Conecta el
celular y el PC a la misma red WiFi y pasa la URL de video al recolector:

```powershell
# IP Webcam (Android): la app muestra la IP, p. ej. http://192.168.1.10:8080
python ai/collection/check_camera.py --camera http://192.168.1.10:8080/video
python ai/collection/collect_data.py --person P01 --session A --target 100 `
    --camera-source http://192.168.1.10:8080/video --source smartphone
```

> Espeja el video del celular (modo selfie) para que MediaPipe reconozca
> la mano derecha/izquierda correctamente; el recolector ya procesa en
> modo selfie. Marca `--source smartphone` para dejar la trazabilidad en
> el metadata (columna `source`).

**B2. Navegador móvil:** el frontend es responsive y funciona en el
navegador del celular; la cámara del dispositivo alimenta a MediaPipe Hands
y la inferencia TF.js corre en el mismo teléfono. El navegador exige un
contexto seguro: `http://localhost` sirve para probar en el mismo equipo,
pero desde un celular se necesita una URL `https://` (por ejemplo, hosting
HTTPS, un túnel HTTPS o `npx serve frontend --ssl`). No uses
`http://localhost` en el teléfono: allí `localhost` significa el propio
celular.

### Sin cámara disponible en el PC

El proyecto se desarrolla y verifica sin cámara:

- `check_camera.py` y `collect_data.py` están listos y se ejecutan apenas
  exista una cámara (local o por URL).
- `validate_dataset.py` y `preprocess.py` funcionan con dataset vacío y
  muestran claramente: **"Dataset vacío: primero debes recolectar
  muestras."**
- NO se generan datos sintéticos presentados como reales: el dataset real
  solo proviene de la cámara; los tests usan mini-datos sintéticos solo
  para comprobar el pipeline (requisito 27) y viven en carpetas temporales
  de pytest, nunca en `ai/dataset/`.

Los comandos de entrenamiento, evaluación, conversión TF.js y ejecución del
frontend se documentan en sus fases correspondientes (README completo).
