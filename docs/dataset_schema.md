# ESQUEMA DE DATOS DEL DATASET REAL (F14)

> Especificación técnica del formato de `ai/dataset/`. Complementa a
> `docs/protocolo_dataset.md` (cómo se recolecta) y `docs/clases_lsc.md`
> (qué se recolecta).

## 1. Principios

1. **1 muestra = 1 ejecución.** Para las 8 señas dinámicas (HOLA, ADIOS,
   GRACIAS, POR_FAVOR, SI, NO, AYUDA, AMIGO) la ejecución conserva la
   **secuencia completa** de fotogramas. Las estáticas (BIEN, MAL) conservan
   la retención breve de la pose.
2. **Trazabilidad total:** cada muestra tiene un `sample_id` único y su
   `metadata.json`; nada se sobrescribe ni se borra en silencio.
3. **Los landmarks NO son anónimos** (ver `docs/consentimiento.md`).
4. **Los frames nunca se convierten en muestras independientes** (F14 §18):
   una ejecución permanece junta y una persona permanece dentro de un solo
   split.

## 2. Estructura de directorios

```
ai/dataset/
├── raw/
│   ├── P01/
│   │   ├── A/            # sesión A
│   │   │   ├── HOLA/
│   │   │   │   └── P01_A_HOLA_0001/
│   │   │   │       ├── metadata.json
│   │   │   │       ├── guide.json
│   │   │   │       └── frames/
│   │   │   │           ├── frame_0001.json
│   │   │   │           └── ...
│   │   │   ├── ADIOS/   # dinámicas: secuencia completa
│   │   │   └── ...
│   │   ├── B/
│   │   └── C/
│   ├── P02/
│   └── ...
│   └── _discarded/      # tecla D del recolector (quality='invalid',
│                        #  issue='operator_discard'); se CUENTAN, no se borran
├── processed/           # features.npy, labels.npy, splits.json (F15)
├── metadata/            # (legacy pre-F14: metadata.csv ya NO se escribe)
└── reports/
    └── dataset_report.json   # estadísticas reales (F14 §23)
```

## 3. sample_id

`{PERSONA}_{SESION}_{CLASE}_{n:04d}` — p. ej. `P01_A_GRACIAS_0003`.

- El número es el **contador máximo existente + 1** (nunca sobrescribe).
- Valida el patrón `P\d+_[ABC]_[A-Z0-9_]+_\d{4}`.
- Persona: `P01`, `P02`, ... (IDs anónimos; no nombres reales en el dataset).

## 4. metadata.json (por muestra)

```json
{
  "sample_id": "P01_A_HOLA_0001",
  "person_id": "P01",
  "session_id": "A",
  "class": "HOLA",
  "sign_type": "static",
  "timestamp": "2026-08-13T10:30:00.123456",
  "source": "smartphone",
  "hand_count": 1,
  "hand_count_max": 1,
  "frame_count": 25,
  "duration_ms": 833,
  "guide_frame": 13,
  "quality": "valid",
  "issues": []
}
```

Campos adicionales en dinámicas: `"sequence_id"` (== sample_id; en este
esquema 1 muestra = 1 secuencia).

- `hand_count`: manos en el **fotograma guía** (debe coincidir con
  `EXPECTED_HANDS[clase]`).
- `hand_count_max`: máximo observado durante la captura.
- `quality`: `valid` | `invalid` (lo fija el validador, nunca borra).
- `issues`: lista de problemas detectados (vacía si valid).

## 5. guide.json (fotograma guía)

```json
{
  "frame_index": 13,
  "features": [ ...126 floats... ],
  "hands": ["Left"],
  "hand_count": 1,
  "t_ms": 480,
  "note": "Fotograma guía: referencia del MVP estático; no representa la seña dinámica completa."
}
```

- **Es una REFERENCIA del MVP** (el MLP entrena con UN vector por ejecución).
- **NO es la representación completa** de una seña dinámica; la secuencia
  completa vive en `frames/` para el modelo temporal futuro.
- Selección determinista (`select_guide_frame`): entre los fotogramas con el
  número de manos esperado, el más cercano a la mediana de la configuración;
  si ninguno lo cumple, el de mayor número de manos (el recolector solo
  guarda muestras que sí alcanzaron el número esperado).

## 6. frames/frame_XXXX.json

```json
{
  "features": [ ...126 floats normalizados... ],
  "hands": ["Left", "Right"],
  "hand_count": 2,
  "t_ms": 120
}
```

- `features`: vector 126-d idéntico al de `features.py`/`features.js`
  (normalización por muñeca + palma; bloques izquierda/derecha).
- `t_ms`: milisegundos desde el inicio de la sesión del recolector
  (monotónicos) → permite calcular `duration_ms` y velocidad de la trayectoria.

## 7. Control de calidad (F14 §10)

`ai/preprocessing/validate.py` → `validate_sample()` aplica por muestra:

1. metadata.json presente y JSON válido; campos obligatorios presentes.
2. Clase ∈ CLASES (10 validadas F13); persona `P\d+`; sesión ∈ {A,B,C};
   timestamp ISO válido; sign_type consistente con la clase.
3. `sample_id` coherente con persona/sesión/clase.
4. guide.json presente; features 126-d, finitos (sin NaN/Inf), rango
   razonable (< 100 tras normalización).
5. `hand_count` del guía == manos esperadas de la clase
   (`EXPECTED_HANDS`), p. ej. GRACIAS → 2.
6. Dinámicas: ≥ 2 fotogramas (secuencia real). Estáticas: ≥ 1.
7. `frame_count` == número real de archivos; numeración secuencial sin
   huecos (`frame_0001.json` ... `frame_NNNN.json`).
8. `duration_ms` ≥ 0; `guide_frame` dentro del rango; dinámicas con
   `sequence_id`.

Si algo falla → `quality="invalid"` + `issues[]` persistidos; **la muestra no
se borra** y aparece en el reporte.

## 8. Descarte por operador (tecla D)

La muestra se **mueve** a `raw/_discarded/{persona}/{sesion}/{clase}/...` y su
metadata queda `quality="invalid", issues=["operator_discard"]`. El validador
la cuenta como "descartada" (no como inválida).

## 9. Split por persona y anti-leakage (F14 §17–18)

- El split asigna **personas completas** (config `TEST_PERSONS` /
  `VALIDATION_PERSONS` o reparto 70/15/15 con seed fija).
- `verify_no_leakage()` comprueba: cada muestra en exactamente un split;
  cada persona en exactamente un split; las secuencias agrupadas permanecen
  juntas.
- El preprocesado genera **UN vector por ejecución** (el del guía): los
  frames de una dinámica nunca se reparten entre splits.

## 10. Reportes

- `ai/preprocessing/validate_dataset.py` — estado del dataset (CLI, exit 0/1).
- `ai/preprocessing/report_dataset.py` — escribe
  `ai/dataset/reports/dataset_report.json` + `docs/piloto_dataset_report.md`
  SOLO con datos reales; con dataset vacío no fabrica estadísticas.
