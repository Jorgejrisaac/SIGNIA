# PROTOCOLO DE RECOLECCIÓN DEL DATASET LSC (F13)

> Documento de diseño previo a la recolección. Define CÓMO, QUIÉN y CON QUÉ
> criterios se capturará el dataset real de SIGNIA. Se ejecutará solo cuando el
> usuario lo confirme y disponga de webcam/participantes (F14 en adelante).
>
> **No se recolecta en F13.** Este documento fija la especificación; ninguna
> línea de código cambia por F13.

## 1. Objetivo

Construir un dataset de **landmarks de manos (126 features por muestra)** para
las 10 señas LSC validadas en `docs/clases_lsc.md`, con:

- **Separación por persona** estricta (TRAIN / VALIDATION / TEST con personas
  distintas — sin filtración entre personas).
- Variabilidad realista documentada y controlada.
- Consentimiento y privacidad explícitos.

## 2. Clases y esquema de muestra

| ID | Etiqueta | Manos esperadas (hand_count) | Fotograma guía (punto de la trayectoria a capturar) |
|----|----------|------------------------------|-----------------------------------------------------|
| C01 | HOLA | 1 | Saludo informal: mano abierta dedos juntos hacia la sien/frente, instante del despegue (variante canónica del signante) |
| C02 | ADIOS | 1 | Mano '5' palma hacia adelante (palma visible a la cámara) |
| C03 | GRACIAS | 2 | Contacto de la mano con la palma de la mano contraria |
| C04 | POR_FAVOR | 1 | Puño cerrado sobre el pecho |
| C05 | SI | 1 | Puño cerrado, palma hacia la cámara |
| C06 | NO | 1 | Mano en '1' moviéndose repetidamente de lado a lado (variante 2 canónica) |
| C07 | AYUDA | 2 | Puño apoyado sobre la palma de la otra mano (antes del avance) |
| C08 | BIEN | 1 | Puño con pulgar extendido hacia arriba |
| C09 | MAL | 1 | Puño con pulgar extendido hacia abajo |
| C10 | AMIGO | 2 | Yemas de los índices en contacto (cualquier instante de contacto) |

Reglas:

- Cada muestra = **1 frame** guardado como `.npy` (126-d normalizado) + fila en
  `metadata.csv`. La normalización (muñeca = origen, escala = palma) es
  idéntica en recolección, entrenamiento e inferencia
  (`ai/preprocessing/features.py` y `ai/training/config.py`).
- El participante ejecuta la seña hasta el fotograma guía y el recolector
  guarda en el instante correcto. Para las 7 dinámicas
  (`DYNAMIC_REQUIRES_SEQUENCE`: ADIOS, GRACIAS, POR_FAVOR, SI, NO, AYUDA,
  AMIGO) este fotograma es el ancla de consistencia del dataset.
- `hand_count` esperado (1 o 2) se valida automáticamente en el control de
  calidad (QC): muestras con count ≠ esperado se descartan con la tecla `D`
  del recolector (`ai/collection/collect_data.py`).

## 3. Participantes

| Parámetro | Mínimo | Ideal | Nota |
|-----------|--------|-------|------|
| Personas totales | **5** | **10 o más** | Más personas = mejor generalización |
| Usuarios nativos de LSC (sordos) | 1 | ≥ 50 % del grupo | La validación lingüística la hace un instructor sordo o intérprete certificado (paso 0) |
| Mano dominante | — | ~10–20 % zurdos | Solo si hay zurdos disponibles; ver §7 (decisión de espejo) |
| Edades / contexto | — | Mezcla (p. ej. 3 rangos) | Variabilidad demográfica documentada |

- **IDs:** `P01`, `P02`, …, asignados en orden de incorporación (es el
  `person_id` de `metadata.csv` y el parámetro `--person` del recolector).
- **Historial por persona** (se registra en `docs/recoleccion/participantes.md`
  cuando exista): edad (rango), mano dominante, experiencia LSC, sesiones
  realizadas, incidencias.

### Paso 0 — Validación lingüística (obligatorio antes de cada persona)

1. Un **instructor sordo / intérprete certificado de LSC** revisa las 10 fichas
   de `docs/clases_lsc.md` contra la seña real (video INSOR Educativo:
   https://educativo.insor.gov.co/diccionario/).
2. Si una ficha difiere de la producción real, se actualiza
   `docs/clases_lsc.md` (nuevo estado) **antes de recolectar esa clase**.
3. El instructor graba (o supervisa) el **video guía de cada clase** usado en
   la sesión (referencia visual para el participante y para el QC).

## 4. Muestras por clase y persona

- **Objetivo inicial: 100 muestras por persona × clase** (ejemplo de
  referencia, ver abajo). Con 10 personas: **1.000 muestras por clase**,
  10.000 en total.
- **NO se asume que 100 sea óptimo.** Es un punto de partida; el tamaño final
  se decide con las curvas de aprendizaje del primer modelo (F14+). El objetivo
  configurado en `config.py` (`TARGET_SAMPLES_PER_CLASS = 500`) es un mínimo
  por clase acorde al requisito 8; si el modelo ya converge con menos, no se
  malgasta tiempo de participantes.
- Estructura por sesión (ver §5): 3 sesiones (A/B/C) × ~34 muestras por
  persona × clase = ~102; o 2 sesiones × 50 = 100. Se permite ±10 %.

## 5. Sesiones y descansos

- **Sesiones A, B, C en días distintos** (mín. 24–48 h entre sesiones) para
  variar postura, ropa, luz ambiente y estado físico del participante.
- Una sesión = 10 clases × ~34 muestras ≈ 340 capturas ≈ **45–60 minutos**
  efectivos.
- **Descansos obligatorios:** 5 min cada 2 clases; pausa de 10 min entre la
  clase 5 y la 6; se sugiere no superar 60 min por sesión (fatiga = variación
  no deseada del gesto).
- Orden de clases por sesión: aleatorio fijo (mismo orden para todos los
  participantes en una sesión dada, cambiando entre sesiones) para evitar
  sesgos de fatiga por clase.

## 6. Condiciones y variabilidad controladas

| Factor | Configuración |
|--------|---------------|
| Distancia a cámara | **0,6 – 1,2 m** (variar entre muestras: ≈⅓ corta, ≈⅓ media, ≈⅓ larga) |
| Posición en el cuadro | Mano ± 20 % del centro: centro, arriba, abajo, izquierda, derecha (rotación de la cámara o desplazamiento del cuerpo) |
| Orientación | Rotación natural de la muñeca (dentro del rango del gesto); NO rotaciones extremas que deformen la seña |
| Iluminación | ≥ 2 escenarios por sesión: luz frontal (ventana/panel) y luz lateral (lámpara), sin contraluz directo |
| Fondo | Liso claro u oscuro, sin objetos movibles; el participante con ropa de contraste variable entre sesiones |
| Velocidad | Gestos a velocidad NATURAL (la mayoría) + ~20 % más lentos; descartar gestos atropellados que pierdan tracking |
| Tamaño aparente | Cubrir zoom (acercar/alejar cámara) se obtiene variando la distancia; no usar zoom digital |

La cámara se fija en trípode; el participante **no sostiene el dispositivo**.

## 7. Mano dominante y espejo

- Las señas de 1 mano usan la **mano dominante** del participante.
- El recolector guarda los landmarks sin modificar y etiqueta los bloques
  izquierda/derecha con la handedness de MediaPipe Tasks Vision sobre el
  frame **sin espejar** (la misma convención que la inferencia del frontend;
  ver `ai/collection/hand_tracker.py`). Decisión de diseño (se documenta en
  el informe): si hay ≥ 2 zurdos, se mantienen como variabilidad real; si hay
  1 solo, se decide al analizar el primer modelo (riesgo: confusión por
  espejo). En el MVP se recomienda **excluir zurdos o duplicarlos con espejo
  explícito** en preprocesado (nunca en silencio).

## 8. Nomenclatura, errores y descartes

- Archivos: `{RAW_DIR}/{persona}/{clase}/{persona}_{clase}_{sesion}_{rep}.npy`
  (p. ej. `P03/HOLA/P03_HOLA_B_21.npy`); `metadata.csv` con
  `sample_id,class,person_id,hand_count,timestamp,source,split` (generado por
  `collect_data.py`).
- `source`: `sesion-A|B|C`. `split`: se asigna en preprocesado, **no en
  recolección** (ver §9).
- **Errores típicos y acción:**
  - Tracking perdido o mano parcialmente fuera de cuadro → tecla `D`
    (descartar) y repetir.
  - `hand_count` ≠ esperado para la clase → descartar.
  - Gesto incorrecto (variante NO-1, olvido del fotograma guía) → el asistente
    descarta y el participante repite con el video guía visible.
  - Parpadeos de luz o sombra dura → revisar y repetir el bloque.
- **QC por bloque (obligatorio):** al terminar cada clase, el asistente revisa
  una muestra de cada ~10 guardadas; al terminar la persona, el instructor
  revisa la sesión completa y firma la hoja de registro
  (`docs/recoleccion/registro_*.md`).

## 9. Splits por persona (obligatorio, sin filtración)

- **Regla:** una persona completa va a SOLO un split. Nunca se reparten
  muestras de la misma persona entre TRAIN y TEST.
- Con **10 personas** (propuesta):
  - TRAIN: 6 personas (60 %) → ~6.000 muestras (600 por clase)
  - VALIDATION: 2 personas (20 %) → ~2.000 (200 por clase)
  - TEST: 2 personas (20 %) → ~2.000 (200 por clase)
- Con **5 personas** (mínimo viable): TRAIN 3, VALIDATION 1, TEST 1
  (60/20/20 %) — se advierte que TEST con 1 persona es poco representativo.
- Implementación: al fijar los IDs, `config.py` se actualiza con
  `TEST_PERSONS` y `VALIDATION_PERSONS` (las listas HOY vacías se llenan en la
  fase de recolección; el resto por defecto va a TRAIN). El split por persona
  se verifica en `ai/preprocessing/validate_dataset.py`.
- El informe de entrenamiento debe reportar métricas **por persona de TEST**
  (no solo agregadas) para detectar personas difíciles.

## 10. Consentimiento y privacidad

- **Los landmarks NO son anónimos por sí solos:** la geometría de la mano
  (forma, tamaño) y patrones de movimiento pueden asociarse a una persona. Se
  trata como dato personal (protección de datos, Ley 1581 de 2012, Colombia).
- **Documento de consentimiento informado obligatorio** (se firmará antes de
  la primera sesión; borrador en `docs/consentimiento.md`):
  - Qué se captura (landmarks de manos, no video/audio; excepción: video guía
    con autorización separada).
  - Finalidad (entrenar el modelo local de SIGNIA), almacenamiento local
    (`ai/`, sin nube), no publicación, no transferencia a terceros.
  - Derecho a retirarse y borrar sus datos en cualquier momento (procedimiento
    de borrado: eliminar sus archivos + filas y **reentrenar sin sus datos**).
  - Menores de edad: firma del acudiente/nombre del menor (en Colombia,
    principado de consentimiento; las instituciones educativas pueden
    requerir autorización institucional adicional).
- El video guía solo se conserva con autorización explícita y se borra al
  terminar la persona (o se guarda con nombre ficticio y restricciones).
- Recolección: el usuario confirma que dispone de webcam y participantes antes
  de F14; si no hay participantes, NO se recolecta y el dataset se diseña
  (este documento) sin ejecutarse.

## 11. Lo que NO se hace con este protocolo

- NO se recolecta ni se entrena en F13 (esperar confirmación).
- NO se cambia el formato de 126 features ni las rutas de `config.py`.
- NO se añaden clases fuera de la lista validada de `clases_lsc.md`.
- NO se promete reconocimiento de señas dinámicas por el MLP estático: las 8
  dinámicas quedan marcadas `DYNAMIC_REQUIRES_SEQUENCE` y el modelo temporal es
  una evolución futura explícita.
- NO se comparte ningún dato fuera del equipo del proyecto.