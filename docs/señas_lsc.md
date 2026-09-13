# FICHAS LSC — VALIDACIÓN DE SEÑAS

> **Regla del proyecto (requisito 2):** no se inventa ninguna seña. Cada clase
> necesita una fuente verificable (texto, imagen o video) antes de recolectar
> datos definitivos. Las clases sin verificación quedan **PENDIENTE_VALIDACION**
> y pueden reemplazarse.

## Fuentes oficiales y fiables (estado: confirmadas)

| # | Fuente | Tipo | Acceso |
|---|--------|------|--------|
| 1 | **Diccionario Básico de la Lengua de Señas Colombiana (DBLSC)** — INSOR / Instituto Caro y Cuervo, 2006 | PDF con descripciones escritas, fotos y videos | https://www.colombiaaprende.edu.co/sites/default/files/files_public/2022-04/Diccionario-lengua-de-senas.pdf (también: https://lenguasyliteraturasnativas.caroycuervo.gov.co/publicaciones/2023/11/Diccionario_lengua_de_senas_colombiana.pdf) |
| 2 | **INSOR Educativo — Diccionario y Repositorio Virtual Bilingüe LSC–Español** | Diccionario en línea con videos | https://educativo.insor.gov.co/diccionario/ |
| 3 | **FENASCOL** (Federación Nacional de Sordos de Colombia) | Organización sorda nacional | https://www.fenascol.org.co |
| 4 | **Estudio científico colombiano**: "Reconocimiento de lengua de señas colombiana mediante redes neuronales convolucionales y captura de movimiento" (2022) — utilizó las señas Hola, Gracias, Bien, Mal, No, Sí, Por favor, ¿Cómo estás?, Bienvenido, Con mucho gusto | Artículo SciELO | http://www.scielo.org.co/scielo.php?pid=S0123-921X2022000400070&script=sci_arttext |
| 5 | Video: "Greetings and Courtesies in Colombian Sign Language" (saludos y cortesías en LSC) | Video | https://www.youtube.com/watch?v=MJZBTY_kXtI |

## Aviso sobre el alcance del MVP

El MVP clasifica **un frame estático** (configuración de la mano: forma de los
dedos, orientación, uso de 1 o 2 manos) extraído de 21 landmarks por mano.

Esto significa:

1. Las señas **esencialmente dinámicas** (trayectoria, como DECIR-ADIÓS que
   mueve la mano hacia los lados) se reconocerán por su *configuración en el
   instante de la captura*. El modelo puede confundirlas; eso se mitiga
   recolectando siempre el mismo punto de la trayectoria (fotograma guía) y con
   el suavizado temporal.
2. Las señas que dependen de **posición corporal** (tocar la frente, el mentón)
   no pueden distinguirse con landmarks de mano normalizados (tras la
   normalización la mano queda centrada en su muñeca y se pierde la ubicación
   respecto al cuerpo). Las clases afectadas se marcan como ambigüedad conocida.
3. Las señas **no manuales** (mover la cabeza, gestos faciales) NO son
   reconocibles en este MVP.
4. La normalización es invariante a escala y traslación pero **no a rotación**:
   la orientación de la mano respecto a la cámara sí es información útil para
   el modelo.

## Fichas por clase

Estado: `VALIDADA` (fuente con descripción textual verificada) |
`PENDIENTE_VALIDACION` (hay indicios de uso en fuentes académicas o el
diccionario, pero falta confirmar la configuración exacta con un video/insor
antes de recolectar el dataset definitivo).

> **Actualización F13:** las 10 clases quedaron **VALIDADA** contra el texto
> íntegro del DBLSC (PDF oficial, citas en docs/clases_lsc.md, incluye
> corrección de errores de GRACIAS, AYUDA y AMIGO). La tabla maestra y la
> clasificación STATIC/DYNAMIC viven en **docs/clases_lsc.md** (canónico).

### HOLA
- **Estado:** VALIDADA (F13 — cita DBLSC completa); **variante canónica
  actualizada (decisión del signante)**.
- **Fuente:** DBLSC (Fórmulas de cortesía: DECIR-HOLA, impreso 489 / PDF 541). INSOR Educativo — Diccionario.
- **Descripción (DBLSC):** "La mano en '5' se coloca con la palma hacia adentro.
  Las cejas se elevan y se vocaliza la palabra. La mano siempre se dirige hacia
  el lugar del interlocutor."
- **Variante canónica para el dataset (saludo informal, decisión del signante):**
  "Forma Estándar (Saludo Informal): posición: mano dominante abierta con los
  dedos juntos y extendidos; movimiento: llevar la mano hacia un lado de la
  frente o sien (similar a un saludo militar, pero mucho más suave y relajado)
  y realizar un ligero movimiento hacia afuera o hacia adelante, despegándola
  de la cabeza; expresión: acompañarla siempre con una sonrisa y contacto
  visual directo."
- **Manos:** 1 (mano derecha o dominante).
- **Relevante para el modelo:** configuración de dedos juntos y extendidos
  hacia la sien; el despegue hacia afuera/adelante es movimiento observable en
  landmarks → `DYNAMIC_REQUIRES_SEQUENCE`.
- **Ambigüedad conocida:** la variante DBLSC (palma hacia adentro, estática)
  también aparece en la forma neutra de varias señas de cortesía; la
  vocalización y la sonrisa no son capturables. El dataset usa la variante
  informal canónica.

### ADIOS
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (Fórmulas de cortesía: DECIR-ADIÓS, impreso 489 / PDF 541).
- **Descripción (DBLSC):** "(Gesto natural) La mano en '5' con la palma hacia
  adelante y la punta de los dedos hacia arriba se mueve hacia los lados."
- **Manos:** 1.
- **Relevante para el modelo:** dedos extendidos, palma hacia adelante
  (opuesto de HOLA: palma adentro → riesgo de confusión HOLA/ADIOS por
  orientación; se mitiga con fotograma guía consistente).
- **Ambigüedad conocida:** seña dinámica (movimiento lateral). Con un único
  frame puede confundirse con HOLA si el fotograma guía no es consistente.

### GRACIAS
- **Estado:** VALIDADA (F13 — cita DBLSC completa; CORREGIDA)
- **Fuente:** DBLSC (Fórmulas de cortesía: AGRADECER/DECIR-GRACIAS/GRACIAS-A, impreso 485 / PDF 537).
- **Descripción (DBLSC):** "La mano en '5' con los dedos juntos excepto el
  pulgar y la palma hacia atrás, toca el mentón con las yemas de los dedos e
  inmediatamente se dirige hacia la palma de la mano contraria, hasta hacer
  contacto con ella. La cabeza se inclina."
- **Manos:** 2 (la mano dominante contacta la palma de la otra).
- **Relevante para el modelo:** configuración '5' dedos juntos salvo pulgar;
  fotograma guía: el contacto entre la mano y la palma contraria.
- **Ambigüedad conocida:** seña dinámica (mentón → palma); la descripción
  preliminar (palma sobre el pecho hacia adelante) era DE-NADA/CON-MUCHO-GUSTO
  (impreso 489), NO GRACIAS — corregida. La inclinación de cabeza no es
  capturable. `DYNAMIC_REQUIRES_SEQUENCE`.

### POR_FAVOR
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (Fórmulas de cortesía: POR-FAVOR, impreso 491 / PDF 543) +
  est. científico SciELO 2022 (clase "Por favor" entrenada exitosamente).
- **Descripción (DBLSC):** "La mano cerrada con la palma hacia atrás describe
  círculos sobre el pecho."
- **Manos:** 1.
- **Relevante para el modelo:** puño cerrado (configuración capturable);
  fotograma guía: cualquier instante del círculo (puño sobre el pecho).
- **Ambigüedad conocida:** seña dinámica sobre el **pecho** — la posición
  corporal se pierde tras la normalización por muñeca; riesgo de confusión con
  SI (también puño, orientación de palma opuesta). `DYNAMIC_REQUIRES_SEQUENCE`.

### SI
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (SÍ, impreso 458 / PDF 510) + est. científico SciELO 2022
  (clase "Sí" entrenada con éxito).
- **Descripción (DBLSC):** "La mano cerrada con la palma hacia adelante se
  flexiona hacia abajo varias veces." (la variante 2 de la entrada es cabeceo,
  no manual; la variante manual es la 1).
- **Manos:** 1.
- **Relevante para el modelo:** puño cerrado palma hacia adelante; fotograma
  guía: cualquier instante de la flexión (la configuración no cambia).
- **Ambigüedad conocida:** seña dinámica repetitiva; riesgo de confusión con
  POR_FAVOR (misma configuración, orientación de palma opuesta);
  `DYNAMIC_REQUIRES_SEQUENCE`.

### NO
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (NO, impreso 452 / PDF 504) + est. científico SciELO 2022
  (clase "No" entrenada con éxito).
- **Descripción (DBLSC):** entrada con 3 variantes: (1) "La mano en 'V', pero
  con el pulgar enfrentado a los dos dedos y las puntas de estos hacia
  adelante, une las yemas de los tres dedos"; (2) "La mano en '1' se mueve
  repetidamente de un lado a otro"; (3) cabeceo (no manual).
- **Manos:** 1.
- **Relevante para el modelo:** **Canónica para el dataset: variante 2** (mano
  en '1' moviéndose repetidamente de lado a lado; configuración constante,
  capturable como secuencia) — decisión del signante. La variante 1 (pinza de 3
  dedos) y la 3 (cabeceo) no se recolectan en el MVP.
- **Ambigüedad conocida:** variante 1 es estática y la 3 no manual; si un
  signante produce la variante 1, se descarta (tecla D) y se repite con la
  variante 2 canónica.

### AYUDA
- **Estado:** VALIDADA (F13 — cita DBLSC completa; CORREGIDA)
- **Fuente:** DBLSC (Sociedad → ACCIONES Y DISTRACCIONES: AYUDAR/AYUDA, impreso 243 / PDF 295).
- **Descripción (DBLSC):** "La mano cerrada con el borde externo hacia abajo se
  coloca sobre la palma de la mano contraria y luego ambas se mueven hacia
  adelante."
- **Manos:** 2.
- **Relevante para el modelo:** puño sobre la palma de la otra mano (contacto
  de 2 manos); fotograma guía: el contacto antes del avance.
- **Ambigüedad conocida:** la descripción preliminar ("mano B golpea el
  mentón") era de **¿QUIÉN?** (impreso 483 / PDF 535) — el fragmento web
  mezcló la oración de ejemplo que contenía AYUDAR. Seña dinámica;
  `DYNAMIC_REQUIRES_SEQUENCE`.

### BIEN
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (Sociedad → LENGUAJE Y COMUNICACIÓN, Calificar: BIEN,
  impreso 473 / PDF 525) + est. científico SciELO 2022 (clase "Bien" entrenada
  con éxito).
- **Descripción (DBLSC):** "(Gesto natural) La mano cerrada con el pulgar
  extendido se coloca con el borde externo hacia abajo y la yema del pulgar
  hacia adelante. Puede acompañarse de un movimiento corto de arriba hacia
  abajo."
- **Manos:** 1.
- **Relevante para el modelo:** puño con pulgar extendido hacia arriba;
  configuración estática → `STATIC_COMPATIBLE_WITH_MLP`. El movimiento corto es
  opcional.
- **Ambigüedad conocida:** espejo de MAL (pulgar abajo); separable por la
  dirección de la punta del pulgar.

### MAL
- **Estado:** VALIDADA (F13 — cita DBLSC completa)
- **Fuente:** DBLSC (Sociedad → LENGUAJE Y COMUNICACIÓN, Calificar: MAL,
  impreso 475 / PDF 527) + est. científico SciELO 2022 (clase "Mal" entrenada
  con éxito).
- **Descripción (DBLSC):** "(Gesto natural) La mano cerrada con el pulgar
  extendido y la punta hacia abajo se coloca con la palma hacia adelante.
  Puede acompañarse de movimientos cortos de arriba hacia abajo."
- **Manos:** 1.
- **Relevante para el modelo:** puño con pulgar extendido hacia abajo;
  configuración estática → `STATIC_COMPATIBLE_WITH_MLP` (espejo de BIEN).
- **Ambigüedad conocida:** espejo de BIEN; mitigada por la dirección del pulgar.

### AMIGO
- **Estado:** VALIDADA (F13 — cita DBLSC completa; CORREGIDA)
- **Fuente:** DBLSC (relaciones de parentesco y afinidad: AMIGO, impreso 206 /
  PDF 258, entrada completa extraída).
- **Descripción (DBLSC):** "Las manos en '1', una con la palma hacia arriba y
  la otra hacia abajo, se acercan hasta que los índices se tocan por las yemas.
  Enseguida las manos giran sobre las muñecas hasta quedar en la posición
  inversa y nuevamente las yemas se tocan."
- **Manos:** 2.
- **Relevante para el modelo:** dos manos en '1' con yemas de índices en
  contacto; fotograma guía: cualquiera de los instantes de contacto.
- **Ambigüedad conocida:** la descripción preliminar ("círculos de índice y
  pulgar entrelazados") era la seña **ASL**, no LSC — corregida. Seña dinámica
  (rotación de muñecas); `DYNAMIC_REQUIRES_SEQUENCE`.

## Procedimiento de validación (responsable: usuario)

1. Abrir https://educativo.insor.gov.co/diccionario/ (o el PDF del DBLSC) y
   buscar cada clase de la lista.
2. Registrar en la ficha: configuración de la mano (forma de dedos), orientación
   de la palma, uso de 1 o 2 manos, trayectoria, fotograma guía recomendado.
3. Si una seña es puramente no manual o depende de posición corporal no
   observable, marcarla `NO_RECONOCIBLE` y reemplazarla por otra clase
   documentada.
4. Actualizar `ai/training/config.py` (CLASES) con la lista final validada.
