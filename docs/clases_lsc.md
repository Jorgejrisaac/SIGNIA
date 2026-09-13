# CLASES LSC — VALIDACIÓN CIENTÍFICA Y DISEÑO DEL DATASET (F13)

> **Regla del proyecto (requisito 2):** no se inventa ninguna seña. Cada clase
> necesita una fuente verificable antes de recolectar datos definitivos.
>
> Este documento es el resultado de la **Fase 13**: las 10 clases fueron
> verificadas contra el texto íntegro del **Diccionario Básico de la Lengua de
> Señas Colombiana (DBLSC, INSOR / Instituto Caro y Cuervo, 2006)**. Las
> citas textuales se extrajeron del PDF oficial
> (https://www.colombiaaprende.edu.co/sites/default/files/files_public/2022-04/Diccionario-lengua-de-senas.pdf),
> 632 páginas. Cada cita indica **página impresa del libro y página del PDF**
> (PDF = impresa + 52).

## 1. Tabla maestra de clases (estado: VALIDADA)

| ID | Etiqueta (config.py) | Significado | Tipo | Manos | Movimiento | Fuente | Estado |
|----|----------------------|-------------|------|-------|------------|--------|--------|
| C01 | HOLA | Saludo ("decir hola") | locución | 1 | DBLSC: "La mano en '5' se coloca con la palma hacia adentro... La mano siempre se dirige hacia el lugar del interlocutor" (no manual: cejas elevadas, vocalización). **Canónica para el MVP: saludo informal del signante** — mano abierta con dedos juntos y extendidos hacia la sien/frente, con ligero despegue hacia afuera/adelante (movimiento suave, tipo saludo militar relajado); acompañada de sonrisa y contacto visual | DBLSC "DECIR-HOLA", impreso 489 (PDF 541) | VALIDADA (variante actualizada por el signante) |
| C02 | ADIOS | Despedida ("decir adiós") | locución | 1 | "La mano en '5' con la palma hacia adelante y la punta de los dedos hacia arriba **se mueve hacia los lados**" | DBLSC "DECIR-ADIÓS", impreso 489 (PDF 541) | VALIDADA |
| C03 | GRACIAS | Agradecer / gracias | locución | 2 | "La mano en '5' con los dedos juntos excepto el pulgar y la palma hacia atrás, toca el mentón con las yemas de los dedos e inmediatamente se dirige hacia la palma de la mano contraria, hasta hacer contacto con ella. La cabeza se inclina" | DBLSC "AGRADECER/DECIR GRACIAS/GRACIAS A", impreso 485 (PDF 537) | VALIDADA |
| C04 | POR_FAVOR | Solicitud cortés ("por favor") | locución | 1 | "La mano cerrada con la palma hacia atrás **describe círculos sobre el pecho**" | DBLSC "POR-FAVOR", impreso 491 (PDF 543) | VALIDADA |
| C05 | SI | Afirmación ("sí") | adverbio | 1 | "La mano cerrada con la palma hacia adelante **se flexiona hacia abajo varias veces**" (variante 2 de la entrada: cabeceo, no manual) | DBLSC "SÍ", impreso 458 (PDF 510) | VALIDADA |
| C06 | NO | Negación ("no") | adverbio | 1 | Tres variantes documentadas en la entrada: (1) "La mano en 'V', pero con el pulgar enfrentado a los dos dedos... une las yemas de los tres dedos" (estática); (2) "La mano en '1' se mueve repetidamente de un lado a otro"; (3) cabeceo (no manual). **Canónica para el MVP: variante 2 (dinámica) — decisión del signante** | DBLSC "NO", impreso 452 (PDF 504) | VALIDADA |
| C07 | AYUDA | Ayudar / ayuda | verbo/nombre | 2 | "La mano cerrada con el borde externo hacia abajo se coloca sobre la palma de la mano contraria y luego ambas se mueven hacia adelante" | DBLSC "AYUDAR/AYUDA", impreso 243 (PDF 295) | VALIDADA |
| C08 | BIEN | Bien, correctamente | adverbio | 1 | "La mano cerrada con el pulgar extendido se coloca con el borde externo hacia abajo y la yema del pulgar hacia adelante. Puede acompañarse de un movimiento corto de arriba hacia abajo" | DBLSC "BIEN", impreso 473 (PDF 525) | VALIDADA |
| C09 | MAL | Mal, de modo imperfecto | adverbio | 1 | "La mano cerrada con el pulgar extendido y la punta hacia abajo se coloca con la palma hacia adelante. Puede acompañarse de movimientos cortos de arriba hacia abajo" | DBLSC "MAL", impreso 475 (PDF 527) | VALIDADA |
| C10 | AMIGO | Amigo/a | nombre | 2 | "Las manos en '1', una con la palma hacia arriba y la otra hacia abajo, se acercan hasta que los índices se tocan por las yemas. Enseguida las manos giran sobre las muñecas hasta quedar en la posición inversa y nuevamente las yemas se tocan" | DBLSC "AMIGO", impreso 206 (PDF 258) | VALIDADA |

**Nota de estado:** las 10 clases quedan `VALIDADA` porque cada una tiene texto
descriptivo exacto en el DBLSC. Ninguna clase queda `PENDIENTE_VALIDACION` ni
`REEMPLAZAR`. La **validación lingüística por una persona sorda usuaria de LSC
o un intérprete certificado sigue siendo un paso obligatorio del protocolo de
recolección** (docs/protocolo_dataset.md, paso 0): si el experto observa una
diferencia con la descripción del diccionario, se actualiza la ficha y el
estado antes de cerrar el dataset.

## 2. Clasificación temporal: STATIC vs DYNAMIC (crítica para el modelo)

El MVP actual (MLP sobre 1 frame de 126 features) reconoce **configuraciones de
la mano** (forma de dedos, orientación, uso de manos). Toda seña cuyo
significado depende de una **trayectoria o movimiento** NO puede reconocerse
fielmente con un frame suelto.

| ID | Clase | Clasificación | Compatibilidad MVP |
|----|-------|---------------|--------------------|
| C01 | HOLA | DINÁMICA en su variante canónica (saludo informal: mano abierta dedos juntos hacia la sien con ligero despegue); la variante DBLSC (palma adentro, estática) no se recolecta | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: instante del despegue de la sien |
| C02 | ADIOS | DINÁMICA (movimiento lateral) | `DYNAMIC_REQUIRES_SEQUENCE` — en MVP se captura el fotograma guía (palma hacia la cámara) |
| C03 | GRACIAS | DINÁMICA (mentón → palma contraria) | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: contacto con la otra mano |
| C04 | POR_FAVOR | DINÁMICA (círculos sobre el pecho) + **posición corporal** (pecho) | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: puño cerrado; la posición del pecho se pierde tras la normalización |
| C05 | SI | DINÁMICA (flexión repetida de la muñeca) | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: puño cerrado palma adelante |
| C06 | NO | DINÁMICA en su variante canónica (mano en '1' de lado a lado); la variante 1 (pinza) y la 3 (cabeceo) no se recolectan | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: cualquier instante del vaivén (configuración constante) |
| C07 | AYUDA | DINÁMICA (avance de ambas manos) | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: puño apoyado sobre la palma contraria |
| C08 | BIEN | **ESTÁTICA** (pulgar extendido hacia arriba; el movimiento es opcional) | `STATIC_COMPATIBLE_WITH_MLP` |
| C09 | MAL | **ESTÁTICA** (pulgar extendido hacia abajo; el movimiento es opcional) | `STATIC_COMPATIBLE_WITH_MLP` |
| C10 | AMIGO | DINÁMICA (rotación de muñecas con contacto) | `DYNAMIC_REQUIRES_SEQUENCE` — fotograma guía: índices en contacto (cualquier instante de contacto) |

Resumen:

- **2 estáticas compatibles con el MLP actual:** BIEN, MAL.
- **8 dinámicas que requieren modelo temporal** (`TEMPORAL_MODEL_REQUIRED`) para
  un reconocimiento fiel: HOLA, ADIOS, GRACIAS, POR_FAVOR, SI, NO, AYUDA,
  AMIGO.
- Estrategia honesta para el MVP: cada clase dinámica se recolecta **siempre en
  el mismo fotograma guía** (ver tabla anterior y protocolo de recolección), de
  modo que el MLP aprende la *configuración congelada*; se documenta el riesgo
  de confusión. El modelo temporal (secuencias, ej. MLP sobre ventanas de
  frames o LSTM) queda señalado como evolución futura (`TEMPORAL_MODEL_REQUIRED`),
  **no se simula ni se promete** en la versión actual.

## 3. LSC ≠ ASL ≠ español signado

- La **Lengua de Señas Colombiana (LSC)** es una lengua con gramática propia,
  independiente del español y del ASL. Las fuentes usadas son colombianas y
  específicas de LSC (DBLSC; INSOR; FENASCOL).
- **Ejemplo concreto de diferencia con ASL:** la seña AMIGO del ASL (los dedos
  índice y pulgar de ambas manos forman círculos que se entrelazan) NO es la
  seña AMIGO del LSC, que según el DBLSC usa dos manos en '1' cuyas yemas de
  los índices se tocan y giran sobre las muñecas. Durante la investigación de
  F13 la descripción de ASL apareció en búsquedas genéricas de internet; fue
  **descartada** por la cita directa del DBLSC.
- **Ejemplo de diferencia con el español signado (calco):** las señas del DBLSC
  tienen categoría gramatical propia (locuciones, adverbios, verbos) y el
  diccionario documenta modificaciones gramaticales propias de LSC (modo,
  aspecto, direccionalidad hacia el objeto, p. ej. GRACIAS y AYUDAR pueden
  orientarse hacia el interlocutor). El proyecto usa **lexemas aislados**
  validados en LSC, no calcos del español.
- Referencia académica complementaria: estudio colombiano que entrenó
  exactamente estas señas LSC (Hola, Gracias, Bien, Mal, No, Sí, Por favor,
  ¿Cómo estás?, Bienvenido, Con mucho gusto) — "Reconocimiento de lengua de
  señas colombiana mediante redes neuronales convolucionales y captura de
  movimiento", Tecnura vol. 26 n.º 74 (2022), DOI 10.14483/22487638.19213
  (Plazas, Gutiérrez, Suárez, González — UPTC). La lista de clases coincide
  parcialmente con la nuestra, lo que refuerza la elección.

## 4. Correcciones a la investigación preliminar (fichas antiguas)

Durante la búsqueda web preliminar (F13) se obtuvieron descripciones
**incompletas o erróneas** que la extracción del texto íntegro del PDF corrige:

| Clase | Descripción web preliminar | Corrección (texto DBLSC íntegro) |
|-------|----------------------------|----------------------------------|
| GRACIAS | "Palma sobre el pecho se mueve hacia adelante hasta quedar palma arriba" | Eso es DE-NADA / CON-MUCHO-GUSTO (impreso 489). El GRACIAS del DBLSC toca el **mentón** y luego contacta la **palma de la otra mano** (2 manos) |
| AYUDA | "Mano en 'B' golpea el mentón" | Esa descripción pertenece a **¿QUIÉN?** (impreso 483, PDF 535); el fragmento web mezcló la oración de ejemplo que contenía AYUDAR. AYUDAR real: puño sobre la palma de la otra mano, ambas avanzan (2 manos) |
| AMIGO | "Círculos de índice y pulgar que se entrelazan" | Esa es la seña **ASL**; en LSC son dos manos en '1' cuyas yemas de índices se tocan y rotan sobre las muñecas |

`docs/señas_lsc.md` (fichas F10) se actualizó en F13 para reflejar estas
correcciones y el estado VALIDADA.

## 5. Ambigüedades conocidas y mitigaciones

| Par | Riesgo | Mitigación |
|-----|--------|-----------|
| HOLA vs ADIOS | Con la variante canónica actual difieren claramente: HOLA es mano abierta dedos juntos hacia la sien (despegue), ADIOS es '5' palma adelante moviéndose de lado a lado | Recolectar cada una en su fotograma guía consistente (HOLA: despegue de la sien; ADIOS: palma visible); configuración y trayectoria separables en landmarks |
| BIEN vs MAL | Puño con pulgar extendido: difieren solo en la dirección del pulgar | La coordenada de la punta del pulgar distingue ambas configuraciones (arriba vs abajo) — separables por configuración |
| SI vs POR_FAVOR | Ambos son puño cerrado; difieren en orientación (palma adelante vs palma atrás) y en que POR_FAVOR se articula sobre el pecho | Orientación de la palma observable; la posición del pecho NO es observable tras la normalización → se documenta como limitación |
| NO variantes 1 vs 2 | Signantes reales pueden usar la variante estática (pinza de 3 dedos) | El dataset usa la variante 2 canónica (mano en '1' oscilando — decisión del signante); si un participante produce la variante 1, se descarta y se repite con la 2 |
| GRACIAS, AYUDA, AMIGO (2 manos) | Requieren ambas manos visibles y sincronizadas | El pipeline captura 2 manos (features 0..125); hand_count debe ser 2 para estas clases en el control de calidad; descartar muestras con una sola mano |
| POR_FAVOR, GRACIAS (posición corporal) | Tocar pecho/mentón no es observable tras normalizar por muñeca | Se documenta como limitación real; el MVP reconoce por configuración + orientación; la posición corporal quedará disponible solo con modelo temporal + landmarks corporales (fuera de alcance actual) |

## 6. Registro de fuentes verificadas (para citar)

1. **DBLSC** — Diccionario Básico de la Lengua de Señas Colombiana. INSOR /
   Instituto Caro y Cuervo, 2006. PDF oficial:
   https://www.colombiaaprende.edu.co/sites/default/files/files_public/2022-04/Diccionario-lengua-de-senas.pdf
   (también:
   https://lenguasyliteraturasnativas.caroycuervo.gov.co/publicaciones/2023/11/Diccionario_lengua_de_senas_colombiana.pdf).
   Entradas extraídas: DECIR-HOLA 489, DECIR-ADIÓS 489, AGRADECER/DECIR-GRACIAS/GRACIAS-A 485,
   POR-FAVOR 491, SÍ 458, NO 452, AYUDAR/AYUDA 243, BIEN 473, MAL 475, AMIGO 206.
2. **INSOR Educativo — Diccionario y Repositorio Virtual Bilingüe LSC–Español**
   (videos para la validación visual): https://educativo.insor.gov.co/diccionario/
3. **FENASCOL** — Federación Nacional de Sordos de Colombia:
   https://www.fenascol.org.co
4. **Estudio académico colombiano** (mismas señas LSC entrenadas con éxito):
   Tecnura 26(74), 2022 — DOI 10.14483/22487638.19213;
   http://www.scielo.org.co/scielo.php?pid=S0123-921X2022000400070&script=sci_arttext

**Nota de método (honestidad):** la verificación de F13 se hizo sobre el texto
impreso del diccionario (descripciones escritas por lingüistas), que es la
autoridad para la *configuración articulatoria*. Los **videos** del diccionario
en línea de INSOR Educativo y la revisión por un usuario nativo de LSC son el
paso visual pendiente, que forma parte del protocolo de recolección (paso 0).
