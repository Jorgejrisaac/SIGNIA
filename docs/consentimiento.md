# CONSENTIMIENTO INFORMADO — RECOLECCIÓN DE DATOS (F14)

> Borrador para la recolección del dataset real de SIGNIA. Se firma (o se
> registra el consentimiento verbal ante el instructor) **antes** de la
> primera sesión de cada participante. Responsable de la recolección: el
> usuario del proyecto.

## 1. Finalidad

Recolectar datos de **landmarks de las manos** (coordenadas de 21 puntos por
mano) para entrenar un modelo local de reconocimiento de 10 señas de la
Lengua de Señas Colombiana (LSC), como parte del proyecto de investigación
SIGNIA.

## 2. Qué se almacena y qué NO

| Sí se almacena | NO se almacena |
|----------------|----------------|
| Landmarks de manos (21 puntos ×3 por mano, por fotograma) | Video original ni imágenes de la cámara |
| Configuración de la mano (forma de dedos, orientación) | Audio ni voz |
| Metadatos: ID anónimo (P01…), sesión (A/B/C), clase de seña, fechas y horas | Nombre real, documento de identidad, rostro |
| (Opcional, con autorización aparte) video guía de la seña | Ubicación (GPS) |

**Los landmarks NO se consideran automáticamente anónimos:** la geometría de
la mano puede asociarse indirectamente a una persona. Por eso se usan IDs
anónimos y se limita el acceso a los archivos.

## 3. Dónde se almacena y durante cuánto tiempo

- Almacenamiento **local** en el computador del proyecto, en `ai/dataset/`.
- **No se sube a ninguna nube ni se envía a terceros**.
- Los datos se conservarán mientras el proyecto los necesite para entrenar y
  evaluar el modelo; al terminar el proyecto se ofrecerá borrarlos (ver §4).

## 4. Derechos del participante

- Retirarse en cualquier momento sin explicaciones.
- Solicitar la eliminación de sus datos: se borran sus archivos y
  metadatos, y el modelo se **reentrena sin sus datos** la próxima vez.
- Preguntar qué datos suyos existen (el recolector puede listarlos por ID).
- Si es menor de edad: firma del acudiente o representante legal
  (instituciones educativas pueden exigir autorización adicional).

## 5. Videos

- En la recolección NO se guardan videos de la sesión (solo landmarks).
- Si se graba un **video guía** de la seña (para validación lingüística por
  instructor sordo), se solicita autorización explícita separada, se conserva
  con nombre ficticio y se elimina al terminar la participación.

## 6. Declaración

He leído (o me explicaron en LSC) la información anterior. Entiendo qué datos
se recolectan, dónde se guardan, que no se comparten, y que puedo retirarme o
pedir el borrado de mis datos en cualquier momento. Autorizo la recolección
de mis landmarks de manos con ID anónimo para el proyecto SIGNIA.

| | Nombre | Firma | Fecha |
|---|--------|-------|-------|
| Participante (o acudiente si es menor) | | | |
| Instructor / intérprete (testigo) | | | |
| Responsable del proyecto | | | |

> Protección de datos (Colombia): Ley 1581 de 2012 y Decreto 1377 de 2013 —
> datos personales tratados con finalidad académica, acceso restringido y
> derecho de supresión.
