"""Recolector de landmarks LSC — esquema F14 (una muestra = una ejecución).

Uso:
    python ai/collection/collect_data.py --person P01 --session A --target 100
    python ai/collection/collect_data.py --person P01 --session B --target 100 \
        --camera-source http://192.168.1.10:8080/video --source smartphone

Modos de captura (según SIGN_TYPES de ai/training/config.py):
    ESTÁTICAS (BIEN, MAL):
        ESPACIO  captura la pose: registra STATIC_HOLD_FRAMES fotogramas
                 mientras el participante mantiene la posición y guarda la
                 ejecución con su fotograma representativo (guide.json).
    DINÁMICAS (HOLA, ADIOS, GRACIAS, POR_FAVOR, SI, NO, AYUDA, AMIGO):
        R        inicia/termina la grabación de la SECUENCIA completa
                 (inicio -> trayectoria -> final); se auto-detiene a los
                 MAX_SEQUENCE_SECONDS. Se conservan TODOS los fotogramas.

Controles (ventana):
    ESPACIO / R  capturar (según el modo de la clase actual)
    <- / ->      cambiar de clase (alternativas: coma / punto)
    D            descartar la última muestra de la clase actual (se MUEVE a
                 raw/_discarded/ con quality='invalid'; nunca se borra en
                 silencio)
    Q            salir (si hay una grabación en curso, NO se guarda)

Esquema de guardado (docs/dataset_schema.md):
    ai/dataset/raw/{persona}/{sesion}/{clase}/{sample_id}/
        metadata.json   # trazabilidad completa (sample_id, persona, sesión,
                        #  clase, timestamps, hand_count, frames, guía...)
        guide.json      # fotograma guía (REFERENCIA del MVP, nunca la
                        #  representación completa de una seña dinámica)
        frames/*.json   # TODA la secuencia (dinámicas) / retención (estáticas)

Reglas F14:
- IDs únicos: nunca se sobrescriben (contador max + 1 por persona/sesión/clase).
- La muestra se guarda solo si en algún fotograma se observó el número de
  manos esperado para la clase (EXPECTED_HANDS); si no, se avisa y se
  reintenta (no se guarda basura).
- Reanudable: al iniciar, cuenta las muestras de la persona en la SESIÓN
  actual por clase y continúa desde ahí; la clase se salta si alcanzó
  --target. Las sesiones A/B/C son independientes entre sí.

--camera-source acepta un índice de cámara local o una URL MJPEG/RTSP
(celular por WiFi); ver README "Cámaras: cómo usar SIGNIA".
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.collection.hand_tracker import (  # noqa: E402
    HandTracker,
    open_camera,
)
from ai.preprocessing.features import hands_to_features  # noqa: E402
from ai.preprocessing.samples import (  # noqa: E402
    count_samples_for_person,
    mark_invalid,
    next_sample_id,
    sample_dir,
    select_guide_frame,
    write_frame,
    write_guide,
    write_metadata,
)
from ai.training.config import (  # noqa: E402
    CAMERA_INDEX,
    CLASES,
    DISCARDED_DIRNAME,
    EXPECTED_HANDS,
    MAX_SEQUENCE_SECONDS,
    PERSON_ID_DEFAULT,
    RAW_DIR,
    SESSIONS,
    SESSION_DEFAULT,
    SIGN_TYPES,
    STATIC_HOLD_FRAMES,
)

PERSON_RE = re.compile(r"^P\d+$")

# Códigos extendidos que devuelve cv2.waitKeyEx() para las flechas.
# Windows/OpenCV: (VK_LEFT << 16) = 0x250000 = 2424832,
#                 (VK_RIGHT << 16) = 0x270000 = 2555904.
# X11/GTK (Linux): 0xFF51 / 0xFF53. Con waitKey() simple (sin Ex) estos
# códigos se pierden (quedan en 0), por eso las flechas "no hacían nada".
LEFT_ARROW_CODES = (2424832, 0xFF51)
RIGHT_ARROW_CODES = (2555904, 0xFF53)


def interpret_key(key_code: int) -> str | None:
    """Traduce un código de cv2.waitKeyEx() a una acción del recolector.

    - prev_class / next_class: flechas <- -> (waitKeyEx) + alternativas
      ASCII `,`/`<` y `.`/`>` (paridad con el comportamiento anterior).
    - capture: ESPACIO o R (solo 'r' minúscula en reposo, paridad previa).
    - discard: D.  quit: Q.
    - None: tecla ignorada.
    """
    if key_code in LEFT_ARROW_CODES or key_code in (ord(","), ord("<")):
        return "prev_class"
    if key_code in RIGHT_ARROW_CODES or key_code in (ord("."), ord(">")):
        return "next_class"
    if key_code in (ord(" "), ord("r")):
        return "capture"
    if key_code == ord("d"):
        return "discard"
    if key_code == ord("q"):
        return "quit"
    return None


def discard_last_sample(raw_dir: Path, person_id: str, session_id: str,
                        cls: str) -> Path | None:
    """Mueve la última muestra válida de la clase a raw/_discarded/ y la
    marca quality='invalid' (operator_discard). NUNCA borra en silencio.

    Estructura del dataset (docs/dataset_schema.md):
        raw/{persona}/{sesion}/{clase}/{sample_id}/

    Devuelve la ruta movida o None si no hay muestras que descartar.
    """
    base = raw_dir / person_id / session_id / cls
    pattern = re.compile(rf"^{re.escape(person_id)}_{re.escape(session_id)}_"
                         rf"{re.escape(cls)}_\d{{4}}$")
    candidates = sorted(
        [d for d in base.glob("*") if d.is_dir() and pattern.match(d.name)]
    ) if base.exists() else []
    if not candidates:
        return None

    sample = candidates[-1]
    dest = raw_dir / DISCARDED_DIRNAME / person_id / session_id / cls \
        / sample.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sample), str(dest))
    mark_invalid(dest, ["operator_discard"])
    return dest


class Collector:
    def __init__(self, person_id: str, session_id: str, target_per_class: int,
                 source: str, camera_source: int | str = CAMERA_INDEX):
        if not PERSON_RE.match(person_id):
            raise ValueError(
                f"person_id inválido '{person_id}': usa el formato P01, P02, ..."
            )
        if session_id not in SESSIONS:
            raise ValueError(
                f"session_id inválido '{session_id}': usa {', '.join(SESSIONS)}"
            )
        self.person_id = person_id
        self.session_id = session_id
        self.target = max(target_per_class, 1)
        self.source = source
        self.camera_source = camera_source
        self.class_index = 0
        self.counts: dict[str, int] = {}
        self.total_saved = 0
        self.tracker = HandTracker(max_hands=2)

        self.recording = False
        self.buffer: list[dict] = []
        self._t0_ms = int(time.monotonic() * 1000)
        self._record_start_ms = 0

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        # Cuenta SOLO la sesión actual: A, B y C son sesiones independientes
        # (las muestras de otras sesiones no deben bloquear la captura).
        self.counts = count_samples_for_person(
            RAW_DIR, person_id, self.session_id
        )
        self.total_saved = sum(self.counts.values())

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cap = open_camera(self.camera_source)

        print("\n=== SIGNIA — RECOLECTOR DE DATOS LSC (F14) ===\n")
        print(f"Persona: {self.person_id} | Sesión: {self.session_id} | "
              f"Objetivo por clase: {self.target}")
        print("Controles: ESPACIO/R=capturar  <-/->=clase  D=descartar  Q=salir\n")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Error de lectura de cámara.")
                    break

                hands = self.tracker.detect(frame)
                left = next((h.landmarks for h in hands if h.label == "Left"), None)
                right = next((h.landmarks for h in hands if h.label == "Right"), None)
                features = hands_to_features(left, right)
                labels = [h.label for h in hands]
                t_ms = int(time.monotonic() * 1000) - self._t0_ms

                if self.recording:
                    self.buffer.append({
                        "index": len(self.buffer) + 1,
                        "features": features,
                        "hands": labels,
                        "hand_count": len(labels),
                        "t_ms": t_ms,
                    })
                    elapsed = t_ms - self._record_start_ms
                    cls = self._current_class()
                    if SIGN_TYPES[cls] == "static" and \
                            len(self.buffer) >= STATIC_HOLD_FRAMES:
                        self._stop_and_save()
                    elif SIGN_TYPES[cls] == "dynamic" and \
                            elapsed >= int(MAX_SEQUENCE_SECONDS * 1000):
                        self._stop_and_save(auto=True)

                canvas = self.tracker.draw(frame, hands)
                self._draw_status(canvas, hands)
                cv2.imshow("SIGNIA — Recolector", canvas)

                # waitKeyEx (NO waitKey): las flechas solo llegan como códigos
                # extendidos (0x250000/0x270000 en Windows); con waitKey + &0xFF
                # se pierden y quedan en 0.
                key = cv2.waitKeyEx(1)
                if key == ord("q"):
                    if self.recording:
                        print("  [!] Grabación en curso NO guardada (se descarta).")
                    break
                elif self.recording:
                    # R termina una DINÁMICA antes del auto-stop (la estática
                    # siempre completa sus STATIC_HOLD_FRAMES).
                    if SIGN_TYPES[self._current_class()] == "dynamic" and \
                            key in (ord("r"), ord("R")):
                        self._stop_and_save()
                    continue
                elif interpret_key(key) == "prev_class":
                    self._change_class(-1)
                elif interpret_key(key) == "next_class":
                    self._change_class(1)
                elif interpret_key(key) == "capture":
                    self._start_capture()
                elif interpret_key(key) == "discard":
                    self._discard_last()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()
            print(f"\nGuardadas {self.total_saved} muestras para {self.person_id}.")
            for cls in CLASES:
                print(f"  {cls}: {self.counts.get(cls, 0)}/{self.target}")

    # ------------------------------------------------------------- operaciones
    def _current_class(self) -> str:
        return CLASES[self.class_index]

    def _current_mode(self) -> str:
        return SIGN_TYPES[self._current_class()]

    def _change_class(self, delta: int) -> None:
        self.class_index = (self.class_index + delta) % len(CLASES)
        print(f"Clase: {self._current_class()} ({self._current_mode()})")

    def _start_capture(self) -> None:
        cls = self._current_class()
        if self.counts.get(cls, 0) >= self.target:
            print(f"  [x] {cls} ya alcanzó {self.target} muestras.")
            return
        self.recording = True
        self.buffer = []
        self._record_start_ms = int(time.monotonic() * 1000) - self._t0_ms
        if SIGN_TYPES[cls] == "static":
            print(f"  [●] Capturando pose ({STATIC_HOLD_FRAMES} fotogramas)...")
        else:
            print(f"  [●] Grabando secuencia (máx. {MAX_SEQUENCE_SECONDS:.0f} s)... "
                  f"pulsa R para terminar.")

    def _stop_and_save(self, auto: bool = False) -> None:
        self.recording = False
        cls = self._current_class()
        expected = EXPECTED_HANDS[cls]

        if not self.buffer:
            print("  [!] Sin fotogramas: muestra no guardada.")
            return
        if SIGN_TYPES[cls] == "dynamic" and len(self.buffer) < 2:
            print("  [!] Secuencia demasiado corta (<2 fotogramas): "
                  "muestra no guardada, reintenta.")
            self.buffer = []
            return
        if self.counts.get(cls, 0) >= self.target:
            print(f"  [x] {cls} ya alcanzó {self.target} muestras: "
                  f"grabación descartada.")
            self.buffer = []
            return

        selection = select_guide_frame(self.buffer, expected)
        if selection is None:
            print("  [!] Sin fotogramas válidos: muestra no guardada.")
            self.buffer = []
            return
        if not selection["expected_ok"]:
            print(f"  [!] Nunca se observaron las {expected} mano(s) esperada(s) "
                  f"para {cls}: muestra NO guardada, reintenta.")
            self.buffer = []
            return

        guide = selection["frame"]
        duration_ms = max(0, self.buffer[-1]["t_ms"] - self.buffer[0]["t_ms"])
        sample_id = next_sample_id(RAW_DIR, self.person_id, self.session_id, cls)
        out_dir = sample_dir(RAW_DIR, self.person_id, self.session_id, cls, sample_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for entry in self.buffer:
            write_frame(out_dir, entry["index"],
                        np.asarray(entry["features"], dtype=np.float32),
                        entry["hands"], entry["t_ms"])

        write_guide(out_dir, guide["index"],
                    np.asarray(guide["features"], dtype=np.float32),
                    guide["hands"], guide["t_ms"])

        metadata = {
            "sample_id": sample_id,
            "person_id": self.person_id,
            "session_id": self.session_id,
            "class": cls,
            "sign_type": SIGN_TYPES[cls],
            "timestamp": datetime.now().isoformat(),
            "source": self.source,
            "hand_count": guide["hand_count"],
            "hand_count_max": max(e["hand_count"] for e in self.buffer),
            "frame_count": len(self.buffer),
            "duration_ms": duration_ms,
            "guide_frame": guide["index"],
            "quality": "valid",
            "issues": [],
        }
        if SIGN_TYPES[cls] == "dynamic":
            metadata["sequence_id"] = sample_id
        write_metadata(out_dir, metadata)

        self.counts[cls] = self.counts.get(cls, 0) + 1
        self.total_saved += 1
        print(f"  [+] {cls} n.º {self.counts[cls]}/{self.target} | "
              f"{len(self.buffer)} frames | guía #{guide['index']} | "
              f"{duration_ms} ms"
              + (" (auto)" if auto else ""))
        self.buffer = []

    def _discard_last(self) -> None:
        cls = self._current_class()
        moved = discard_last_sample(
            RAW_DIR, self.person_id, self.session_id, cls
        )
        if moved is None:
            print(f"  [!] No hay muestras de {cls} que descartar.")
            return
        self.counts[cls] = max(0, self.counts.get(cls, 0) - 1)
        self.total_saved = max(0, self.total_saved - 1)
        print(f"  [-] {cls}: {moved.name} movida a "
              f"{DISCARDED_DIRNAME}/ (marcada operator_discard)")

    # ------------------------------------------------------------------- UI
    def _draw_status(self, canvas: np.ndarray, hands) -> None:
        cls = self._current_class()
        mode = self._current_mode()
        expected = EXPECTED_HANDS[cls]
        count = self.counts.get(cls, 0)
        detected = len(hands)
        quality_ok = detected == expected
        color = (34, 197, 94) if quality_ok else (239, 68, 68)

        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, h - 130), (w, h), (15, 23, 42), -1)
        cv2.putText(canvas, f"PERSON: {self.person_id}   SESSION: {self.session_id}",
                    (10, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)
        cv2.putText(canvas, f"CLASS: {cls}  ({mode.upper()})",
                    (10, h - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (248, 250, 252), 2)
        cv2.putText(canvas, f"Samples: {count} / {self.target}",
                    (10, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
        status = f"Hands: {detected}/{expected}  Quality: " + (
            "OK" if quality_ok else "NO MANO" if detected == 0 else "X")
        cv2.putText(canvas, status, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.recording:
            rec = f"REC {len(self.buffer)} frames ({self.buffer[-1]['t_ms']} ms)" \
                if self.buffer else "REC ..."
            cv2.putText(canvas, rec, (w - 380, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (239, 68, 68), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recolector de landmarks LSC (SIGNIA)")
    parser.add_argument("--person", default=PERSON_ID_DEFAULT,
                        help="Identificador anónimo del participante (P01, P02, ...)")
    parser.add_argument("--session", default=SESSION_DEFAULT,
                        help="Sesión de recolección (A, B o C; días distintos)")
    parser.add_argument("--target", type=int, default=100,
                        help="Muestras objetivo por clase (referencia inicial)")
    parser.add_argument("--source", default="webcam",
                        help="Origen de los datos (p. ej. 'webcam', 'smartphone')")
    parser.add_argument("--camera-source", default=CAMERA_INDEX,
                        help="Índice de cámara local (0) o URL de video del celular")
    args = parser.parse_args()
    Collector(person_id=args.person, session_id=args.session,
              target_per_class=args.target, source=args.source,
              camera_source=args.camera_source).run()


if __name__ == "__main__":
    main()
