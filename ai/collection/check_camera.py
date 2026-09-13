"""Verificación de la Fase 2: cámara + detección de landmarks.

Muestra el video de la webcam (o un flujo de video remoto) con los 21
landmarks por mano y la etiqueta de handedness (Left/Right).
Presiona Q para salir.

Uso:
    python ai/collection/check_camera.py                 # webcam local (0)
    python ai/collection/check_camera.py --camera 1
    python ai/collection/check_camera.py --camera http://192.168.1.10:8080/video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.collection.hand_tracker import HandTracker, open_camera  # noqa: E402
from ai.training.config import CAMERA_INDEX  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=CAMERA_INDEX,
                        help="Índice de cámara local (0) o URL de video del celular")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara (índice {args.camera}).")
    tracker = HandTracker(max_hands=2)
    print("Cámara iniciada. Presiona Q para salir.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error de lectura de cámara.")
                break
            hands = tracker.detect(frame)
            canvas = tracker.draw(frame, hands)
            cv2.imshow("SIGNIA — Verificación de cámara", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    main()