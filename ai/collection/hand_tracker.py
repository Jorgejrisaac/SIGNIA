"""Detección de manos con MediaPipe Tasks Vision (envuelto para SIGNIA).

API: `mp.tasks.vision.HandLandmarker` (MediaPipe 1.x). La API legacy
`mp.solutions.hands` fue ELIMINADA en MediaPipe 1.0.

El recolector usa EXACTAMENTE la misma librería (tasks-vision), el MISMO
archivo de modelo (`hand_landmarker.task`) y la MISMA convención que la
inferencia del frontend (`frontend/js/handTracking.js`):

- El frame de detección NO se espeja (el frontend alimenta el <video> sin
  espejar; el espejo es solo visualización CSS).
- La etiqueta de handedness de Tasks Vision se usa tal cual (igual que
  `mapHandsResult()`), por lo que los bloques izquierda/derecha del vector
  de features son idénticos entre entrenamiento y la inferencia del navegador.
- La GUI SÍ muestra el frame espejado (natural para el operador); los
  landmarks se espejan solo al dibujar sobre esa vista.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from ai.preprocessing.features import HAND_CONNECTIONS
from ai.training.config import HAND_LANDMARKER_MODEL

# Umbral de confianza de handedness (reservado para validación futura de
# etiquetas ambiguas; no filtra detecciones).
HANDEDNESS_MIN_SCORE = 0.6


def open_camera(source: int | str):
    """Abre la cámara o un flujo de video (fuente configurable).

    - int:  índice de cámara local (0 = webcam integrada/externa).
    - str:  URL de video (MJPEG/RTSP) p. ej. cámara de celular vía
            "IP Webcam" (http://192.168.x.x:8080/video) o DroidCam.
            Ver README, sección "Cámaras: cómo usar SIGNIA".
    """
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {source}. "
            "Revisa conexión/permisos (webcam) o la URL (celular)."
        )
    return cap


@dataclass
class DetectedHand:
    label: str          # "Left" | "Right" (bloque asignado, ver assign_hands)
    score: float        # confianza de la clasificación de handedness
    landmarks: np.ndarray  # (21, 3) normalizadas [0, 1] relativas al frame


def assign_hands(landmarks, handedness) -> list[DetectedHand]:
    """Réplica PYTHON de `mapHandsResult()` del frontend (handTracking.js).

    Asigna cada mano a su bloque ("Left"/"Right") con EXACTAMENTE la misma
    lógica que la inferencia en el navegador, incluido el fallback por orden
    de aparición cuando la etiqueta falta o se duplica (quirk del modo VIDEO).

    Garantiza que los datos recolectados produzcan los mismos bloques
    izquierda/derecha que el frontend en tiempo de inferencia.
    """
    hands: list[DetectedHand] = []
    left_taken = False
    right_taken = False
    for i, lm in enumerate(landmarks):
        label, score = "", 0.0
        if i < len(handedness) and handedness[i]:
            label = handedness[i][0].category_name or ""
            score = float(handedness[i][0].score)
        points = np.array(
            [[p.x, p.y, p.z] for p in lm],
            dtype=np.float64,
        )
        if label == "Left" and not left_taken:
            block, left_taken = "Left", True
        elif label == "Right" and not right_taken:
            block, right_taken = "Right", True
        elif not left_taken:
            block, left_taken = "Left", True
        else:
            block, right_taken = "Right", True
        hands.append(DetectedHand(label=block, score=score, landmarks=points))
    return hands


class HandTracker:
    """HandLandmarker de MediaPipe Tasks Vision (Python, MediaPipe 1.x).

    API pública idéntica a la implementación anterior (detect/draw/close),
    por lo que `collect_data.py` no cambia.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ):
        self.max_hands = max_hands
        model_path = Path(model_path or HAND_LANDMARKER_MODEL)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo MediaPipe no encontrado: {model_path}\n"
                "Se espera frontend/vendor/mediapipe/hand_landmarker.task "
                "(el mismo que usa el frontend)."
            )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )
        self._start_monotonic = time.monotonic()
        self._last_ts_ms = 0

    def detect(self, bgr_frame: np.ndarray) -> list[DetectedHand]:
        """Detecta las manos en un frame BGR (SIN espejar, como el frontend)."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # tasks-vision (modo VIDEO) exige timestamps estrictamente crecientes.
        ts_ms = int((time.monotonic() - self._start_monotonic) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(image, ts_ms)
        return assign_hands(result.hand_landmarks or [], result.handedness or [])

    @staticmethod
    def draw(frame_bgr: np.ndarray, hands: list[DetectedHand]) -> np.ndarray:
        """Dibuja los landmarks sobre la vista espejada (solo visualización).

        El canvas se espeja para el operador, pero los landmarks están en
        coordenadas del frame SIN espejar: x_visual = 1 - x.
        """
        canvas = cv2.flip(frame_bgr, 1)
        for hand in hands:
            points = hand.landmarks
            h, w = canvas.shape[:2]
            for a, b in HAND_CONNECTIONS:
                p1 = (int((1 - points[a][0]) * w), int(points[a][1] * h))
                p2 = (int((1 - points[b][0]) * w), int(points[b][1] * h))
                cv2.line(canvas, p1, p2, (255, 255, 255), 1)
            for (x, y, _) in points:
                cv2.circle(
                    canvas, (int((1 - x) * w), int(y * h)), 4, (37, 99, 235), -1
                )
            label = f"{hand.label} ({hand.score:.2f})"
            cv2.putText(
                canvas, label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (34, 197, 94), 2,
            )
        return canvas

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                # mediapipe 1.x: close() puede fallar si el dispatcher C++
                # ya se destruyó (teardown/GC); el recurso queda liberado.
                pass
            self._landmarker = None