"""Tests del HandTracker (MediaPipe Tasks Vision) SIN cámara.

No abre ninguna cámara ni flujo de video: usa frames sintéticos (negros) y
landmarks sintéticos. Verifica:
- la paridad de la asignación izquierda/derecha con el frontend
  (mapHandsResult en handTracking.js), que es la base de la paridad F8;
- el contrato del vector de features de 126-d que consume collect_data.py;
- que el tracker inicializa y detecta sin cámara (pipeline real de MediaPipe).

Ejecutar:  python -m pytest tests/test_hand_tracker.py -v
"""

from types import SimpleNamespace

import numpy as np
import pytest

mp = pytest.importorskip("mediapipe")

from ai.collection.hand_tracker import (  # noqa: E402
    HandTracker,
    assign_hands,
)
from ai.preprocessing.features import FEATURES_SIZE, hands_to_features  # noqa: E402
from ai.training.config import HAND_LANDMARKER_MODEL  # noqa: E402


def _lm(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _cat(label, score):
    return SimpleNamespace(category_name=label, score=score)


def _hand_landmarks(seed=0):
    """21 landmarks sintéticos (muñeca y MCP medio bien posicionados)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.1, 0.9, size=(21, 3))
    pts[0] = [0.5, 0.5, 0.0]     # muñeca
    pts[9] = [0.5, 0.35, 0.0]    # MCP medio (palma)
    return [_lm(*p) for p in pts]


class TestAssignHandsParidadConFrontend:
    """assign_hands() debe replicar EXACTAMENTE mapHandsResult() del frontend."""

    def test_etiquetas_normales(self):
        hands = assign_hands(
            [_hand_landmarks(1), _hand_landmarks(2)],
            [[_cat("Left", 0.9)], [_cat("Right", 0.8)]],
        )
        assert [h.label for h in hands] == ["Left", "Right"]

    def test_asignacion_por_etiqueta_no_por_posicion(self):
        # Igual que el frontend: la posición no importa, la etiqueta sí.
        hands = assign_hands(
            [_hand_landmarks(1), _hand_landmarks(2)],
            [[_cat("Right", 0.8)], [_cat("Left", 0.9)]],
        )
        assert [h.label for h in hands] == ["Right", "Left"]

    def test_fallback_por_orden_sin_etiqueta(self):
        # handedness ausente (quirk del modo VIDEO): 1ª -> Left, 2ª -> Right.
        hands = assign_hands([_hand_landmarks(1), _hand_landmarks(2)], [])
        assert [h.label for h in hands] == ["Left", "Right"]

    def test_etiqueta_duplicada_fallback(self):
        hands = assign_hands(
            [_hand_landmarks(1), _hand_landmarks(2)],
            [[_cat("Left", 0.9)], [_cat("Left", 0.7)]],
        )
        assert [h.label for h in hands] == ["Left", "Right"]

    def test_estructura_landmarks_21x3(self):
        hands = assign_hands([_hand_landmarks(1)], [])
        assert hands[0].landmarks.shape == (21, 3)
        assert hands[0].landmarks.dtype == np.float64

    def test_score_conservado(self):
        hands = assign_hands(
            [_hand_landmarks(1)],
            [[_cat("Left", 0.987)]],
        )
        assert hands[0].score == pytest.approx(0.987)


class TestContratoFeatures126d:
    def test_bloques_izquierda_derecha(self):
        hands = assign_hands([_hand_landmarks(1), _hand_landmarks(2)], [])
        left = next(h.landmarks for h in hands if h.label == "Left")
        right = next(h.landmarks for h in hands if h.label == "Right")
        feats = hands_to_features(left, right)
        assert feats.shape == (FEATURES_SIZE,)
        assert feats.dtype == np.float32
        assert np.any(feats[0:63] != 0.0)
        assert np.any(feats[63:126] != 0.0)

    def test_mano_ausente_ceros(self):
        assert np.all(hands_to_features(None, None) == 0.0)


@pytest.fixture(scope="module")
def tracker():
    if not HAND_LANDMARKER_MODEL.exists():
        pytest.skip("falta frontend/vendor/mediapipe/hand_landmarker.task")
    tr = HandTracker(max_hands=2)
    yield tr
    tr.close()


class TestHandTrackerSinCamara:
    """Pipeline real de MediaPipe sobre frames sintéticos (sin cámara)."""

    def test_inicializacion_sin_camara(self, tracker):
        assert tracker.max_hands == 2

    def test_detect_frame_sintetico_vacio(self, tracker):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        assert tracker.detect(blank) == []

    def test_detect_secuencial_timestamps_crecientes(self, tracker):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(3):
            assert tracker.detect(blank) == []

    def test_close_idempotente(self, tracker):
        tracker.close()
        tracker.close()