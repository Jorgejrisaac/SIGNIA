"""Normalización y empaquetado de landmarks (REFERENCIA del proyecto).

Esta transformación es la ÚNICA fuente de verdad durante entrenamiento
y debe replicarse EXACTAMENTE en el frontend (frontend/js/features.js)
— se verifica con el test de paridad de la Fase 8.

Formato del vector de features (126 dimensiones):
  [0..62]   mano "Left" según MediaPipe Tasks Vision (frame sin espejar)
  [63..125] mano "Right"
  Si una mano no está presente, su bloque se rellena con ceros.

Normalización por mano:
  1. Origen en la muñeca (landmark 0).
  2. Escala = distancia muñeca -> MCP del dedo medio (landmark 9).
  3. Cada landmark = (landmark - muñeca) / escala.
  El resultado es invariante a traslación y escala (no a rotación: la
  orientación de la mano es información útil para el modelo).
"""

from __future__ import annotations

import numpy as np

from ai.training.config import (
    ANCHOR_LANDMARK,
    COORDS_PER_LANDMARK,
    EPSILON,
    FEATURES_SIZE,
    LANDMARKS_PER_HAND,
    LEFT_HAND_SLICE,
    RIGHT_HAND_SLICE,
    SCALE_LANDMARK,
)

# MediaPipe: muñeca = 0, MCP medio = 9, puntas de dedos = 4,8,12,16,20
HAND_CONNECTIONS = frozenset(
    {
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    }
)


def normalize_hand(landmarks: np.ndarray) -> np.ndarray:
    """Normaliza una mano (21, 3) según el procedimiento documentado.

    Devuelve una matriz(21, 3) con la mano centrada en la muñeca y
    escalada por el tamaño de la palma. La muñeca queda en (0, 0, 0).
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    wrist = landmarks[ANCHOR_LANDMARK]
    scale = float(np.linalg.norm(landmarks[SCALE_LANDMARK] - wrist))
    if scale < EPSILON:
        scale = EPSILON
    return (landmarks - wrist) / scale


def hands_to_features(
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
) -> np.ndarray:
    """Empaqueta ambas manos en el vector de 126 dimensiones.

    - left_hand / right_hand: matriz (21, 3) o None si la mano no está.
    - La mano ausente se rellena con ceros.
    - El formato resultante es el definido en el docstring del módulo.
    """
    features = np.zeros(FEATURES_SIZE, dtype=np.float32)
    if left_hand is not None:
        features[LEFT_HAND_SLICE] = normalize_hand(left_hand).astype(np.float32).ravel()
    if right_hand is not None:
        features[RIGHT_HAND_SLICE] = (
            normalize_hand(right_hand).astype(np.float32).ravel()
        )
    return features


def features_to_hands(features: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Inversa de hands_to_features (útil para depuración y visualización)."""
    features = np.asarray(features, dtype=np.float64).reshape(2, 21, 3)
    left = features[0] if np.any(features[0]) else None
    right = features[1] if np.any(features[1]) else None
    return left, right