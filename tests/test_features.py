"""Tests de la normalización (única fuente de verdad del proyecto).

Ejecutar:  python -m pytest tests/ -v
"""

import numpy as np
import pytest

from ai.preprocessing.features import (
    FEATURES_SIZE,
    hands_to_features,
    normalize_hand,
)

NPC = pytest.importorskip("numpy")


def _synthetic_hand(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hand = rng.uniform(0.1, 0.9, size=(21, 3))
    hand[0] = [0.5, 0.5, 0.0]     # muñeca
    hand[9] = [0.5, 0.35, 0.0]    # MCP medio (pálma)
    return hand


class TestNormalizeHand:
    def test_tamaño_y_tipo(self):
        out = normalize_hand(_synthetic_hand())
        assert out.shape == (21, 3)
        assert out.dtype == np.float64

    def test_muñeca_en_origen(self):
        out = normalize_hand(_synthetic_hand())
        assert np.allclose(out[0], 0.0)

    def test_escala_da_tamaño_palma_1(self):
        out = normalize_hand(_synthetic_hand())
        assert np.linalg.norm(out[9] - out[0]) == pytest.approx(1.0)

    def test_invariante_a_traslacion(self):
        a = normalize_hand(_synthetic_hand())
        b = normalize_hand(_synthetic_hand() + np.array([0.3, -0.4, 0.1]))
        assert np.allclose(a, b)

    def test_invariante_a_escala(self):
        a = normalize_hand(_synthetic_hand())
        b = normalize_hand(_synthetic_hand() * 2.0)
        assert np.allclose(a, b)


class TestHandsToFeatures:
    def test_dos_manos_dimension(self):
        feats = hands_to_features(_synthetic_hand(1), _synthetic_hand(2))
        assert feats.shape == (FEATURES_SIZE,)
        assert feats.dtype == np.float32

    def test_una_mano_rellena_la_otra_con_ceros(self):
        feats = hands_to_features(None, _synthetic_hand(2))
        assert np.all(feats[0:63] == 0.0)
        assert np.any(feats[63:126] != 0.0)

    def test_mezcla_izquierda_antes_que_derecha(self):
        left, right = _synthetic_hand(1), _synthetic_hand(2)
        feats = hands_to_features(left, right)
        assert np.allclose(feats[0:63], normalize_hand(left).ravel())
        assert np.allclose(feats[63:126], normalize_hand(right).ravel())

    def test_ninguna_mano_ceros(self):
        assert np.all(hands_to_features(None, None) == 0.0)