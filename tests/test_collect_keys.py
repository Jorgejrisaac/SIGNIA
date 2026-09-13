"""Tests del manejo de teclado del recolector (navegación de clases).

Regresión del bug: la UI anunciaba `<-/->` para cambiar de clase, pero el
código usaba `cv2.waitKey(1) & 0xFF`, que en Windows PIERDE los códigos
extendidos de las flechas (2424832 / 2555904 quedan en 0). El recolector
ahora usa `cv2.waitKeyEx(1)` y `interpret_key()` para traducir teclas.

`interpret_key()` es pura y se verifica con los códigos reales que devuelve
OpenCV en Windows (documentados y reproducibles sin cámara ni ventana).

Ejecutar:  python -m pytest tests/test_collect_keys.py -v
"""

import pytest

from ai.collection.collect_data import (
    LEFT_ARROW_CODES,
    RIGHT_ARROW_CODES,
    interpret_key,
)

LEFT_ARROW_WINDOWS = 2424832   # 0x250000, VK_LEFT << 16
RIGHT_ARROW_WINDOWS = 2555904  # 0x270000, VK_RIGHT << 16


class TestFlechas:
    def test_flecha_izquierda_windows(self):
        assert interpret_key(LEFT_ARROW_WINDOWS) == "prev_class"

    def test_flecha_derecha_windows(self):
        assert interpret_key(RIGHT_ARROW_WINDOWS) == "next_class"

    def test_codigos_declarados_coinciden(self):
        assert LEFT_ARROW_WINDOWS in LEFT_ARROW_CODES
        assert RIGHT_ARROW_WINDOWS in RIGHT_ARROW_CODES

    def test_flechas_x11_linux(self):
        assert interpret_key(0xFF51) == "prev_class"
        assert interpret_key(0xFF53) == "next_class"

    def test_flecha_mascada_como_waitkey_no_dispara(self):
        # Lo que pasaba con waitKey() & 0xFF: la flecha llegaba como 0.
        assert interpret_key(0) is None


class TestAlternativasAscii:
    def test_comas_y_angulos(self):
        assert interpret_key(ord(",")) == "prev_class"
        assert interpret_key(ord("<")) == "prev_class"
        assert interpret_key(ord(".")) == "next_class"
        assert interpret_key(ord(">")) == "next_class"


class TestRestoDeTeclasParidad:
    def test_captura_espacio_y_r_minuscula(self):
        assert interpret_key(ord(" ")) == "capture"
        assert interpret_key(ord("r")) == "capture"

    def test_r_mayuscula_en_reposo_ignorada(self):
        # Paridad con el comportamiento anterior: R mayúscula solo actúa
        # DURANTE una grabación dinámica (manejado en el loop, no aquí).
        assert interpret_key(ord("R")) is None

    def test_d_y_q(self):
        assert interpret_key(ord("d")) == "discard"
        assert interpret_key(ord("q")) == "quit"

    def test_teclas_irrelevantes_ignoradas(self):
        for k in (ord("a"), ord("x"), ord("1"), ord("Q"), ord("D")):
            assert interpret_key(k) is None