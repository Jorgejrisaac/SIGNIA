"""Tests del descarte (D) del recolector F14 (regresión del TypeError).

El bug original: `_discard_last()` llamaba `sample_dir(RAW_DIR, person,
session, cls)` con 4 argumentos, pero `sample_dir()` exige 5
(`sample_id`) -> TypeError al pulsar D. El descarte ahora usa
`discard_last_sample()`, que construye la ruta de la clase directamente
(`raw/{persona}/{sesion}/{clase}/`) y nunca toca la estructura del dataset.

Todos los tests usan un raw_dir temporal (tmp_path): JAMÁS el dataset real.

Ejecutar:  python -m pytest tests/test_collect_discard.py -v
"""

import json

import pytest

from ai.collection.collect_data import discard_last_sample

PERSON = "P01"
SESSION = "A"
CLS = "HOLA"


def _make_sample(raw, sample_id: str, cls: str = CLS) -> None:
    d = raw / PERSON / SESSION / cls / sample_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps({"sample_id": sample_id, "quality": "valid"}),
        encoding="utf-8",
    )


class TestDiscardLastSample:
    def test_descarta_la_ultima_muestra(self, tmp_path):
        for i in range(1, 4):
            _make_sample(tmp_path, f"{PERSON}_{SESSION}_{CLS}_{i:04d}")
        moved = discard_last_sample(tmp_path, PERSON, SESSION, CLS)
        assert moved is not None
        assert moved.name == f"{PERSON}_{SESSION}_{CLS}_0003"
        assert moved.parent == tmp_path / "_discarded" / PERSON / SESSION / CLS
        assert moved.exists()
        meta = json.loads((moved / "metadata.json").read_text(encoding="utf-8"))
        assert meta["quality"] == "invalid"
        assert meta["issues"] == ["operator_discard"]
        rest = sorted(d.name for d in (tmp_path / PERSON / SESSION / CLS).iterdir())
        assert rest == [f"{PERSON}_{SESSION}_{CLS}_0001",
                        f"{PERSON}_{SESSION}_{CLS}_0002"]

    def test_sin_muestras_devuelve_none_sin_error(self, tmp_path):
        # Regresión del TypeError: antes fallaba ANTES de comprobar si había
        # muestras (sample_dir con 4 argumentos).
        (tmp_path / PERSON / SESSION / CLS).mkdir(parents=True)
        assert discard_last_sample(tmp_path, PERSON, SESSION, CLS) is None

    def test_clase_inexistente_devuelve_none(self, tmp_path):
        assert discard_last_sample(tmp_path, PERSON, SESSION, CLS) is None

    def test_no_toca_otras_clases(self, tmp_path):
        _make_sample(tmp_path, f"{PERSON}_{SESSION}_ADIOS_0001", cls="ADIOS")
        _make_sample(tmp_path, f"{PERSON}_{SESSION}_{CLS}_0001")
        discard_last_sample(tmp_path, PERSON, SESSION, CLS)
        adios = tmp_path / PERSON / SESSION / "ADIOS" / \
            f"{PERSON}_{SESSION}_ADIOS_0001"
        assert adios.exists()

    def test_ignora_directorios_no_coincidentes(self, tmp_path):
        # Un dir que no sigue el patrón de sample_id no es candidato.
        _make_sample(tmp_path, f"{PERSON}_{SESSION}_{CLS}_0002")
        (tmp_path / PERSON / SESSION / CLS / "notas").mkdir(exist_ok=True)
        moved = discard_last_sample(tmp_path, PERSON, SESSION, CLS)
        assert moved.name == f"{PERSON}_{SESSION}_{CLS}_0002"

    def test_descarte_en_cadena_no_afecta_resto(self, tmp_path):
        for i in range(1, 4):
            _make_sample(tmp_path, f"{PERSON}_{SESSION}_{CLS}_{i:04d}")
        discard_last_sample(tmp_path, PERSON, SESSION, CLS)
        discard_last_sample(tmp_path, PERSON, SESSION, CLS)
        rest = sorted(d.name for d in (tmp_path / PERSON / SESSION / CLS).iterdir())
        assert rest == [f"{PERSON}_{SESSION}_{CLS}_0001"]
        discarded = sorted(
            d.name for d in (tmp_path / "_discarded" / PERSON / SESSION / CLS).iterdir()
        )
        assert discarded == [f"{PERSON}_{SESSION}_{CLS}_0002",
                             f"{PERSON}_{SESSION}_{CLS}_0003"]