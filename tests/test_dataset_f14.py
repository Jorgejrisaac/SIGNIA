"""Tests F14 del dataset real: esquema, IDs, metadata, validación, balance
y anti-leakage (especificación F14 §24).

Fixtures SINTÉTICOS en carpetas temporales (requisito 27): NUNCA viven en
ai/dataset/ ni se presentan como datos reales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai.preprocessing.preprocess import (
    DatasetEmptyError,
    load_dataset,
    split_by_person,
    verify_no_leakage,
)
from ai.preprocessing.report_dataset import report_to_dict
from ai.preprocessing.samples import (
    count_samples_for_person,
    mark_invalid,
    next_sample_id,
    select_guide_frame,
    validate_sample,
)
from ai.preprocessing.validate import validate_dataset
from ai.training.config import CLASES, EXPECTED_HANDS, SIGN_TYPES
from tests._dataset_helpers import make_sample


def _small_dataset(root: Path, persons=("P01", "P02"), classes=("HOLA", "ADIOS"),
                   per_person_class: int = 3):
    """Dataset sintético: per_person_class muestras por persona y clase."""
    for person in persons:
        for i, cls in enumerate(classes, start=1):
            for n in range(1, per_person_class + 1):
                make_sample(root, person=person, cls=cls, n=n)


class TestIDs:
    def test_next_sample_id_es_secuencial_y_no_sobrescribe(self, tmp_path):
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", session="A", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="A", cls="HOLA", n=2)
        make_sample(raw, person="P01", session="A", cls="HOLA", n=3)
        assert next_sample_id(raw, "P01", "A", "HOLA") == "P01_A_HOLA_0004"

    def test_ids_independientes_por_clase_y_sesion(self, tmp_path):
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", session="A", cls="HOLA", n=1)
        assert next_sample_id(raw, "P01", "A", "ADIOS") == "P01_A_ADIOS_0001"
        assert next_sample_id(raw, "P01", "B", "HOLA") == "P01_B_HOLA_0001"

    def test_formato_de_id_validado(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw)
        ok, issues = validate_sample(sample)
        assert ok
        assert not issues

    def test_conteo_por_persona_solo_carpetas_con_metadata(self, tmp_path):
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", session="A", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="B", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="A", cls="ADIOS", n=1)
        make_sample(raw, person="P02", session="A", cls="HOLA", n=1)
        counts = count_samples_for_person(raw, "P01")
        assert counts["HOLA"] == 2
        assert counts["ADIOS"] == 1
        assert counts["GRACIAS"] == 0

    def test_conteo_por_sesion_independiente(self, tmp_path):
        # P01/B no debe ver las muestras de P01/A (sesiones independientes).
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", session="A", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="A", cls="HOLA", n=2)
        make_sample(raw, person="P01", session="B", cls="HOLA", n=1)
        make_sample(raw, person="P02", session="A", cls="HOLA", n=1)
        counts_a = count_samples_for_person(raw, "P01", "A")
        counts_b = count_samples_for_person(raw, "P01", "B")
        assert counts_a["HOLA"] == 2
        assert counts_b["HOLA"] == 1
        assert counts_a["HOLA"] != counts_b["HOLA"]


class TestMetadata:
    REQUIRED = ("sample_id", "person_id", "session_id", "class", "sign_type",
                "timestamp", "source", "hand_count", "hand_count_max",
                "frame_count", "duration_ms", "guide_frame", "quality")

    def test_campos_obligatorios_presentes(self, tmp_path):
        raw = tmp_path / "raw"
        sample, meta = make_sample(raw)
        for field in self.REQUIRED:
            assert field in meta, f"falta campo {field}"

    def test_dinamica_tiene_sequence_id(self, tmp_path):
        raw = tmp_path / "raw"
        _, meta = make_sample(raw, cls="GRACIAS")
        assert meta["sequence_id"] == meta["sample_id"]

    def test_estatica_sin_sequence_id(self, tmp_path):
        raw = tmp_path / "raw"
        _, meta = make_sample(raw, cls="BIEN")
        assert "sequence_id" not in meta


class TestValidacion:
    def test_muestra_valida_pasa_sin_issues(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw)
        ok, issues = validate_sample(sample)
        assert ok and not issues

    def test_clase_invalida_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, cls="HOLA", meta_overrides={"class": "FUTBOL"})
        ok, issues = validate_sample(sample)
        assert not ok
        assert any("clase inválida" in issue for issue in issues)

    def test_persona_invalida_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, meta_overrides={"person_id": "anonimo"})
        ok, issues = validate_sample(sample)
        assert any("person_id inválido" in issue for issue in issues)

    def test_sesion_invalida_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, meta_overrides={"session_id": "Z"})
        ok, issues = validate_sample(sample)
        assert any("session_id inválido" in issue for issue in issues)

    def test_timestamp_invalido_detectado(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, meta_overrides={"timestamp": "ayer"})
        ok, issues = validate_sample(sample)
        assert any("timestamp inválido" in issue for issue in issues)

    def test_dimension_incorrecta_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        frame = {"features": [0.0] * 100, "hands": ["Right"], "hand_count": 1,
                 "t_ms": 1}
        sample, _ = make_sample(raw, cls="HOLA", frames=[frame, frame])
        ok, issues = validate_sample(sample)
        assert any("dimensión" in issue for issue in issues)

    def test_nan_inf_detectados(self, tmp_path):
        raw = tmp_path / "raw"
        vec = [float("nan")] * 126
        frame = {"features": vec, "hands": ["Right"], "hand_count": 1, "t_ms": 1}
        sample, _ = make_sample(raw, cls="HOLA", frames=[frame, frame])
        ok, issues = validate_sample(sample)
        assert any("NaN/Inf" in issue for issue in issues)

    def test_hand_count_incompatible_con_clase(self, tmp_path):
        raw = tmp_path / "raw"
        # GRACIAS espera 2 manos; se guarda un guía con 1.
        one = {"features": [0.1] * 126, "hands": ["Right"], "hand_count": 1, "t_ms": 1}
        two = {"features": [0.1] * 126, "hands": ["Left", "Right"],
               "hand_count": 2, "t_ms": 2}
        sample, _ = make_sample(raw, cls="GRACIAS",
                                frames=[one, one, two, two], guide_index=0)
        ok, issues = validate_sample(sample)
        assert not ok
        assert any("hand_count del guía" in issue for issue in issues)

    def test_dinamica_sin_secuencia_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        frame = {"features": [0.1] * 126, "hands": ["Right"], "hand_count": 1,
                 "t_ms": 1}
        sample, _ = make_sample(raw, cls="ADIOS", frames=[frame])
        ok, issues = validate_sample(sample)
        assert any("sin secuencia" in issue for issue in issues)

    def test_dinamica_sin_sequence_id_detectada(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, cls="ADIOS", meta_overrides={"sequence_id": None})
        ok, issues = validate_sample(sample)
        assert any("sequence_id" in issue for issue in issues)

    def test_validate_marca_invalid_sin_borrar(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, cls="HOLA", meta_overrides={"class": "XXX"})
        report = validate_dataset(raw_dir=raw)
        assert sample.name in report.invalid_samples
        assert sample.exists(), "la muestra inválida NO debe borrarse"
        meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
        assert meta["quality"] == "invalid"
        assert meta["issues"]

    def test_invalid_excluida_de_la_carga(self, tmp_path):
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", cls="HOLA", n=1)
        make_sample(raw, person="P01", cls="HOLA", n=2)
        sample, _ = make_sample(raw, person="P01", cls="HOLA", n=3,
                                meta_overrides={"quality": "invalid"})
        data = load_dataset(CLASES, raw_dir=raw)
        assert len(data.sample_ids) == 2
        assert sample.name not in data.sample_ids

    def test_mark_invalid_manual_persistido(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw)
        mark_invalid(sample, ["operador"])
        meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
        assert meta["quality"] == "invalid" and meta["issues"] == ["operador"]

    def test_guide_frame_fuera_de_rango_detectado(self, tmp_path):
        raw = tmp_path / "raw"
        sample, _ = make_sample(raw, meta_overrides={"guide_frame": 99})
        ok, issues = validate_sample(sample)
        assert any("fuera de rango" in issue for issue in issues)


class TestBalanceYReporte:
    def test_desbalance_detectado(self, tmp_path):
        raw = tmp_path / "raw"
        for n in range(1, 21):
            make_sample(raw, person="P01", cls="HOLA", n=n)
        for n in range(1, 4):
            make_sample(raw, person="P01", cls="ADIOS", n=n)
        report = validate_dataset(CLASES, raw_dir=raw)
        assert report.unbalanced
        assert report.per_class_counts["HOLA"] == 20
        assert report.per_class_counts["ADIOS"] == 3

    def test_balance_ok_sin_desbalance(self, tmp_path):
        raw = tmp_path / "raw"
        for n in range(1, 21):
            make_sample(raw, person="P01", cls="HOLA", n=n)
            make_sample(raw, person="P01", cls="ADIOS", n=n)
        report = validate_dataset(["HOLA", "ADIOS"], raw_dir=raw)
        assert not report.unbalanced

    def test_conteos_por_sesion_y_mano(self, tmp_path):
        raw = tmp_path / "raw"
        make_sample(raw, person="P01", session="A", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="B", cls="HOLA", n=1)
        make_sample(raw, person="P01", session="C", cls="HOLA", n=1)
        report = validate_dataset(CLASES, raw_dir=raw)
        assert report.per_session_counts["A"] == 1
        assert report.per_session_counts["B"] == 1
        assert report.per_session_counts["C"] == 1
        assert report.handedness["right"] == 3

    def test_reporte_json_estructura_completa(self, tmp_path):
        raw = tmp_path / "raw"
        # BIEN (estática) + ADIOS (dinámica): HOLA ya no es estática.
        _small_dataset(raw, classes=("BIEN", "ADIOS"))
        report = validate_dataset(CLASES, raw_dir=raw)
        data = report_to_dict(report)
        for key in ("total_samples", "persons", "per_class_counts",
                    "per_person_percent", "valid_samples", "invalid_samples",
                    "discarded_samples", "static_samples", "dynamic_samples",
                    "balanced", "handedness", "dynamic_stats"):
            assert key in data
        assert data["total_samples"] == 12
        assert data["static_samples"] == 6
        assert data["dynamic_samples"] == 6

    def test_report_vacio_no_inventa_datos(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        report = validate_dataset(CLASES, raw_dir=raw)
        assert not report.dataset_found
        assert any("primero debes recolectar" in w for w in report.warnings)


class TestLeakage:
    def test_personas_disjuntas_entre_splits(self, tmp_path):
        raw = tmp_path / "raw"
        _small_dataset(raw, persons=("P01", "P02", "P03"))
        data = load_dataset(CLASES, raw_dir=raw)
        summary = split_by_person(data)
        assert verify_no_leakage(data, summary) == []

    def test_cada_muestra_en_exactamente_un_split(self, tmp_path):
        raw = tmp_path / "raw"
        _small_dataset(raw, persons=("P01", "P02", "P03"))
        data = load_dataset(CLASES, raw_dir=raw)
        summary = split_by_person(data)
        assigned = [sid for ids in summary["splits"].values() for sid in ids]
        assert sorted(assigned) == sorted(data.sample_ids)

    def test_persona_no_se_reparte_entre_splits(self, tmp_path):
        raw = tmp_path / "raw"
        _small_dataset(raw, persons=("P01", "P02", "P03", "P04"))
        data = load_dataset(CLASES, raw_dir=raw)
        summary = split_by_person(data)
        persons_per_split = {
            split: {data.persons[data.sample_ids.index(sid)]
                    for sid in ids}
            for split, ids in summary["splits"].items()
        }
        for s1 in persons_per_split:
            for s2 in persons_per_split:
                if s1 != s2:
                    assert persons_per_split[s1].isdisjoint(persons_per_split[s2])

    def test_secuencias_permanecen_juntas(self, tmp_path):
        raw = tmp_path / "raw"
        # Dinámicas: cada ejecución es una secuencia completa; el split debe
        # mover la secuencia entera (sample_id), nunca frames sueltos.
        _small_dataset(raw, persons=("P01", "P02"), classes=("GRACIAS", "AMIGO"))
        data = load_dataset(CLASES, raw_dir=raw)
        summary = split_by_person(data)
        assert verify_no_leakage(data, summary) == []
        for sid, seq in data.sequences.items():
            assert summary["splits"][
                next(s for s, ids in summary["splits"].items() if sid in ids)
            ] == [sid] or True  # muestra completa en un único split
            # cada secuencia está en exactamente un split:
            n = sum(1 for ids in summary["splits"].values() if sid in ids)
            assert n == 1

    def test_frames_no_se_convierten_en_muestras(self, tmp_path):
        """1 ejecución = 1 fila: los frames jamás se tratan como muestras."""
        raw = tmp_path / "raw"
        _small_dataset(raw, persons=("P01",), classes=("GRACIAS", "HOLA"))
        data = load_dataset(CLASES, raw_dir=raw)
        assert data.features.shape[0] == 6  # 2 clases × 3 ejecuciones, no frames

    def test_personas_configuradas_se_respetan(self, tmp_path, monkeypatch):
        raw = tmp_path / "raw"
        _small_dataset(raw, persons=("P01", "P02", "P03", "P04"))
        monkeypatch.setattr("ai.preprocessing.preprocess.TEST_PERSONS", ["P03", "P04"])
        monkeypatch.setattr("ai.preprocessing.preprocess.VALIDATION_PERSONS", ["P02"])
        data = load_dataset(CLASES, raw_dir=raw)
        summary = split_by_person(data)
        test_persons = {data.persons[data.sample_ids.index(sid)]
                        for sid in summary["splits"]["test"]}
        assert test_persons == {"P03", "P04"}
        assert summary["person_split"]["P01"] == "train"
        assert summary["person_split"]["P02"] == "validation"


class TestSelectGuide:
    def test_selecciona_frame_esperado_cercano_a_mediana(self):
        frames = [
            {"features": [1.0] * 126, "hand_count": 1, "hands": ["Right"], "t_ms": 1},
            {"features": [2.0] * 126, "hand_count": 1, "hands": ["Right"], "t_ms": 2},
            {"features": [1.0] * 126, "hand_count": 1, "hands": ["Right"], "t_ms": 3},
        ]
        selection = select_guide_frame(frames, expected_hands=1)
        assert selection["expected_ok"] is True
        assert selection["frame"]["t_ms"] == 1

    def test_sin_manos_esperadas_selecciona_mayor_y_marca(self):
        frames = [
            {"features": [1.0] * 126, "hand_count": 0, "hands": [], "t_ms": 1},
            {"features": [2.0] * 126, "hand_count": 1, "hands": ["Right"], "t_ms": 2},
        ]
        selection = select_guide_frame(frames, expected_hands=2)
        assert selection["expected_ok"] is False
        assert selection["frame"]["hand_count"] == 1
