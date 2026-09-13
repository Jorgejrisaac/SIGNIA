"""Tests del pipeline de datos (F5/F14) con dataset sintético MÍNIMO.

IMPORTANTE (requisito 27): estos datos sintéticos existen SOLO para
comprobar que el pipeline funciona. NUNCA se mezclan con el dataset real
de LSC (viven en carpetas temporales de pytest).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ai.preprocessing.preprocess import (
    DatasetEmptyError,
    build_dataset,
    load_dataset,
    split_by_person,
)
from ai.preprocessing.validate import validate_dataset
from tests._dataset_helpers import make_sample

TMP = "tmp_test_dataset"
CLASSES = ["HOLA", "ADIOS"]


def _build_synthetic_dataset(root: Path, persons: list[str], classes: list[str],
                             per_person_class: int = 3):
    """Dataset sintético en el esquema F14 (directorios de muestra)."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for person in persons:
        for cls in classes:
            for i in range(per_person_class):
                make_sample(raw, person=person, cls=cls, n=i + 1)
    return raw


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    root = tmp_path / TMP
    monkeypatch.setattr("ai.preprocessing.preprocess.RAW_DIR", root / "raw")
    monkeypatch.setattr("ai.preprocessing.preprocess.PREPROCESSED_FEATURES",
                        root / "features.npy")
    monkeypatch.setattr("ai.preprocessing.preprocess.PREPROCESSED_LABELS",
                        root / "labels.npy")
    monkeypatch.setattr("ai.preprocessing.preprocess.PREPROCESSED_SPLITS",
                        root / "splits.json")
    return root


class TestDatasetVacio:
    def test_dataset_vacio_lanza_error_claro(self, synthetic_dataset):
        with pytest.raises(DatasetEmptyError, match="primero debes recolectar"):
            load_dataset(CLASSES)

    def test_validate_reporta_vacio_sin_lanzar(self, synthetic_dataset):
        report = validate_dataset(CLASSES, raw_dir=synthetic_dataset / "raw")
        assert not report.dataset_found
        assert any("primero debes recolectar" in w for w in report.warnings)

    def test_build_dataset_sin_datos_es_exit_code_1(self, synthetic_dataset):
        with pytest.raises(DatasetEmptyError):
            build_dataset(CLASSES)


class TestCarga:
    def test_carga_y_formato(self, synthetic_dataset):
        _build_synthetic_dataset(synthetic_dataset, ["P01"], CLASSES)
        data = load_dataset(CLASSES)
        assert data.features.shape == (6, 126)
        assert data.features.dtype == np.float32
        assert data.labels.shape == (6,)

    def test_muestra_sin_metadata_es_huerfana(self, synthetic_dataset):
        root = synthetic_dataset
        raw = root / "raw"
        sample, _ = make_sample(raw, person="P01", cls="HOLA", n=1,
                                with_metadata=False)
        # Sin muestras válidas no se puede construir dataset:
        with pytest.raises(DatasetEmptyError):
            load_dataset(CLASSES, raw_dir=raw)
        # La validación SÍ reporta el directorio huérfano por separado:
        report = validate_dataset(CLASSES, raw_dir=raw)
        assert report.missing_files
        assert any("metadata.json" in w for w in report.warnings)

    def test_dinamicas_conservan_secuencia_en_la_carga(self, synthetic_dataset):
        _build_synthetic_dataset(synthetic_dataset, ["P01"], ["GRACIAS"])
        data = load_dataset(["GRACIAS"])
        assert data.sequences.keys() == set(data.sample_ids)
        seq = data.sequences[data.sample_ids[0]]
        assert seq["sign_type"] == "dynamic"
        assert seq["frame_count"] > 0


class TestSplitPorPersona:
    def test_dos_personas_no_se_mezclan(self, synthetic_dataset):
        _build_synthetic_dataset(synthetic_dataset, ["P01", "P02"], CLASSES)
        data = load_dataset(CLASSES, raw_dir=synthetic_dataset / "raw")
        summary = split_by_person(data)
        train_ids, test_ids = summary["splits"]["train"], summary["splits"]["test"]
        assert set(train_ids).isdisjoint(set(test_ids))
        train_persons = {person for sid, person in zip(data.sample_ids, data.persons)
                         if sid in train_ids}
        test_persons = {person for sid, person in zip(data.sample_ids, data.persons)
                        if sid in test_ids}
        assert len(train_persons) == 1 and len(test_persons) == 1
        assert train_persons != test_persons

    def test_una_persona_fallback_aleatorio_documentado(self, synthetic_dataset, capsys):
        _build_synthetic_dataset(synthetic_dataset, ["P01"], CLASSES)
        data = load_dataset(CLASSES, raw_dir=synthetic_dataset / "raw")
        summary = split_by_person(data)
        assert summary["random_fallback_single_person"] is True
        assert any(split for split in summary["splits"].values())
        assert "Una sola persona" in capsys.readouterr().out

    def test_tres_personas_reparto_completo(self, synthetic_dataset):
        _build_synthetic_dataset(synthetic_dataset, ["P01", "P02", "P03"], CLASSES)
        data = load_dataset(CLASSES, raw_dir=synthetic_dataset / "raw")
        summary = split_by_person(data)
        assert set(summary["splits"]) == {"train", "validation", "test"}
        for split_ids in summary["splits"].values():
            assert len(split_ids) > 0


class TestBuildCompleto:
    def test_build_dataset_guarda_productos(self, synthetic_dataset):
        _build_synthetic_dataset(synthetic_dataset, ["P01", "P02", "P03"], CLASSES)
        summary = build_dataset(CLASSES)
        assert (synthetic_dataset / "features.npy").exists()
        assert (synthetic_dataset / "labels.npy").exists()
        assert (synthetic_dataset / "splits.json").exists()
        info = json.loads((synthetic_dataset / "splits.json").read_text(encoding="utf-8"))
        assert info["n_train"] + info["n_validation"] + info["n_test"] == summary[
            "n_train"] + summary["n_validation"] + summary["n_test"]
        assert info["leakage_problems"] == []

    def test_build_dataset_una_muestra_por_ejecucion(self, synthetic_dataset):
        """Los frames de una dinámica nunca generan filas independientes."""
        _build_synthetic_dataset(synthetic_dataset, ["P01"], ["GRACIAS"])
        summary = build_dataset(["GRACIAS"])
        assert summary["n_train"] + summary["n_validation"] + summary["n_test"] == 3