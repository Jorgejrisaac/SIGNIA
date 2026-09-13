"""Tests de F6 (entrenamiento) y F7 (evaluación).

Usan mini-datasets SINTÉTICOS en carpetas temporales de pytest,
SOLO para comprobar que el pipeline funciona (requisito 27).
NUNCA tocan ai/dataset/ ni presentan resultados como reales.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from ai.preprocessing.preprocess import DatasetEmptyError  # noqa: E402
import ai.training.evaluate as eval_mod  # noqa: E402
from ai.training.evaluate import (  # noqa: E402
    PERSON_SPLIT_LABEL,
    RANDOM_FALLBACK_LABEL,
    ModelNotFoundError,
    evaluate,
)
from ai.training.train import (  # noqa: E402
    DataPaths,
    HyperParams,
    TrainingDataError,
    train,
)
from tests._dataset_helpers import make_sample  # noqa: E402

CLASSES = ["HOLA", "ADIOS"]

# Hiperparámetros mínimos para que los tests sean rápidos.
TEST_HYPER = HyperParams(
    units=(16, 8), dropout=0.1, epochs=3, batch_size=16,
    learning_rate=1e-2, patience_early=2, patience_lr=2,
    validation_fraction=0.2, seed=42,
)


def _make_synthetic(root: Path, persons: list[str],
                    classes: list[str] = CLASSES,
                    per_person_class: int = 6, noise: float = 0.02) -> Path:
    """Crea un mini-dataset separable en tmp (esquema F14): features con
    valor medio +1.0 para la clase 0 y -1.0 para la clase 1 (+ ruido)."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for person in persons:
        for cls_idx, cls in enumerate(classes):
            for i in range(per_person_class):
                vec = rng.normal(1.0 if cls_idx == 0 else -1.0, noise,
                                 size=63).astype(np.float32)
                vec = np.concatenate([np.zeros(63, dtype=np.float32), vec])
                frame = {"features": [float(v) for v in vec],
                         "hands": ["Right"], "hand_count": 1, "t_ms": (i + 1) * 33}
                make_sample(raw, person=person, cls=cls, n=i + 1,
                            frames=[frame, frame, frame], guide_index=0)
    return raw


def _paths(root: Path) -> DataPaths:
    return DataPaths(
        metadata_csv=root / "metadata.csv",
        raw_dir=root / "raw",
        model_file=root / "models" / "signia_model_test.keras",
        labels_file=root / "models" / "labels_test.json",
        config_file=root / "models" / "train_config_test.json",
        log_csv=root / "reports" / "training_log_test.csv",
    )


def _eval_cm_sum(cm) -> int:
    return int(np.array(cm).sum())


class TestF6Entrenamiento:
    def test_dataset_vacio_error_claro(self, tmp_path):
        paths = _paths(tmp_path)
        paths.metadata_csv = tmp_path / "no_existe.csv"
        with pytest.raises(DatasetEmptyError, match="primero debes recolectar"):
            train(CLASSES, TEST_HYPER, paths)

    def test_features_incorrectas_error_claro(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir(parents=True)
        # Modelos de features de 100 dimensiones (formato inválido: debe ser 126).
        rng = np.random.default_rng(5)
        for i in range(4):
            vec = rng.random(100).astype(np.float32)
            frame = {"features": [float(v) for v in vec],
                     "hands": ["Right"], "hand_count": 1, "t_ms": i * 33}
            make_sample(raw, person="P01", cls="HOLA", n=i + 1,
                        frames=[frame, frame], guide_index=0)
        with pytest.raises(DatasetEmptyError, match="sin muestras válidas"):
            train(CLASSES, TEST_HYPER, _paths(tmp_path))

    def test_una_sola_clase_error_claro(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1", "P2"], ["HOLA"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        with pytest.raises(TrainingDataError, match="al menos 2 clases"):
            train(["HOLA", "ADIOS"], TEST_HYPER, paths)

    def test_clases_faltantes_aviso_pero_entrena(self, tmp_path, capsys):
        # Dataset con 2 clases presentes; la configuración tiene 10.
        raw = _make_synthetic(tmp_path, ["P1", "P2"], ["HOLA", "ADIOS"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        from ai.training.config import CLASES as CLASES_10
        artifact = train(CLASES_10, TEST_HYPER, paths)
        out = capsys.readouterr().out
        assert "SIN muestras en el dataset" in out
        assert artifact["n_classes"] == 10

    def test_artefactos_completos(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1", "P2"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        artifact = train(CLASSES, TEST_HYPER, paths)

        assert paths.model_file.exists()                       # modelo guardado
        assert paths.config_file.exists()
        assert paths.log_csv.exists()
        with open(paths.labels_file, encoding="utf-8") as fh:
            assert json.load(fh) == CLASSES                   # etiquetas guardadas

        cfg = json.loads(paths.config_file.read_text(encoding="utf-8"))
        for key in ("python_version", "tensorflow_version", "keras_version",
                    "seed", "classes", "normalization", "persons",
                    "hyperparameters", "training_date"):
            assert key in cfg, f"falta {key} en artefacto"
        assert cfg["seed"] == TEST_HYPER.seed
        assert cfg["n_classes"] == len(CLASSES)
        assert cfg["normalization"]["feature_size"] == 126
        assert cfg["validation_source"] == "sampled-from-train"  # 2 personas
        assert cfg["best_model"]["val_accuracy"] > 0.5           # separable
        assert artifact["best_model"]["epochs_trained"] == 3

        # Métricas por epoch registradas (CSVLogger): 1 encabezado + 3 epochs.
        # Keras 3 inserta líneas vacías entre filas: se filtran.
        lines = [l for l in paths.log_csv.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1 + 3

    def test_modelo_cargable_y_predice(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1", "P2"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        train(CLASSES, TEST_HYPER, paths)

        model = tf.keras.models.load_model(paths.model_file)    # cargable
        X = np.random.default_rng(1).normal(0, 1, size=(5, 126)).astype(np.float32)
        proba = model.predict(X, verbose=0)
        assert proba.shape == (5, 2)                            # n salidas = n clases
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)   # softmax


class TestF7Evaluacion:
    def test_sin_modelo_error_claro(self, tmp_path):
        with pytest.raises(ModelNotFoundError, match="Modelo no entrenado"):
            evaluate(CLASSES, model_file=tmp_path / "no.keras",
                     labels_file=tmp_path / "no.json")

    def test_evaluacion_generalizacion_por_persona(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1", "P2"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        train(CLASSES, TEST_HYPER, paths)

        report = evaluate(CLASSES, model_file=paths.model_file,
                          labels_file=paths.labels_file,
                          metadata_csv=paths.metadata_csv, raw_dir=paths.raw_dir,
                          reports_dir=tmp_path / "reports")

        assert report["evaluation_label"] == PERSON_SPLIT_LABEL
        assert report["is_person_split"] is True
        m = report["metrics"]
        assert 0.0 < m["accuracy"] <= 1.0
        assert all(0.0 <= v <= 1.0 for v in
                   (m["precision_weighted"], m["recall_weighted"], m["f1_weighted"]))
        assert report["n_test_samples"] > 0
        assert len(report["test_persons"]) == 1                # test = 1 persona completa
        assert set(report["classes_present_in_test"]) == set(CLASSES)

        cm = report["confusion_matrix"]
        assert np.array(cm).shape == (2, 2)
        assert _eval_cm_sum(cm) == report["n_test_samples"]    # confusión coherente
        assert report["per_class"]["HOLA"]["f1-score"] > 0.5

        # Reportes persistidos en disco
        reports = list((tmp_path / "reports").glob("evaluation_*.json"))
        pngs = list((tmp_path / "reports").glob("confusion_matrix_*.png"))
        assert reports and pngs

    def test_evaluacion_fallback_una_persona(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        train(CLASSES, TEST_HYPER, paths)
        report = evaluate(CLASSES, model_file=paths.model_file,
                          labels_file=paths.labels_file,
                          raw_dir=paths.raw_dir)
        assert report["evaluation_label"] == RANDOM_FALLBACK_LABEL
        assert report["is_person_split"] is False
        assert report["test_persons"] == ["P1"]

    def test_matriz_confusion_con_clase_ausente_en_test(self, tmp_path,
                                                        monkeypatch):
        """Regresión: una clase con support 0 en test NO debe desaparecer
        de la matriz de confusión (sklearn la descarta si no se pasa
        labels= explícito, desalineando filas/columnas con trained_classes)."""
        raw = _make_synthetic(tmp_path, ["P1"], per_person_class=20)
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        train(CLASSES, TEST_HYPER, paths)
        real_split = eval_mod.split_by_person

        def solo_adios_en_test(dataset):
            summary = real_split(dataset)
            summary["splits"]["test"] = [
                sid for sid in summary["splits"]["test"] if "ADIOS" in sid]
            return summary

        monkeypatch.setattr(eval_mod, "split_by_person", solo_adios_en_test)
        report = evaluate(CLASSES, model_file=paths.model_file,
                          labels_file=paths.labels_file,
                          raw_dir=paths.raw_dir)

        assert report["per_class"]["HOLA"]["support"] == 0
        cm = np.array(report["confusion_matrix"])
        assert cm.shape == (2, 2)                  # nunca pierde la fila/columna
        assert int(cm.sum()) == report["n_test_samples"]
        assert cm[0, :].sum() == 0                 # fila HOLA existe y está alineada