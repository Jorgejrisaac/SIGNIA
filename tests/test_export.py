"""Tests de la Fase 8: exportación Keras→TF.js y paridad Python↔JavaScript.

El modelo TF.js se exporta desde un modelo entrenado con mini-datos
SINTÉTICOS en carpetas temporales (nunca ai/dataset/): verifica el
pipeline de conversión y la paridad numérica, no la precisión LSC.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402

from ai.training.export_tfjs import ExportError, export_tfjs  # noqa: E402
from ai.training.train import DataPaths, HyperParams, train  # noqa: E402
from ai.preprocessing.features import hands_to_features  # noqa: E402

from tests.test_training import _make_synthetic, _paths  # noqa: E402

CLASSES = ["HOLA", "ADIOS"]
PARITY_JS_DIR = PROJECT_ROOT / "tests" / "js_parity"
NODE = shutil.which("node")


def _train_and_export(tmp_path: Path, persons: list[str] = ("P1", "P2", "P3")):
    raw = _make_synthetic(tmp_path, list(persons))
    paths = _paths(tmp_path)
    paths.raw_dir = raw
    hyper = HyperParams(units=(16, 8), dropout=0.1, epochs=3, batch_size=16,
                        learning_rate=1e-2, patience_early=2, patience_lr=2, seed=42)
    train(CLASSES, hyper, paths)
    out_dir = tmp_path / "tfjs"
    metadata = export_tfjs(model_file=paths.model_file, out_dir=out_dir,
                           model_version=1, train_config_file=paths.config_file)
    return paths, out_dir, metadata


def _raw_hand(rng: np.random.Generator, seed_offset: int) -> list:
    """Landmarks crudos reproducibles (21 x [x, y, z])."""
    base = rng.uniform(0.15, 0.85, size=(21, 3)) + seed_offset * 0.01
    base[0] = [0.5, 0.7, 0.0]     # muñeca
    base[9] = [0.5, 0.55, 0.0]    # MCP medio (palma)
    return base.tolist()


def _write_cases(model, out_dir: Path) -> Path:
    """Fixture compartido: entradas idénticas para Python y JavaScript."""
    rng = np.random.default_rng(2026)
    left_2 = _raw_hand(rng, 1)
    right_1 = _raw_hand(rng, 2)
    right_2 = _raw_hand(rng, 3)
    left_1 = _raw_hand(rng, 4)
    near = [[0.5 + 0.05 * (i % 7), 0.5 - 0.04 * i, 0.02 * (i % 3) - 0.01]
            for i in range(21)]
    near[0] = [0.5, 0.5, 0.0]
    near[9] = [0.5, 0.35, 0.0]
    degenerate = [[0.5, 0.5, 0.0]] * 21  # palma de tamaño ~0 → EPSILON

    raw_pairs = [
        ("one_right", None, right_1),
        ("one_left", left_1, None),
        ("two_hands", left_2, right_2),
        ("near_range", None, near),
        ("degenerate_palm", None, degenerate),
        ("negative_z", None, [[x, y, -0.1 * (i + 1)] for i, (x, y, _) in enumerate(right_2)]),
    ]
    vectors = []
    normalization_cases = []
    for name, left, right in raw_pairs:
        vec = hands_to_features(
            np.array(left) if left else None,
            np.array(right) if right else None,
        )
        vectors.append(vec)
        normalization_cases.append({
            "name": name, "left": left, "right": right,
            "expected_features": vec.tolist(),
        })

    # Vectores ya normalizados (cercanos al rango esperado del modelo).
    # Solo van a la sección de predicción: no son landmarks crudos, así que
    # la normalización (mano ausente → ceros) no aplica.
    rng2 = np.random.default_rng(99)
    for i in range(3):
        v = rng2.normal(0, 1.0, size=126).astype(np.float32)
        v[63:] *= 0.5
        vectors.append(v)

    X = np.stack(vectors)
    probs = model.predict(X, verbose=0)
    cases = [
        {"name": name, "features": v.tolist(), "expected": p.tolist()}
        for name, v, p in zip(
            [n for n, _, _ in raw_pairs] + [f"normalized_vector_{i}" for i in range(3)],
            X, probs,
        )
    ]
    cases_file = out_dir / "parity_cases.json"
    with open(cases_file, "w", encoding="utf-8") as fh:
        json.dump({
            "expected_classes": CLASSES,
            "normalization_cases": normalization_cases,
            "cases": cases,
        }, fh)
    return cases_file


def _run_parity(tmp_path, out_dir: Path, cases_file: Path, extra: list | None = None) -> subprocess.CompletedProcess:
    cmd = [NODE, str(PARITY_JS_DIR / "parity.mjs"),
           "--model-dir", str(out_dir), "--cases", str(cases_file)]
    if extra:
        cmd += extra
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          timeout=300, cwd=tmp_path)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f8")
    paths, out_dir, metadata = _train_and_export(tmp)
    model = tf.keras.models.load_model(paths.model_file)
    cases_file = _write_cases(model, out_dir)
    return {"tmp": tmp, "out": out_dir, "metadata": metadata, "cases": cases_file}


class TestConversion:
    def test_artefactos_model_json_y_shards(self, exported):
        assert (exported["out"] / "model.json").exists()
        shards = list(exported["out"].glob("*.bin"))
        assert shards, "no hay shards binarios"
        assert (exported["out"] / "metadata.json").exists()

    def test_metadata_completa(self, exported):
        meta = exported["metadata"]
        for key in ("model_name", "model_version", "exported_at", "training_date",
                    "python_version", "tensorflow_version", "keras_version",
                    "architecture", "classes", "n_classes", "input_shape",
                    "output_shape", "n_features", "normalization", "seed",
                    "persons", "validation_source", "best_val_accuracy"):
            assert key in meta, f"falta {key}"
        assert meta["classes"] == CLASSES
        assert meta["n_features"] == 126
        assert meta["normalization"]["wrist_index"] == 0
        assert meta["normalization"]["palm_index"] == 9
        assert meta["input_shape"] == [None, 126]
        assert meta["output_shape"] == [None, 2]
        assert meta["model_version"] == 1

    def test_sin_modelo_error_claro(self, tmp_path):
        with pytest.raises(ExportError, match="Modelo no entrenado"):
            export_tfjs(model_file=tmp_path / "no.keras", out_dir=tmp_path / "out")

    def test_sin_train_config_error_claro(self, tmp_path):
        # Existe modelo pero no el artefacto de train_config.
        from tests.test_training import _make_synthetic
        raw = _make_synthetic(tmp_path, ["P1", "P2"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        hyper = HyperParams(units=(8, 4), dropout=0.1, epochs=1, batch_size=16, seed=42)
        train(CLASSES, hyper, paths)
        paths.config_file.unlink()  # falta SOLO el artefacto de esta prueba
        with pytest.raises(ExportError, match="Falta"):
            export_tfjs(model_file=paths.model_file, out_dir=tmp_path / "out2",
                        train_config_file=paths.config_file)

    def test_clases_inconsistentes_error_claro(self, tmp_path):
        raw = _make_synthetic(tmp_path, ["P1", "P2"])
        paths = _paths(tmp_path)
        paths.raw_dir = raw
        hyper = HyperParams(units=(8, 4), dropout=0.1, epochs=1, batch_size=16, seed=42)
        train(CLASSES, hyper, paths)

        # Corrompe el artefacto de entrenamiento (clases de tamaño incorrecto).
        cfg = json.loads(paths.config_file.read_text(encoding="utf-8"))
        cfg["classes"] = CLASSES + ["EXTRA"]
        cfg["n_classes"] = len(cfg["classes"])
        paths.config_file.write_text(json.dumps(cfg), encoding="utf-8")

        with pytest.raises(ExportError, match="no coinciden"):
            export_tfjs(model_file=paths.model_file, out_dir=tmp_path / "tfjs_bad",
                        train_config_file=paths.config_file)


@pytest.mark.skipif(NODE is None, reason="Node.js no está instalado")
class TestParidadJS:
    def test_carga_y_paridad_completa(self, exported):
        result = _run_parity(exported["tmp"], exported["out"], exported["cases"])
        assert result.returncode == 0, f"node falló:\n{result.stdout}\n{result.stderr}"
        assert "PASS — Python ↔ TF.js parity" in result.stdout

    def test_deteccion_de_mismatch(self, exported):
        # Contamina las probabilidades esperadas: el detector DEBE fallar.
        bad = json.loads(exported["cases"].read_text(encoding="utf-8"))
        for c in bad["cases"]:
            c["expected"][0] = (c["expected"][0] + 0.5) % 1.0
        bad_file = exported["tmp"] / "parity_cases_bad.json"
        bad_file.write_text(json.dumps(bad), encoding="utf-8")
        result = _run_parity(exported["tmp"], exported["out"], bad_file)
        assert result.returncode != 0
        assert "FAIL — maximum probability difference" in result.stdout

    def test_modelo_tfjs_no_cargable_error_claro(self, exported):
        empty = exported["tmp"] / "tfjs_empty"
        empty.mkdir(exist_ok=True)
        (empty / "model.json").write_text("{", encoding="utf-8")  # JSON roto
        result = _run_parity(exported["tmp"], empty, exported["cases"])
        assert result.returncode != 0
        assert "FAIL" in (result.stdout + result.stderr)