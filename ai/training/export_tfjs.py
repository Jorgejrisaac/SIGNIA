"""Exportación del modelo Keras entrenado a TensorFlow.js (Fase 8).

Pipeline:
    ai/models/signia_model.keras  (modelo entrenado y evaluado)
    → tensorflowjs.converters.save_keras_model
    → frontend/models/signia/{model.json, shards, metadata.json}

El modelo SE CARGA y se convierte; NO se vuelve a entrenar.

La metadata (metadata.json) es la ÚNICA fuente de verdad de:
    - orden de clases (los dos runtimes la leen)
    - configuración de normalización (muñeca 0, palma 9, 126 features)
    - procedencia del modelo (fecha, TF, Keras, seed, personas, métrica)

Uso:
    python ai/training/export_tfjs.py
    python ai/training/export_tfjs.py --model-file ruta --out-dir frontend/models/signia
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
import tensorflowjs as tfjs  # noqa: E402

from ai.preprocessing.features import (  # noqa: E402
    ANCHOR_LANDMARK,
    COORDS_PER_LANDMARK,
    FEATURES_SIZE,
    HAND_CONNECTIONS,
    LANDMARKS_PER_HAND,
    SCALE_LANDMARK,
)
from ai.training.config import (  # noqa: E402
    FRONTEND_METADATA_FILE,
    FRONTEND_MODELS_DIR,
    MODEL_FILE,
    MODEL_NAME,
    MODEL_VERSION,
    TRAIN_CONFIG_FILE,
)


class ExportError(Exception):
    """No se puede exportar el modelo (falta modelo o configuración)."""


def _normalize_keras3_topology(model_json: Path) -> None:
    """Keras 3 serializa el InputLayer como `batch_shape`; TensorFlow.js
    espera `batch_input_shape` (formato Keras 2). Sin esta normalización
    el navegador no podría cargar el model.json exportado."""
    with open(model_json, encoding="utf-8") as fh:
        topology = json.load(fh)
    layers = (topology.get("modelTopology", {})
              .get("model_config", {}).get("config", {}).get("layers", []))
    changed = any(
        layer.get("class_name") == "InputLayer" and "batch_shape" in layer.get("config", {})
        for layer in layers
    )
    if not changed:
        return
    for layer in layers:
        if layer.get("class_name") == "InputLayer":
            cfg = layer["config"]
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
    with open(model_json, "w", encoding="utf-8") as fh:
        json.dump(topology, fh)


def load_train_config(config_file: Path | None = None) -> dict:
    """Artefacto de reproducibilidad generado por train.py."""
    config_file = config_file or TRAIN_CONFIG_FILE
    if not config_file.exists():
        raise ExportError(
            f"Falta {config_file}: primero entrena el modelo "
            "(python ai/training/train.py)."
        )
    with open(config_file, encoding="utf-8") as fh:
        return json.load(fh)


def export_tfjs(model_file: Path | None = None,
                out_dir: Path | None = None,
                model_version: int = MODEL_VERSION,
                train_config_file: Path | None = None) -> dict:
    """Convierte el modelo Keras evaluado a TensorFlow.js. Devuelve la metadata."""
    model_file = Path(model_file) if model_file else MODEL_FILE
    out_dir = Path(out_dir) if out_dir else FRONTEND_MODELS_DIR
    if not model_file.exists():
        raise ExportError(
            f"Modelo no entrenado: no existe {model_file}. "
            "Ejecuta primero: python ai/training/train.py"
        )

    model = tf.keras.models.load_model(model_file)
    in_shape = tuple(d if d is None else int(d) for d in model.input_shape)
    out_shape = tuple(d if d is None else int(d) for d in model.output_shape)
    if in_shape != (None, FEATURES_SIZE):
        raise ExportError(
            f"Input shape inesperado: {in_shape} (se esperaba (None, {FEATURES_SIZE})). "
            "El modelo no corresponde al formato SIGNIA."
        )
    n_classes = out_shape[-1]

    train_config = load_train_config(train_config_file)
    classes = train_config.get("classes")
    if not classes or len(classes) != n_classes:
        raise ExportError(
            f"Clases de train_config.json ({len(classes or [])}) no coinciden "
            f"con la salida del modelo ({n_classes}). No se exporta un modelo "
            "inconsistente."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # Conversión oficial de la librería (misma versión instalada en este repo).
    tfjs.converters.save_keras_model(model, str(out_dir))
    # Keras 3 no es directamente cargable por TF.js: normaliza el model.json.
    _normalize_keras3_topology(out_dir / "model.json")

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": int(model_version),
        "exported_at": datetime.now().isoformat(),
        "training_date": train_config.get("training_date"),
        "python_version": train_config.get("python_version"),
        "tensorflow_version": train_config.get("tensorflow_version"),
        "keras_version": train_config.get("keras_version"),
        "architecture": train_config.get("architecture"),
        "classes": classes,
        "n_classes": int(n_classes),
        "input_shape": list(in_shape),
        "output_shape": list(out_shape),
        "n_features": FEATURES_SIZE,
        "normalization": {
            "method": "origen en muñeca + escala por tamaño de palma",
            "wrist_index": ANCHOR_LANDMARK,
            "palm_index": SCALE_LANDMARK,
            "landmarks_per_hand": LANDMARKS_PER_HAND,
            "coords_per_landmark": COORDS_PER_LANDMARK,
            "hands": 2,
            "hand_order": ["left", "right"],
            "missing_hand_fill": "zeros",
            "feature_size": FEATURES_SIZE,
        },
        "seed": train_config.get("seed"),
        "persons": train_config.get("persons"),
        "validation_source": train_config.get("validation_source"),
        "random_fallback_single_person": train_config.get(
            "random_fallback_single_person"),
        "best_val_accuracy": train_config.get("best_model", {}).get("val_accuracy"),
        "source_keras": str(Path(model_file)),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    manifest_files = sorted(p.name for p in out_dir.glob("*.bin"))
    print("\n=== EXPORTACIÓN A TENSORFLOW.JS COMPLETADA ===")
    print(f"model.json   : {out_dir / 'model.json'}")
    print(f"shards       : {manifest_files}")
    print(f"metadata.json: {out_dir / 'metadata.json'}")
    print(f"classes      : {classes}")
    print(f"shapes       : in={in_shape} out={out_shape}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportación TF.js (F8)")
    parser.add_argument("--model-file", default=str(MODEL_FILE),
                        help="Modelo Keras .keras entrenado")
    parser.add_argument("--out-dir", default=str(FRONTEND_MODELS_DIR),
                        help="Directorio de salida del modelo TF.js")
    parser.add_argument("--model-version", type=int, default=MODEL_VERSION)
    args = parser.parse_args()
    try:
        export_tfjs(model_file=args.model_file, out_dir=args.out_dir,
                    model_version=args.model_version)
    except ExportError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()