"""Entrenamiento del modelo SIGNIA (Fase 6).

Pipeline EXCLUSIVO:
    dataset real → validación → preprocessing → split por persona → entrenamiento → modelo

Uso:
    python ai/training/train.py
    python ai/training/train.py --epochs 300 --units 128,64 --dropout 0.3 --lr 0.001

Salidas:
    ai/models/signia_model.keras      mejor modelo (ModelCheckpoint save_best_only)
    ai/models/labels.json             clases utilizadas
    ai/models/train_config.json       artefacto de reproducibilidad completo
    ai/training/reports/training_log.csv   métricas por epoch (CSVLogger)

Reglas:
    - No modifica el dataset original (solo lectura).
    - Nunca genera datos sintéticos dentro de ai/dataset/.
    - Dataset vacío o inválido → error claro y salida con código 1.
    - Número de clases derivado de la configuración (config.CLASES), nunca
      codificado dentro del modelo.
    - El modelo se entrena con el MISMO preprocesado de inferencia
      (ai/preprocessing/features.py) para que la conversión a TF.js
      (Fase 8) sea una copia exacta.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ai.preprocessing.features import (  # noqa: E402
    ANCHOR_LANDMARK,
    FEATURES_SIZE,
    SCALE_LANDMARK,
)
from ai.preprocessing.preprocess import (  # noqa: E402
    DatasetEmptyError,
    load_dataset,
    split_by_person,
)
from ai.training.config import (  # noqa: E402
    ACTIVATION,
    BATCH_SIZE,
    CLASES,
    DROPOUT_RATE,
    EPOCHS,
    LEARNING_RATE,
    METADATA_CSV,
    MLP_LAYERS,
    MODEL_FILE,
    LABELS_FILE,
    PATIENCE_EARLY_STOPPING,
    PATIENCE_LR_REDUCE,
    RAW_DIR,
    REPORTS_DIR,
    SEED,
    TEST_PERSONS,
    TRAIN_CONFIG_FILE,
    VALIDATION_PERSONS,
    VALIDATION_SPLIT,
)


class TrainingDataError(Exception):
    """Datos insuficientes o inválidos para entrenar."""


@dataclass
class HyperParams:
    """Hiperparámetros entrenamiento (configurables desde config.py o CLI)."""
    units: tuple[int, ...] = tuple(MLP_LAYERS)
    dropout: float = DROPOUT_RATE
    activation: str = ACTIVATION
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    patience_early: int = PATIENCE_EARLY_STOPPING
    patience_lr: int = PATIENCE_LR_REDUCE
    validation_fraction: float = VALIDATION_SPLIT
    seed: int = SEED


@dataclass
class DataPaths:
    """Rutas de entrada/salida (inyectables para pruebas)."""
    metadata_csv: Path = METADATA_CSV
    raw_dir: Path = RAW_DIR
    model_file: Path = MODEL_FILE
    labels_file: Path = LABELS_FILE
    config_file: Path = TRAIN_CONFIG_FILE
    log_csv: Path = Path(REPORTS_DIR) / "training_log.csv"
    test_persons: list[str] = field(default_factory=lambda: list(TEST_PERSONS))
    validation_persons: list[str] = field(
        default_factory=lambda: list(VALIDATION_PERSONS)
    )


def build_model(n_classes: int, hyper: HyperParams) -> tf.keras.Model:
    """Arquitectura MLP configurable. n_classes proviene de la configuración."""
    layers = [tf.keras.Input(shape=(FEATURES_SIZE,), name="input")]
    for i, units in enumerate(hyper.units):
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation=hyper.activation,
                name=f"dense_{i + 1}",
            )
        )
        layers.append(tf.keras.layers.Dropout(hyper.dropout))
    layers.append(
        tf.keras.layers.Dense(n_classes, activation="softmax", name="output")
    )
    model = tf.keras.Sequential(layers, name="signia_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyper.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _class_index_masks(dataset, summary: dict) -> dict[str, np.ndarray]:
    """Máscara booleana por split (mismo orden que dataset.features)."""
    ids = dataset.sample_ids
    return {
        name: np.array([sid in split_ids for sid in ids], dtype=bool)
        for name, split_ids in summary["splits"].items()
    }


def _persons_of(indices: list[int], dataset) -> list[str]:
    return sorted({str(dataset.persons[i]) for i in indices})


def train(classes: list[str] | None = None,
          hyper: HyperParams | None = None,
          paths: DataPaths | None = None) -> dict:
    """Ejecuta el pipeline completo de entrenamiento. Devuelve el resumen."""
    classes = classes or list(CLASES)
    hyper = hyper or HyperParams()
    paths = paths or DataPaths()
    paths.model_file.parent.mkdir(parents=True, exist_ok=True)
    paths.log_csv.parent.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(hyper.seed)

    # 1) Carga + validación del dataset real (no modifica nada).
    dataset = load_dataset(classes, paths.metadata_csv, paths.raw_dir)
    if dataset.anomalies:
        print(f"[!] {len(dataset.anomalies)} anomalías ignoradas durante la carga:")
        for a in dataset.anomalies[:10]:
            print(f"    - {a}")

    # 2) Clases presentes vs. configuradas.
    present = set(np.unique(dataset.labels).tolist())
    missing = [c for i, c in enumerate(classes) if i not in present]
    if len(present) < 2:
        raise TrainingDataError(
            f"Se necesitan al menos 2 clases con muestras para entrenar; "
            f"presentes: {[classes[i] for i in sorted(present)]}. "
            "Recolecta más datos (ai/collection/collect_data.py)."
        )
    if missing:
        print(f"[!] AVISO — clases configuradas SIN muestras en el dataset: "
              f"{missing}. El modelo tendrá {len(classes)} salidas pero "
              "esas clases no fueron entrenadas.")

    # 3) Split por persona (generalización real) o fallback documentado.
    summary = split_by_person(dataset)
    masks = _class_index_masks(dataset, summary)
    train_idx = np.where(masks["train"])[0]
    val_idx = np.where(masks["validation"])[0]
    test_idx = np.where(masks["test"])[0]

    validation_source = "person-split"
    if len(val_idx) == 0 and len(train_idx) > 1:
        print("[!] Sin validation por persona (pocas personas): se reserva "
              f"{hyper.validation_fraction:.0%} del train con seed {hyper.seed} "
              "(documentado en train_config.json).")
        try:
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=min(hyper.validation_fraction, 0.4),
                random_state=hyper.seed,
                stratify=dataset.labels[train_idx],
            )
        except ValueError:
            # Una sola clase en train: stratify no es viable.
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=min(hyper.validation_fraction, 0.4),
                random_state=hyper.seed,
            )
        validation_source = "sampled-from-train"
    if summary["random_fallback_single_person"]:
        validation_source = "random-fallback-single-person"
    if len(val_idx) == 0 or len(test_idx) == 0:
        raise TrainingDataError(
            "No quedan muestras para validation/test tras el split por persona. "
            "Revisa el dataset y la configuración de personas."
        )

    # 4) Construcción del modelo (número de clases = len(classes)).
    X_train, y_train = dataset.features[train_idx], dataset.labels[train_idx]
    X_val, y_val = dataset.features[val_idx], dataset.labels[val_idx]
    model = build_model(n_classes=len(classes), hyper=hyper)
    model.summary()

    # 5) Entrenamiento con EarlyStopping, ModelCheckpoint (best), ReduceLROnPlateau
    #    y registro de métricas por epoch (CSVLogger).
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=hyper.patience_early,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            paths.model_file, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=hyper.patience_lr,
            min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(paths.log_csv, append=False),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=hyper.epochs,
        batch_size=hyper.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # 6) El checkpoint guardó el MEJOR modelo; se recarga para medir con él.
    best = tf.keras.models.load_model(paths.model_file)
    best_metrics = best.evaluate(X_val, y_val, verbose=0)
    best_val_loss, best_val_accuracy = float(best_metrics[0]), float(best_metrics[1])

    # 7) Etiquetas utilizadas (orden de las neuronas de salida).
    with open(paths.labels_file, "w", encoding="utf-8") as fh:
        json.dump(classes, fh, ensure_ascii=False, indent=2)

    # 8) Artefacto de reproducibilidad (requisito: qué se entrenó y con qué).
    artifact = {
        "project": "SIGNIA",
        "training_date": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": tf.keras.__version__,
        "architecture": [
            {"layer": "Input", "shape": (FEATURES_SIZE,)},
            *[
                {"layer": "Dense", "units": u, "activation": hyper.activation}
                for u in hyper.units
            ],
            {"layer": "Dropout", "rate": hyper.dropout},
            {"layer": "Dense", "units": len(classes), "activation": "softmax"},
        ],
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in asdict(hyper).items()},
        "normalization": {
            "method": "origen en muñeca + escala por tamaño de palma "
                      "(idéntico en entrenamiento e inferencia; "
                      "ai/preprocessing/features.py y frontend/js/features.js)",
            "anchor_landmark": ANCHOR_LANDMARK,
            "scale_landmark": SCALE_LANDMARK,
            "feature_size": FEATURES_SIZE,
        },
        "classes": classes,
        "n_classes": len(classes),
        "seed": hyper.seed,
        "dataset": {
            "n_samples": len(dataset.features),
            "per_class": {
                classes[i]: int((dataset.labels == i).sum())
                for i in range(len(classes))
            },
            "n_persons": len(set(dataset.persons.tolist())),
            "persons": sorted(set(dataset.persons.tolist())),
        },
        "persons": {
            "train": _persons_of(train_idx.tolist(), dataset),
            "validation": _persons_of(val_idx.tolist(), dataset),
            "test": _persons_of(test_idx.tolist(), dataset),
        },
        "splits": {
            "n_train": int(len(train_idx)),
            "n_validation": int(len(val_idx)),
            "n_test": int(len(test_idx)),
        },
        "validation_source": validation_source,
        "random_fallback_single_person": summary["random_fallback_single_person"],
        "best_model": {
            "val_loss": best_val_loss,
            "val_accuracy": best_val_accuracy,
            "epochs_trained": len(history.epoch),
            "last_val_accuracy": float(history.history["val_accuracy"][-1]),
        },
    }
    with open(paths.config_file, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Mejor modelo (val_accuracy): {best_val_accuracy:.4f}")
    print(f"Guardado en: {paths.model_file}")
    print(f"Etiquetas  : {paths.labels_file}")
    print(f"Config     : {paths.config_file}")
    print(f"Log epoch  : {paths.log_csv}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento SIGNIA (F6)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--units", type=str, default=None,
                        help="Capas ocultas, p. ej. '128,64'")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    hyper = HyperParams(
        epochs=args.epochs or EPOCHS,
        units=tuple(int(u) for u in args.units.split(",")) if args.units else tuple(MLP_LAYERS),
        dropout=args.dropout if args.dropout is not None else DROPOUT_RATE,
        batch_size=args.batch_size or BATCH_SIZE,
        learning_rate=args.lr if args.lr is not None else LEARNING_RATE,
        seed=args.seed or SEED,
    )
    try:
        train(hyper=hyper)
    except (DatasetEmptyError, TrainingDataError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()