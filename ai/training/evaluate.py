"""Evaluación del modelo SIGNIA sobre el TEST SET separado por persona (F7).

Uso:
    python ai/training/evaluate.py

Reporta:
    accuracy, precision, recall, F1 (weighted y macro), métricas por clase,
    matriz de confusión (consola + PNG), n de test, personas de test y
    clases presentes.

Etiqueta de la evaluación (honesta):
    - "GENERALIZACIÓN POR PERSONA"          si el test usa personas distintas
    - "EVALUACIÓN LIMITADA — FALLBACK ALEATORIO"   si solo hay 1 persona

Salidas en ai/training/reports/:
    evaluation_YYYYmmdd_HHMMSS.json   (métricas completas)
    confusion_matrix_YYYYmmdd_HHMMSS.png

El modelo se carga desde ai/models/signia_model.keras; si no existe,
falla con un error claro ("Modelo no entrenado").
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ai.preprocessing.preprocess import (  # noqa: E402
    DatasetEmptyError,
    load_dataset,
    split_by_person,
)
from ai.training.config import (  # noqa: E402
    CLASES,
    LABELS_FILE,
    METADATA_CSV,
    MODEL_FILE,
    RAW_DIR,
    REPORTS_DIR,
)

PERSON_SPLIT_LABEL = "GENERALIZACIÓN POR PERSONA"
RANDOM_FALLBACK_LABEL = "EVALUACIÓN LIMITADA — FALLBACK ALEATORIO"


class ModelNotFoundError(Exception):
    """No existe modelo entrenado para evaluar."""


class EvaluationDataError(Exception):
    """Problemas con los datos de evaluación (test vacío, etc.)."""


def load_trained_model(model_file: Path = MODEL_FILE,
                       labels_file: Path = LABELS_FILE) -> tuple[tf.keras.Model, list[str]]:
    """Carga modelo + etiquetas. Error claro si no existe modelo."""
    if not model_file.exists():
        raise ModelNotFoundError(
            f"Modelo no entrenado: no existe {model_file}. "
            "Ejecuta primero: python ai/training/train.py"
        )
    if not labels_file.exists():
        raise ModelNotFoundError(
            f"Modelo no entrenado: falta {labels_file}. "
            "Ejecuta primero: python ai/training/train.py"
        )
    model = tf.keras.models.load_model(model_file)
    with open(labels_file, encoding="utf-8") as fh:
        classes = json.load(fh)
    return model, classes


def evaluate(classes: list[str] | None = None,
             model_file: Path = MODEL_FILE,
             labels_file: Path = LABELS_FILE,
             metadata_csv: Path = METADATA_CSV,
             raw_dir: Path = RAW_DIR,
             reports_dir: Path | None = None) -> dict:
    """Evalúa el modelo sobre el test set por persona. Devuelve el reporte."""
    classes = classes or list(CLASES)
    reports_dir = reports_dir or REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, trained_classes = load_trained_model(model_file, labels_file)

    # El test set SIEMPRE proviene del split por persona (mismo pipeline).
    dataset = load_dataset(classes, metadata_csv, raw_dir)
    summary = split_by_person(dataset)
    test_ids = set(summary["splits"]["test"])
    test_idx = np.array([sid in test_ids for sid in dataset.sample_ids], dtype=bool)
    if not test_idx.any():
        raise EvaluationDataError(
            "El split por persona no dejó muestras de TEST. "
            "Revisa el dataset y la configuración de personas."
        )

    X_test = dataset.features[test_idx]
    y_true = dataset.labels[test_idx]
    proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(proba, axis=1)

    accuracy = float(accuracy_score(y_true, y_pred))
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro")
    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(trained_classes))),
    ).tolist()
    report_per_class = classification_report(
        y_true, y_pred, labels=list(range(len(trained_classes))),
        target_names=trained_classes, output_dict=True, zero_division=0,
    )
    test_persons = sorted({str(dataset.persons[i]) for i in np.where(test_idx)[0]})
    classes_present = sorted({trained_classes[i] for i in set(y_true.tolist())})

    is_person_split = not summary["random_fallback_single_person"]
    evaluation_label = PERSON_SPLIT_LABEL if is_person_split else RANDOM_FALLBACK_LABEL

    report = {
        "evaluation_label": evaluation_label,
        "is_person_split": is_person_split,
        "n_test_samples": int(test_idx.sum()),
        "test_persons": test_persons,
        "classes_present_in_test": classes_present,
        "trained_classes": trained_classes,
        "metrics": {
            "accuracy": accuracy,
            "precision_weighted": float(weighted[0]),
            "recall_weighted": float(weighted[1]),
            "f1_weighted": float(weighted[2]),
            "precision_macro": float(macro[0]),
            "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]),
        },
        "per_class": {
            cls: {k: v for k, v in entry.items() if k in (
                "precision", "recall", "f1-score", "support")}
            for cls, entry in report_per_class.items()
            if isinstance(entry, dict) and "f1-score" in entry
        },
        "confusion_matrix": cm,
        "model_file": str(model_file),
        "evaluated_at": datetime.now().isoformat(),
    }

    _print_report(report, trained_classes, cm)
    _save_report(report, reports_dir, trained_classes)

    return report


def _print_report(report: dict, classes: list[str], cm: list) -> None:
    print("\n=== EVALUACIÓN SIGNIA (F7) ===")
    print(f"Tipo : {report['evaluation_label']}")
    print(f"Test : {report['n_test_samples']} muestras | "
          f"personas: {report['test_persons']}")
    print(f"Clases presentes en test: {report['classes_present_in_test']}\n")
    m = report["metrics"]
    print(f"Accuracy            : {m['accuracy']:.4f}")
    print(f"Precision (weighted): {m['precision_weighted']:.4f} | "
          f"macro: {m['precision_macro']:.4f}")
    print(f"Recall    (weighted): {m['recall_weighted']:.4f} | "
          f"macro: {m['recall_macro']:.4f}")
    print(f"F1        (weighted): {m['f1_weighted']:.4f} | "
          f"macro: {m['f1_macro']:.4f}")
    print("\nMétricas por clase:")
    for cls in classes:
        if cls in report["per_class"]:
            pc = report["per_class"][cls]
            print(f"  {cls:<12} prec={pc['precision']:.3f} rec={pc['recall']:.3f} "
                  f"f1={pc['f1-score']:.3f} n={int(pc['support'])}")
    print("\nMatriz de confusión (filas=real, columnas=predicha):")
    header = "        " + "".join(f"{c[:6]:>7}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i][:6]:>7} " + "".join(f"{v:>7}" for v in row))


def _save_report(report: dict, reports_dir: Path, classes: list[str]) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"evaluation_{stamp}.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(report["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), labels=classes)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"SIGNIA — Matriz de confusión\n{report['evaluation_label']}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        png_path = reports_dir / f"confusion_matrix_{stamp}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"\nReportes guardados en:\n  {report_path}\n  {png_path}")
    except Exception as exc:  # nunca impide el reporte de consola
        print(f"\n[!] No se pudo generar la matriz de confusión PNG: {exc}")


def main() -> None:
    try:
        evaluate()
    except (ModelNotFoundError, DatasetEmptyError, EvaluationDataError) as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()