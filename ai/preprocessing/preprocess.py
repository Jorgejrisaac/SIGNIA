"""Preprocesado: carga el dataset (F14), valida y aplica el split por persona.

Uso (después de recolectar datos):
    python ai/preprocessing/preprocess.py

Salida en ai/dataset/processed/:
    - features.npy  : matriz (N, 126) — UN vector por EJECUCIÓN (fotograma guía)
    - labels.npy    : índices de clase (N,)
    - splits.json   : sample_ids por split + asignación por persona

Formato F14 (ai/preprocessing/samples.py):
    - 1 muestra = 1 ejecución = 1 directorio con metadata.json, guide.json y
      frames/ (las dinámicas conservan la SECUENCIA COMPLETA).
    - El vector usado para el MVP es el del fotograma guía de cada ejecución;
      los frames NUNCA se convierten en muestras independientes (evita data
      leakage de persona y de ejecución, especificación F14 §18).

División por persona (requisito 9 / F14 §17):
    - Si se configuran TEST_PERSONS/VALIDATION_PERSONS en config.py, se usan
      esas personas; el resto va a train.
    - Si no, se reparten personas completas: 70/15/15 (train/val/test).
      Con 2 personas: una a train y la otra a test (val = train).
    - Con UNA sola persona: NO hay generalización posible; se advierte y se
      usa un split aleatorio con seed fija (fallback DOCUMENTADO).

El dataset vacío no es un error fatal: muestra el mensaje claro
"Dataset vacío: primero debes recolectar muestras."
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.features import FEATURES_SIZE  # noqa: E402
from ai.preprocessing.samples import (  # noqa: E402
    iter_samples,
    load_frames,
    load_guide,
    load_metadata,
    validate_sample,
)
from ai.training.config import (  # noqa: E402
    CLASES,
    PREPROCESSED_FEATURES,
    PREPROCESSED_LABELS,
    PREPROCESSED_SPLITS,
    RAW_DIR,
    SEED,
    TEST_PERSONS,
    TRAIN_PERSONS,
    VALIDATION_PERSONS,
    VALIDATION_SPLIT,
)


class DatasetEmptyError(Exception):
    """El dataset no existe o no tiene muestras."""


@dataclass
class LoadedDataset:
    features: np.ndarray
    labels: np.ndarray
    persons: np.ndarray
    sample_ids: list[str]
    sequences: dict[str, dict] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


def load_dataset(classes: list[str] | None = None,
                 metadata_csv: Path | None = None,
                 raw_dir: Path | None = None) -> LoadedDataset:
    """Carga UNA muestra por ejecución (fotograma guía). Lanza
    DatasetEmptyError si está vacío.

    metadata_csv es LEGACY del esquema pre-F14: ya no se lee (cada muestra
    lleva su metadata.json). Se conserva el parámetro por compatibilidad con
    llamadas existentes (train/evaluate).
    """
    raw_dir = raw_dir or RAW_DIR
    classes = classes or CLASES
    samples = iter_samples(raw_dir)
    if not samples:
        raise DatasetEmptyError(
            "Dataset vacío: primero debes recolectar muestras "
            "(ai/collection/collect_data.py --person P01 --session A)."
        )

    features, labels, persons, sample_ids, anomalies = [], [], [], [], []
    class_index = {c: i for i, c in enumerate(classes)}
    sequences: dict[str, dict] = {}

    for sample in samples:
        meta = load_metadata(sample)
        sid = meta.get("sample_id", sample.name)
        cls = meta.get("class", "")
        if cls not in class_index:
            anomalies.append(f"clase desconocida '{cls}' (sample {sid})")
            continue
        if meta.get("quality") == "invalid":
            anomalies.append(f"muestra INVALID excluida: {sid}")
            continue
        guide = load_guide(sample)
        vec = np.asarray(guide.get("features"), dtype=np.float32)
        if vec.shape != (FEATURES_SIZE,):
            anomalies.append(f"dimensión inválida en {sid}")
            continue
        features.append(vec)
        labels.append(class_index[cls])
        persons.append(meta.get("person_id", "?"))
        sample_ids.append(sid)
        if meta.get("sign_type") == "dynamic":
            sequences[sid] = {
                "sequence_id": meta.get("sequence_id", sid),
                "sign_type": "dynamic",
                "frame_count": int(meta.get("frame_count", 0)),
                "duration_ms": int(meta.get("duration_ms", 0)),
                "guide_frame": int(meta.get("guide_frame", 0)),
            }

    if not features:
        raise DatasetEmptyError(
            "Dataset vacío (o sin muestras válidas): primero debes recolectar "
            "muestras (ai/collection/collect_data.py)."
        )
    return LoadedDataset(
        features=np.vstack(features).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
        persons=np.array(persons),
        sample_ids=sample_ids,
        sequences=sequences,
        anomalies=anomalies,
    )


def split_by_person(dataset: LoadedDataset) -> dict:
    """Divide por persona. Devuelve dict con los sample_ids por split."""
    persons = sorted(set(dataset.persons.tolist()))
    rng = np.random.default_rng(SEED)

    if TEST_PERSONS or VALIDATION_PERSONS:
        assigned: dict[str, str] = {}
        for p in persons:
            if p in TEST_PERSONS:
                assigned[p] = "test"
            elif p in VALIDATION_PERSONS:
                assigned[p] = "validation"
            elif p in TRAIN_PERSONS:
                assigned[p] = "train"
        unassigned = [p for p in persons if p not in assigned]
        for p in unassigned:
            assigned[p] = "train"
        missing = [p for p in TEST_PERSONS + VALIDATION_PERSONS if p not in persons]
        if missing:
            raise ValueError(
                f"Personas configuradas en config.py que no están en el dataset: "
                f"{missing}. Revisa TEST_PERSONS/VALIDATION_PERSONS."
            )
    elif len(persons) >= 3:
        rng.shuffle(persons)
        n_train = max(1, round(len(persons) * (1 - VALIDATION_SPLIT * 2)))
        n_val = max(1, round(len(persons) * VALIDATION_SPLIT))
        n_test = len(persons) - n_train - n_val
        if n_test < 1 and n_train > 1:
            # Con pocas personas el redondeo 70/15/15 puede dejar test vacío.
            n_train -= 1
            n_test = 1
        assigned = {p: "train" for p in persons[:n_train]}
        assigned.update({p: "validation" for p in persons[n_train:n_train + n_val]})
        assigned.update({p: "test" for p in persons[n_train + n_val:]})
    elif len(persons) == 2:
        assigned = {persons[0]: "train", persons[1]: "test"}
    else:
        # Fallback DOCUMENTADO: una sola persona no permite generalización.
        # Se dividen sus MUESTRAS (80/10/10 con seed fija) para poder
        # ejecutar el pipeline; los resultados NO miden generalización.
        print("[AVISO] Una sola persona en el dataset: split aleatorio por "
              f"muestras (seed {SEED}). Sin separación por persona, los "
              "resultados NO miden generalización entre señantes. "
              "Recolecta con 2+ personas.")
        single_ids = [sid for sid, p in zip(dataset.sample_ids, dataset.persons)
                      if p == persons[0]]
        rng.shuffle(single_ids)
        n_train = max(1, round(len(single_ids) * 0.8))
        n_val = max(1, round(len(single_ids) * 0.1))
        fallback = {
            "train": single_ids[:n_train],
            "validation": single_ids[n_train:n_train + n_val],
            "test": single_ids[n_train + n_val:],
        }
        assigned = {p: "train" for p in persons}

    split_of_person = assigned
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for sid, person in zip(dataset.sample_ids, dataset.persons):
        splits[split_of_person[person]].append(sid)
    if "fallback" in locals():
        splits = fallback

    return {
        "splits": splits,
        "person_split": split_of_person,
        "random_fallback_single_person": len(persons) == 1,
        "seed": SEED,
    }


def verify_no_leakage(dataset: LoadedDataset, summary: dict) -> list[str]:
    """Verifica la integridad de los splits (espec F14 §17 y §18).

    Comprueba que:
    1. Cada muestra (ejecución) está en EXACTAMENTE un split.
    2. Cada persona está en EXACTAMENTE un split (nunca P01 train + P01 test).
    3. Las secuencias completas permanecen juntas (en este esquema 1 muestra
       = 1 secuencia; el check protege también un futuro agrupamiento real).

    Devuelve la lista de problemas; vacía = sin leakage.
    """
    problems: list[str] = []
    splits = summary["splits"]
    split_names = set(splits)
    id_to_split = {sid: split for split, ids in splits.items() for sid in ids}

    # 1) cada sample en exactamente un split
    seen = {}
    for sid in dataset.sample_ids:
        if sid not in id_to_split:
            problems.append(f"Muestra sin split asignado: {sid}")
            continue
        seen[sid] = id_to_split[sid]
    for sid, split in id_to_split.items():
        if sid not in set(dataset.sample_ids):
            problems.append(f"Split contiene una muestra inexistente: {sid}")

    # 2) cada persona en un único split
    person_split: dict[str, str] = {}
    for sid, person in zip(dataset.sample_ids, dataset.persons):
        split = id_to_split.get(sid)
        if split is None:
            continue
        if person in person_split and person_split[person] != split:
            problems.append(
                f"LEAKAGE DE PERSONA: {person} en {person_split[person]} y {split}"
            )
        person_split[person] = split

    # 3) secuencias completas juntas (1 muestra = 1 secuencia en F14)
    for sid, seq in dataset.sequences.items():
        split = id_to_split.get(sid)
        if split is None:
            continue
        for other_sid, other_seq in dataset.sequences.items():
            if other_sid == sid:
                continue
            if other_seq.get("sequence_id") == seq.get("sequence_id"):
                if id_to_split.get(other_sid) != split:
                    problems.append(
                        f"LEAKAGE DE EJECUCIÓN: secuencia {seq['sequence_id']} "
                        f"repartida entre splits"
                    )
    return problems


def build_dataset(classes: list[str] | None = None) -> dict:
    """Carga, divide y guarda el dataset preprocesado. Devuelve el resumen."""
    classes = classes or CLASES
    dataset = load_dataset(classes)
    if dataset.anomalies:
        print(f"[!] {len(dataset.anomalies)} anomalías ignoradas (ver reporte):")
        for a in dataset.anomalies[:20]:
            print(f"    - {a}")

    summary = split_by_person(dataset)
    splits = summary["splits"]
    summary["leakage_problems"] = verify_no_leakage(dataset, summary)
    id_to_person = dict(zip(dataset.sample_ids, dataset.persons))
    summary["per_class"] = {
        classes[i]: int((dataset.labels == i).sum()) for i in range(len(classes))
    }
    summary["n_train"] = len(splits["train"])
    summary["n_validation"] = len(splits["validation"])
    summary["n_test"] = len(splits["test"])

    PREPROCESSED_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    np.save(PREPROCESSED_FEATURES, dataset.features)
    np.save(PREPROCESSED_LABELS, dataset.labels)
    with open(PREPROCESSED_SPLITS, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n=== PREPROCESADO COMPLETADO ===")
    print(f"Muestras: {len(dataset.features)} | Clases: {classes}")
    print(f"Splits  : train={summary['n_train']}  "
          f"validation={summary['n_validation']}  test={summary['n_test']}")
    if summary["leakage_problems"]:
        print(f"[!] {len(summary['leakage_problems'])} problema(s) de leakage:")
        for problem in summary["leakage_problems"][:10]:
            print(f"    - {problem}")
    print(f"Guardado: {PREPROCESSED_FEATURES}")
    return summary


if __name__ == "__main__":
    try:
        build_dataset()
    except DatasetEmptyError as exc:
        print(f"\n[VACÍO] {exc}")
        sys.exit(1)