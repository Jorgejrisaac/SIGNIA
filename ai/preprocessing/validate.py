"""Validación del dataset de SIGNIA (esquema F14: muestras por carpeta).

Cada MUESTRA = una ejecución (ai/preprocessing/samples.py). La validación
recorre raw/P/S/CLASE/SAMPLE y aplica los chequeos de la especificación F14
§10 a cada muestra:

    clase válida · persona válida · timestamp válido · landmarks válidos ·
    dimensiones correctas · sin NaN/Inf · rango razonable · mano esperada ·
    cantidad de manos compatible con la clase · metadata válida

Si una muestra falla se MARCA quality='invalid' + issues[] en su
metadata.json (NUNCA se borra silenciosamente). Las descartadas por el
operador (tecla D) viven en raw/_discarded/ y se cuentan por separado.

Funciona con dataset vacío: reporta el estado y devuelve errores claros.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from ai.preprocessing.samples import (
    iter_discarded,
    iter_samples,
    load_frames,
    load_guide,
    load_metadata,
    mark_invalid,
    validate_sample,
)
from ai.training.config import CLASES, RAW_DIR, SESSIONS

# Indice de bloque por etiqueta de mano MediaPipe en el vector de features.
HANDEDNESS_GUIDE_LABELS = ("left", "right", "both", "none")


@dataclass
class DatasetReport:
    dataset_found: bool = False
    total_samples: int = 0
    classes: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    per_class_counts: dict[str, int] = field(default_factory=dict)
    per_person_counts: dict[str, int] = field(default_factory=dict)
    per_session_counts: dict[str, int] = field(default_factory=dict)
    per_class_person: dict[str, int] = field(default_factory=dict)
    invalid_samples: list[str] = field(default_factory=list)
    discarded_samples: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    invalid_vectors: list[str] = field(default_factory=list)
    unbalanced: bool = False
    handedness: dict[str, int] = field(default_factory=dict)
    dynamic_stats: dict[str, dict] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "dataset_found": self.dataset_found,
            "total_samples": self.total_samples,
            "classes": self.classes,
            "persons": self.persons,
            "sessions": self.sessions,
            "per_class_counts": self.per_class_counts,
            "per_person_counts": self.per_person_counts,
            "per_session_counts": self.per_session_counts,
            "per_class_person": self.per_class_person,
            "invalid_samples": self.invalid_samples,
            "discarded_samples": self.discarded_samples,
            "missing_files": self.missing_files,
            "invalid_vectors": self.invalid_vectors,
            "unbalanced": self.unbalanced,
            "handedness": self.handedness,
            "dynamic_stats": self.dynamic_stats,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def read_metadata(metadata_csv: Path) -> list[dict]:
    """LEGACY (esquema pre-F14): se conserva solo para compatibilidad."""
    if not metadata_csv.exists():
        return []
    with open(metadata_csv, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def validate_dataset(classes: list[str] | None = None,
                     raw_dir: Path = RAW_DIR,
                     metadata_csv: Path | None = None,
                     apply_invalid: bool = True) -> DatasetReport:
    """Valida la estructura del dataset. Nunca lanza por dataset vacío.

    apply_invalid=True persiste quality='invalid' + issues[] en las muestras
    que fallan algún chequeo (espec F14 §10: no borrar en silencio).
    """
    classes = classes or CLASES
    report = DatasetReport()

    # Directorios de muestra sin metadata.json (huérfanos / recorte incompleto)
    # se detectan AUNQUE no haya ninguna muestra válida.
    if raw_dir.exists():
        for person in raw_dir.iterdir():
            if not person.is_dir() or person.name.startswith("_"):
                continue
            for session in person.iterdir():
                if not session.is_dir():
                    continue
                for cls_dir in session.iterdir():
                    if not cls_dir.is_dir():
                        continue
                    for entry in cls_dir.iterdir():
                        if entry.is_dir() and not (entry / "metadata.json").exists() \
                                and (entry / "guide.json").exists():
                            report.missing_files.append(str(entry))
        if report.missing_files:
            report.warnings.append(
                f"{len(report.missing_files)} directorio(s) de muestra sin "
                "metadata.json (huérfanos): revisar recorte/estado."
            )

    samples = iter_samples(raw_dir)
    if not samples:
        report.warnings.append(
            "Dataset vacío: primero debes recolectar muestras "
            "(ai/collection/collect_data.py --person P01 --session A)."
        )
        return report

    report.dataset_found = True
    report.total_samples = len(samples)
    report.per_class_counts = {c: 0 for c in classes}
    report.per_person_counts = {}
    report.per_session_counts = {s: 0 for s in SESSIONS}
    report.handedness = {label: 0 for label in HANDEDNESS_GUIDE_LABELS}
    report.dynamic_stats = {}

    for sample in samples:
        meta = load_metadata(sample)
        sid = meta.get("sample_id", sample.name)
        cls = meta.get("class", "")
        person = meta.get("person_id", "?")
        session = meta.get("session_id", "?")

        ok, issues = validate_sample(sample, classes)
        if not ok:
            report.invalid_samples.append(sid)
            report.problems.extend(f"{sid}: {issue}" for issue in issues)
            if any("features" in i or "dimensión" in i or "NaN" in i or "rango" in i
                   for i in issues):
                report.invalid_vectors.append(sid)
            if apply_invalid:
                mark_invalid(sample, issues)

        if cls in report.per_class_counts:
            report.per_class_counts[cls] += 1
        report.per_person_counts[person] = report.per_person_counts.get(person, 0) + 1
        report.per_session_counts[session] = report.per_session_counts.get(session, 0) + 1
        key = f"{cls}|{person}"
        report.per_class_person[key] = report.per_class_person.get(key, 0) + 1

        guide = load_guide(sample)
        guide_hands = guide.get("hands", [])
        hand_key = (len(guide_hands) >= 2 and "both") or (
            "left" if guide_hands == ["Left"] else (
                "right" if guide_hands == ["Right"] else "none"))
        report.handedness[hand_key] += 1

        if meta.get("sign_type") == "dynamic":
            stats = report.dynamic_stats.setdefault(
                cls, {"samples": 0, "frames_sum": 0, "duration_ms_sum": 0}
            )
            stats["samples"] += 1
            stats["frames_sum"] += int(meta.get("frame_count", 0))
            stats["duration_ms_sum"] += int(meta.get("duration_ms", 0))

    report.classes = list(report.per_class_counts)
    report.persons = sorted(report.per_person_counts)
    report.sessions = sorted(report.per_session_counts)

    for cls, stats in report.dynamic_stats.items():
        stats["avg_frames"] = round(stats["frames_sum"] / max(stats["samples"], 1), 1)
        stats["avg_duration_ms"] = round(
            stats["duration_ms_sum"] / max(stats["samples"], 1), 1
        )

    for sample in iter_discarded(raw_dir):
        report.discarded_samples.append(sample.name)

    counts = list(report.per_class_counts.values())
    if counts:
        report.unbalanced = (max(counts) > 1.5 * min(counts)) if min(counts) > 0 else True

    if report.invalid_samples:
        report.warnings.append(
            f"{len(report.invalid_samples)} muestra(s) marcadas INVALID "
            "(detail en 'problems'; no se borran)."
        )
    if report.missing_files:
        report.warnings.append(
            f"{len(report.missing_files)} directorio(s) de muestra sin "
            "metadata.json (huérfanos o recorte incompleto)."
        )
    if report.discarded_samples:
        report.warnings.append(
            f"{len(report.discarded_samples)} muestra(s) descartada(s) por el "
            f"operador (ver raw/{'_discarded'}/)."
        )
    if len(report.persons) < 2:
        report.warnings.append(
            "LIMITACIÓN: una sola persona en el dataset. No se puede separar "
            "por persona; el split sería aleatorio (documentado). "
            "Para generalización real, recolecta con 2+ personas (piloto: 2)."
        )
    if report.unbalanced:
        report.warnings.append(
            "Desbalance entre clases detectado (proporción > 1.5:1). "
            "Completa las clases con menos muestras."
        )
    return report


def print_report(report: DatasetReport) -> None:
    print("\n=== VALIDACIÓN DEL DATASET SIGNIA (F14) ===\n")
    if not report.dataset_found:
        print("[VACÍO] Dataset vacío: primero debes recolectar muestras.")
        print("  -> python ai/collection/collect_data.py --person P01 --session A ")
        print("  -> python ai/collection/collect_data.py --person P02 --session A")
        return
    print(f"Muestras totales : {report.total_samples}")
    print(f"Personas         : {report.persons}")
    print(f"Sesiones         : {report.sessions}")
    print("\nMuestras por clase:")
    for cls, n in report.per_class_counts.items():
        print(f"  {cls:<12} {n}")
    print("\nMuestras por persona:")
    for person, n in report.per_person_counts.items():
        print(f"  {person:<12} {n}")
    print("\nMuestras por sesión:")
    for session, n in report.per_session_counts.items():
        print(f"  {session:<12} {n}")
    print("\nMano en el fotograma guía:")
    for label, n in report.handedness.items():
        print(f"  {label:<6} {n}")
    if report.dynamic_stats:
        print("\nDinámicas (secuencias):")
        for cls, stats in report.dynamic_stats.items():
            print(f"  {cls:<12} {stats['samples']} muestras | "
                  f"prom. {stats['avg_frames']} frames | "
                  f"prom. {stats['avg_duration_ms']} ms")
    if report.invalid_samples:
        print(f"\n[!] {len(report.invalid_samples)} muestra(s) INVALID:")
        for issue in report.problems[:25]:
            print(f"    - {issue}")
        if len(report.problems) > 25:
            print(f"    ... y {len(report.problems) - 25} más")
    for warning in report.warnings:
        print(f"\n[!] {warning}")