"""Reporte del estado del dataset real (especificación F14 §23).

Uso:
    python ai/preprocessing/report_dataset.py [--raw-dir ...]

Escribe:
    - ai/dataset/reports/dataset_report.json   (estadísticas completas)
    - docs/piloto_dataset_report.md            (informe humano, solo si hay
                                               datos; se regenera en cada
                                               ejecución)

Contenido (F14 §23): total de muestras, personas, clases, sesiones, balance,
válidas/inválidas/descartadas, porcentajes por clase y persona, hand_count,
izquierda/derecha, dinámicas (frames y duración) y problemas detectados.

Con el dataset vacío NO falla: imprime el estado claro y no escribe el
informe (no se fabrican estadísticas inventadas).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.validate import DatasetReport, validate_dataset  # noqa: E402
from ai.training.config import (  # noqa: E402
    CLASES,
    RAW_DIR,
    REPORTS_DATASET_DIR,
    SIGN_TYPES,
)


def report_to_dict(report: DatasetReport) -> dict:
    """Serializa el reporte con porcentajes y estructura F14 §23."""
    base = report.as_dict()
    total = report.total_samples or 0

    per_class = report.per_class_counts
    per_person = report.per_person_counts

    def pct(value: float) -> float:
        return round(100.0 * value / total, 2) if total else 0.0

    per_class_pct = {cls: pct(n) for cls, n in per_class.items()}
    per_person_pct = {p: pct(n) for p, n in per_person.items()}

    sign_types = dict.fromkeys(CLASES, 0)
    for cls, n in per_class.items():
        if cls in SIGN_TYPES:
            sign_types[cls] = n

    return {
        **base,
        "per_class_percent": per_class_pct,
        "per_person_percent": per_person_pct,
        "static_samples": sum(n for cls, n in sign_types.items()
                              if SIGN_TYPES[cls] == "static"),
        "dynamic_samples": sum(n for cls, n in sign_types.items()
                               if SIGN_TYPES[cls] == "dynamic"),
        "valid_samples": total - len(report.invalid_samples),
        "invalid_samples": len(report.invalid_samples),
        "discarded_samples": len(report.discarded_samples),
        "balanced": not report.unbalanced,
    }


def write_markdown_report(report: DatasetReport, out_path: Path) -> None:
    """Genera docs/piloto_dataset_report.md a partir de datos REALES."""
    data = report_to_dict(report)
    total = report.total_samples or 0
    lines = [
        "# INFORME DEL PILOTO / ESTADO DEL DATASET REAL (F14)",
        "",
        f"> Generado automáticamente por `ai/preprocessing/report_dataset.py` "
        f"sobre los datos REALES presentes en `ai/dataset/raw/`. "
        f"Este documento NO fabrica datos: si el dataset está vacío no existe.",
        "",
        f"- **Muestras totales:** {total}",
        f"- **Personas:** {', '.join(report.persons) or '—'} "
        f"({len(report.persons)})",
        f"- **Sesiones:** {', '.join(report.sessions) or '—'}",
        f"- **Válidas:** {data['valid_samples']} | "
        f"**Inválidas (INVALID):** {data['invalid_samples']} | "
        f"**Descartadas por operador:** {data['discarded_samples']}",
        f"- **Balance entre clases:** {'OK' if data['balanced'] else 'DESBALANCE (>1.5:1)'}",
        "",
        "## Muestras por clase",
        "",
        "| Clase | N | % |",
        "|-------|---:|---:|",
    ]
    for cls in CLASES:
        n = report.per_class_counts.get(cls, 0)
        lines.append(f"| {cls} | {n} | {data['per_class_percent'].get(cls, 0.0)}% |")

    lines += ["", "## Muestras por persona", "",
              "| Persona | N | % |", "|---------|---:|---:|"]
    for person in report.persons:
        n = report.per_person_counts.get(person, 0)
        lines.append(f"| {person} | {n} | {data['per_person_percent'].get(person, 0.0)}% |")

    lines += ["", "## Muestras por sesión", "",
              "| Sesión | N |", "|--------|---:|"]
    for session, n in report.per_session_counts.items():
        lines.append(f"| {session} | {n} |")

    lines += ["", "## Mano en el fotograma guía", "",
              "| Mano | N |", "|------|---:|"]
    for label, n in report.handedness.items():
        lines.append(f"| {label} | {n} |")

    if report.dynamic_stats:
        lines += ["", "## Dinámicas (secuencias conservadas)", "",
                  "| Clase | Muestras | Frames prom. | Duración prom. (ms) |",
                  "|-------|---------:|-------------:|--------------------:|"]
        for cls, stats in report.dynamic_stats.items():
            lines.append(
                f"| {cls} | {stats['samples']} | {stats['avg_frames']} | "
                f"{stats['avg_duration_ms']} |")

    if report.invalid_samples:
        lines += ["", "## Muestras inválidas", ""]
        lines.append("| sample | problema |")
        lines.append("|--------|----------|")
        for problem in report.problems:
            parts = problem.split(": ", 1)
            lines.append(f"| {parts[0]} | {parts[1] if len(parts) > 1 else problem} |")
    if report.discarded_samples:
        lines += ["", "## Muestras descartadas (operador)", ""]
        for sid in report.discarded_samples:
            lines.append(f"- `{sid}`")
    if report.problems:
        lines += ["", "## Problemas detectados", ""]
        for problem in report.problems:
            lines.append(f"- {problem}")
    else:
        lines += ["", "## Problemas detectados", "", "- Ninguno."]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Informe humano escrito en {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reporte del dataset real (F14)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    report = validate_dataset(raw_dir=args.raw_dir)
    if not report.dataset_found:
        print("[VACÍO] No hay datos reales todavía: el reporte NO se genera. "
              "Primero recolecta el piloto "
              "(collect_data.py --person P01/--person P02).")
        print("No se escriben estadísticas inventadas.")
        return 1

    out_json = REPORTS_DATASET_DIR / "dataset_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Estado JSON escrito en {out_json}")

    write_markdown_report(report, PROJECT_ROOT / "docs" / "piloto_dataset_report.md")

    print("\n=== RESUMEN ===")
    print(f"Muestras: {report.total_samples} | Personas: {len(report.persons)}")
    print(f"Inválidas: {len(report.invalid_samples)} | "
          f"Descartadas: {len(report.discarded_samples)}")
    if report.unbalanced:
        print("[!] Desbalance entre clases > 1.5:1")
    for problem in report.problems[:10]:
        print(f"[!] {problem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())