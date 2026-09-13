"""CLI de validación del dataset SIGNIA.

Uso:
    python ai/preprocessing/validate_dataset.py

Sale con código 0 si el dataset es utilizable, 1 si está vacío o tiene
anomalías (nunca falla silenciosamente).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.preprocessing.validate import print_report, validate_dataset  # noqa: E402


def main() -> int:
    report = validate_dataset()
    print_report(report)
    if not report.dataset_found or report.missing_files or report.invalid_vectors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())