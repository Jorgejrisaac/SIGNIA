"""Paridad Python <-> JavaScript de la guía temporal (Solución A, F15).

La referencia Python replica EXACTAMENTE la heurística de select_guide_frame
(ai/preprocessing/samples.py): mediana por componente (np.median) y distancia
L1 con desempate a la primera ocurrencia (np.argmin), aplicada a una ventana
de frames válidos.

El test genera ventanas SINTÉTICAS (seed fija, nada de ai/dataset/), calcula
el vector esperado en Python y verifica que el módulo JS
(frontend/js/guideFrame.js) selecciona el MISMO frame y el MISMO vector.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARITY_JS_DIR = PROJECT_ROOT / "tests" / "js_parity"
NODE = shutil.which("node")

FEATURE_SIZE = 126
MIN_FRAMES = 3
WINDOW_SIZE = 12


def _median_l1_selection(frames: list[list[float]]) -> int:
    """Réplica fiel de select_guide_frame sobre una ventana (sin filtro de
    hand_count: la ventana solo contiene frames válidos con mano)."""
    stack = np.vstack([np.asarray(f, dtype=np.float64) for f in frames])
    median = np.median(stack, axis=0)
    dists = np.abs(stack - median).sum(axis=1)
    return int(np.argmin(dists))


def _windows(n: int = 12) -> list[dict]:
    rng = np.random.default_rng(20260815)
    out = []
    for i in range(n):
        n_frames = int(rng.integers(MIN_FRAMES + 2, WINDOW_SIZE + 2))
        frames = rng.normal(0.0, 2.0, size=(n_frames, FEATURE_SIZE)).tolist()
        for frame in frames:
            for j in range(0, FEATURE_SIZE, 21):  # mano ausente -> ceros
                if rng.random() < 0.2:
                    end = min(j + 21 * 3, FEATURE_SIZE)
                    frame[j:end] = [0.0] * (end - j)
        idx = _median_l1_selection(frames)
        out.append({"frames": frames, "expected_index": idx,
                    "expected_vector": frames[idx]})
    return out


@pytest.mark.skipif(NODE is None, reason="Node.js no está instalado")
def test_paridad_guia_temporal_python_js(tmp_path):
    cases_file = tmp_path / "guide_parity_cases.json"
    cases_file.write_text(json.dumps({
        "feature_size": FEATURE_SIZE,
        "min_frames": MIN_FRAMES,
        "windows": _windows(),
    }), encoding="utf-8")

    cmd = [NODE, str(PARITY_JS_DIR / "guideParity.mjs"),
           "--cases", str(cases_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", timeout=300, cwd=tmp_path)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, (
        f"paridad JS falló (exit {proc.returncode})\n{proc.stderr}")


def test_referencia_python_mediana_l1():
    """La propia referencia: casos conocidos (guardia del fixture)."""
    v1 = [1.0] * 4
    v2 = [2.0] * 4
    v3 = [3.0] * 4
    v100 = [100.0] * 4
    frames = [v1, v2, v3, v100]
    idx = _median_l1_selection(frames)  # mediana 2.5 -> empate 2/3 -> primera
    assert idx == 1
    assert _median_l1_selection([v1, v2, v3]) == 1  # mediana 2 -> v2
    assert len(_windows(2)) == 2
    assert all(len(w["frames"][0]) == FEATURE_SIZE for w in _windows(2))