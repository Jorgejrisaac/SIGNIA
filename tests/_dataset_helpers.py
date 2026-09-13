"""Factories de dataset SINTÉTICO en el esquema F14 (muestras por carpeta).

Solo para pruebas (requisito 27): viven en carpetas temporales de pytest,
NUNCA en ai/dataset/ ni se presentan como datos reales.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ai.training.config import EXPECTED_HANDS, SIGN_TYPES


def make_sample(raw_dir: Path, person: str = "P01", session: str = "A",
                cls: str = "HOLA", n: int = 1, guide_index: int = 0,
                frames: list[dict] | None = None,
                meta_overrides: dict | None = None,
                with_metadata: bool = True) -> tuple[Path, dict]:
    """Crea un directorio de muestra válido (o corrompido vía overrides).

    Por defecto genera fotogramas con el número de manos esperado de la
    clase y metadata coherente; `meta_overrides` permite romper campos a
    propósito (clase, persona, frame_count, quality, ...).
    """
    sample_id = f"{person}_{session}_{cls}_{n:04d}"
    sign_type = SIGN_TYPES[cls]
    expected = EXPECTED_HANDS[cls]

    if frames is None:
        rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
        count = 4 if sign_type == "static" else 10
        frames = []
        for i in range(1, count + 1):
            vec = np.zeros(126, dtype=np.float32)
            vec[63:126] = rng.uniform(-1, 1, 63)
            hands = ["Right"]
            if expected == 2:
                vec[0:63] = rng.uniform(-1, 1, 63)
                hands = ["Left", "Right"]
            frames.append({
                "features": [float(v) for v in vec],
                "hands": hands,
                "hand_count": len(hands),
                "t_ms": i * 33,
            })

    guide = frames[guide_index]
    out = raw_dir / person / session / cls / sample_id
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames, start=1):
        (out / "frames" / f"frame_{i:04d}.json").write_text(
            json.dumps(fr), encoding="utf-8"
        )
    (out / "guide.json").write_text(
        json.dumps({**guide, "frame_index": guide_index + 1,
                    "note": "fixture sintético"}), encoding="utf-8"
    )

    meta = {
        "sample_id": sample_id,
        "person_id": person,
        "session_id": session,
        "class": cls,
        "sign_type": sign_type,
        "timestamp": "2026-01-01T10:00:00.000000",
        "source": "sintetico-prueba",
        "hand_count": expected,
        "hand_count_max": expected,
        "frame_count": len(frames),
        "duration_ms": (len(frames) - 1) * 33,
        "guide_frame": guide_index + 1,
        "quality": "valid",
        "issues": [],
    }
    if sign_type == "dynamic":
        meta["sequence_id"] = sample_id
    meta.update(meta_overrides or {})
    if with_metadata:
        (out / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return out, meta
