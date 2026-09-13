"""Esquema de muestras del dataset SIGNIA (F14).

UNA MUESTRA = UNA EJECUCIÓN. Para las señas dinámicas, la muestra conserva
la SECUENCIA COMPLETA de fotogramas; para las estáticas, la retención breve
de la pose (docs/protocolo_dataset.md).

Estructura (ai/dataset/raw/):

    P01/
        A/
            HOLA/
                P01_A_HOLA_0001/
                    metadata.json      # trazabilidad completa (espec F14 §11)
                    guide.json         # fotograma guía: referencia del MVP
                    frames/
                        frame_0001.json    # {"features":[126], "hands":[...],
                        ...                #  "hand_count":n, "t_ms":ms}
                P01_A_HOLA_0002/
        B/
    _discarded/                         # tecla D del recolector: se MUEVEN aquí

Reglas (F14):
- Los IDs nunca se sobrescriben: el siguiente es el contador máximo + 1.
- La validación marca quality="invalid" + issues[] (NUNCA borra en silencio).
- El fotograma guía es una REFERENCIA del MVP estático; no representa la
  seña dinámica completa.
- 1 muestra = 1 fila de entrenamiento: nunca se mezclan frames sueltos como
  muestras independientes (evita data leakage persona/ejecución).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ai.training.config import (
    CLASES,
    DISCARDED_DIRNAME,
    EXPECTED_HANDS,
    SESSIONS,
    SIGN_TYPES,
)

# Límite de cordura para features normalizadas (mano normalizada: muñeca=0,
# palma~1; valores mayores a este límite indican landmarks corruptos).
FEATURE_RANGE_LIMIT = 100.0

FRAME_NAME_RE = re.compile(r"^frame_(\d{4})\.json$")
SAMPLE_ID_RE = re.compile(r"^(P\d+)_([ABC])_([A-Z0-9_]+)_(\d{4})$")

REQUIRED_METADATA_FIELDS = (
    "sample_id",
    "person_id",
    "session_id",
    "class",
    "sign_type",
    "timestamp",
    "source",
    "hand_count",
    "hand_count_max",
    "frame_count",
    "duration_ms",
    "guide_frame",
    "quality",
)


# ---------------------------------------------------------------------------
# Rutas e IDs
# ---------------------------------------------------------------------------
def sample_dir(raw_dir: Path, person: str, session: str, cls: str,
               sample_id: str) -> Path:
    """Ruta del directorio de una muestra (raw/P/S/CLASE/SAMPLE)."""
    return raw_dir / person / session / cls / sample_id


def next_sample_id(raw_dir: Path, person: str, session: str, cls: str) -> str:
    """ID único sin sobrescribir: máximo contador existente + 1.

    El ID codifica persona/sesión/clase: {PERSON}_{SESSION}_{CLASE}_{n:04d}.
    """
    pattern = re.compile(rf"^{re.escape(person)}_{re.escape(session)}_"
                         rf"{re.escape(cls)}_(\d{{4}})$")
    highest = 0
    base = raw_dir / person / session / cls
    if base.exists():
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            m = pattern.match(entry.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return f"{person}_{session}_{cls}_{highest + 1:04d}"


def count_samples_for_person(raw_dir: Path, person: str,
                             session: str | None = None) -> dict[str, int]:
    """Muestras válidas por clase para una persona.

    - session=None: todas las sesiones de la persona (compatibilidad/QC).
    - session="B": SOLO esa sesión (sesiones independientes: el recolector
      no debe ver las muestras de A cuando trabaja en B).
    """
    counts: dict[str, int] = {c: 0 for c in CLASES}
    person_dir = raw_dir / person
    if not person_dir.exists():
        return counts
    for ses in person_dir.iterdir():
        if not ses.is_dir():
            continue
        if session is not None and ses.name != session:
            continue
        for cls_dir in ses.iterdir():
            if not cls_dir.is_dir() or cls_dir.name not in counts:
                continue
            counts[cls_dir.name] += sum(
                1 for e in cls_dir.iterdir()
                if e.is_dir() and (e / "metadata.json").exists()
            )
    return counts


# ---------------------------------------------------------------------------
# Escritura de muestras (recolector)
# ---------------------------------------------------------------------------
def write_frame(out_dir: Path, index: int, features: np.ndarray,
                hands: list[str], t_ms: int) -> None:
    """Guarda un fotograma como JSON legible y trazable."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": [float(v) for v in np.asarray(features, dtype=np.float32).ravel()],
        "hands": list(hands),
        "hand_count": len(hands),
        "t_ms": int(t_ms),
    }
    (frames_dir / f"frame_{index:04d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def write_guide(out_dir: Path, frame_index: int, features: np.ndarray,
                hands: list[str], t_ms: int) -> None:
    payload = {
        "frame_index": int(frame_index),
        "features": [float(v) for v in np.asarray(features, dtype=np.float32).ravel()],
        "hands": list(hands),
        "hand_count": len(hands),
        "t_ms": int(t_ms),
        "note": ("Fotograma guía: referencia del MVP estático; "
                 "no representa la seña dinámica completa."),
    }
    (out_dir / "guide.json").write_text(json.dumps(payload), encoding="utf-8")


def write_metadata(out_dir: Path, metadata: dict) -> None:
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def select_guide_frame(frames: list[dict], expected_hands: int) -> dict | None:
    """Selección determinista del fotograma guía.

    Heurística documentada: entre los fotogramas con el número de manos
    esperado, el más cercano a la mediana de la configuración (robusto a
    outliers). Si ninguno cumple, el de mayor número de manos. Devuelve el
    dict del fotograma o None si no hay fotogramas.
    """
    if not frames:
        return None
    candidates = [f for f in frames if f["hand_count"] == expected_hands]
    if not candidates:
        best = max(frames, key=lambda f: f["hand_count"])
        return {"frame": best, "expected_ok": False}
    stack = np.vstack([np.asarray(f["features"], dtype=np.float64) for f in candidates])
    median = np.median(stack, axis=0)
    dists = np.abs(stack - median).sum(axis=1)
    best = candidates[int(np.argmin(dists))]
    return {"frame": best, "expected_ok": True}


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def iter_samples(raw_dir: Path) -> list[Path]:
    """Todos los directorios de muestra bajo raw/P/S/CLASE/SAMPLE (no _discarded)."""
    out: list[Path] = []
    if not raw_dir.exists():
        return out
    for person in sorted(raw_dir.iterdir()):
        if not person.is_dir() or person.name.startswith("_"):
            continue
        for session in sorted(person.iterdir()):
            if not session.is_dir():
                continue
            for cls_dir in sorted(session.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for sample in sorted(cls_dir.iterdir()):
                    if sample.is_dir() and (sample / "metadata.json").exists():
                        out.append(sample)
    return out


def iter_discarded(raw_dir: Path) -> list[Path]:
    out: list[Path] = []
    base = raw_dir / DISCARDED_DIRNAME
    if not base.exists():
        return out
    for sample in sorted(base.rglob("metadata.json")):
        out.append(sample.parent)
    return out


def load_metadata(sample: Path) -> dict:
    with open(sample / "metadata.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_guide(sample: Path) -> dict:
    with open(sample / "guide.json", encoding="utf-8") as fh:
        return json.load(fh)


def load_frames(sample: Path) -> list[dict]:
    frames: list[dict] = []
    frames_dir = sample / "frames"
    if not frames_dir.exists():
        return frames
    for f in sorted(frames_dir.iterdir()):
        if not FRAME_NAME_RE.match(f.name):
            continue
        with open(f, encoding="utf-8") as fh:
            frames.append(json.load(fh))
    return frames


# ---------------------------------------------------------------------------
# Validación de una muestra (espec F14 §10)
# ---------------------------------------------------------------------------
def validate_sample(sample: Path, classes: list[str] | None = None) -> tuple[bool, list[str]]:
    """Aplica los chequeos de control de calidad a UNA muestra.

    Nunca borra: devuelve (ok, issues) y el invocador puede persistir el
    estado con mark_invalid().
    """
    classes = classes or CLASES
    issues: list[str] = []

    meta_path = sample / "metadata.json"
    if not meta_path.exists():
        return False, ["falta metadata.json"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False, ["metadata.json corrupto"]

    # --- campos obligatorios
    for field in REQUIRED_METADATA_FIELDS:
        if field not in meta:
            issues.append(f"falta campo '{field}'")

    # --- clase válida
    cls = meta.get("class", "")
    if cls not in classes:
        issues.append(f"clase inválida '{cls}'")

    # --- persona válida
    person = meta.get("person_id", "")
    if not re.match(r"^P\d+$", person):
        issues.append(f"person_id inválido '{person}'")

    # --- sesión válida
    session = meta.get("session_id", "")
    if session not in SESSIONS:
        issues.append(f"session_id inválido '{session}'")

    # --- timestamp válido
    ts = meta.get("timestamp", "")
    try:
        datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        issues.append(f"timestamp inválido '{ts}'")

    # --- tipo de seña consistente
    sign_type = meta.get("sign_type", "")
    if cls in SIGN_TYPES and sign_type != SIGN_TYPES[cls]:
        issues.append(f"sign_type '{sign_type}' inconsistente con la clase")

    # --- sample_id coherente
    sid = meta.get("sample_id", "")
    m = SAMPLE_ID_RE.match(sid)
    if not m or m.group(1) != person or m.group(2) != session or m.group(3) != cls:
        issues.append(f"sample_id '{sid}' incoherente con persona/sesión/clase")

    # --- guide.json
    guide_path = sample / "guide.json"
    guide_feat = None
    guide_hand_count = None
    if not guide_path.exists():
        issues.append("falta guide.json")
    else:
        try:
            guide = json.loads(guide_path.read_text(encoding="utf-8"))
            guide_feat = guide.get("features")
            ok, err = _check_vector(guide_feat)
            if not ok:
                issues.append(f"guide features: {err}")
            guide_hand_count = guide.get("hand_count")
            if cls in EXPECTED_HANDS and guide_hand_count != EXPECTED_HANDS[cls]:
                issues.append(
                    f"hand_count del guía ({guide_hand_count}) "
                    f"incompatible con la clase ({EXPECTED_HANDS[cls]} esperadas)"
                )
        except Exception:
            issues.append("guide.json corrupto")
    if cls in EXPECTED_HANDS and meta.get("hand_count") != guide_hand_count:
        issues.append(
            f"hand_count del metadata ({meta.get('hand_count')}) != guía "
            f"({guide_hand_count})"
        )

    # --- frames
    frames = load_frames(sample)
    frame_count = meta.get("frame_count")
    if sign_type == "dynamic" and len(frames) < 2:
        issues.append("seña dinámica sin secuencia (menos de 2 fotogramas)")
    if sign_type == "static" and not frames:
        issues.append("seña estática sin fotogramas")
    if frame_count != len(frames):
        issues.append(f"frame_count {frame_count} != fotogramas reales {len(frames)}")
    for i, fr in enumerate(frames):
        ok, err = _check_vector(fr.get("features"))
        if not ok:
            issues.append(f"frame {i + 1}: {err}")
    if frames:
        names = sorted(sample.glob("frames/*.json"))
        expected = [f"frame_{i + 1:04d}.json" for i in range(len(frames))]
        if [n.name for n in names] != expected:
            issues.append("fotogramas sin numeración secuencial (huecos o archivos extra)")

    # --- consistencias temporales
    duration = meta.get("duration_ms")
    if not isinstance(duration, (int, float)) or duration < 0:
        issues.append("duration_ms inválido")
    gf = meta.get("guide_frame")
    if not isinstance(gf, int) or gf < 1 or gf > len(frames):
        issues.append(f"guide_frame {gf} fuera de rango (1..{len(frames)})")

    # --- dinámicas: sequence_id
    if sign_type == "dynamic" and meta.get("sequence_id") != sid:
        issues.append("sequence_id ausente o distinto de sample_id")

    return not issues, issues


def _check_vector(vec) -> tuple[bool, str]:
    """Dimensiones, NaN/Inf y rango razonable de un vector de features."""
    if vec is None:
        return False, "features ausentes"
    try:
        arr = np.asarray(vec, dtype=np.float64)
    except Exception:
        return False, "features ilegibles"
    if arr.ndim != 1 or arr.shape[0] != 126:
        return False, f"dimensión {arr.shape} (esperada 126)"
    if not np.isfinite(arr).all():
        return False, "valores NaN/Inf"
    if float(np.abs(arr).max()) > FEATURE_RANGE_LIMIT:
        return False, f"valor fuera de rango ({float(np.abs(arr).max()):.1f})"
    return True, ""


def mark_invalid(sample: Path, issues: list[str]) -> None:
    """Persiste quality='invalid' + issues en metadata.json (nunca borra)."""
    meta_path = sample / "metadata.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["quality"] = "invalid"
    meta["issues"] = list(issues)
    write_metadata(sample, meta)
