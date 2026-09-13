/**
 * Paridad de la guía temporal (Solución A, F15) — Python <-> JavaScript.
 *
 * Uso: node guideParity.mjs --cases <cases.json>
 *
 * cases.json (generado por tests/test_guide_parity.py con numpy):
 *   {
 *     "feature_size": 126,
 *     "min_frames": 3,
 *     "windows": [ {"frames": [[...126...], ...], "expected_index": n,
 *                   "expected_vector": [...126...]}, ... ]
 *   }
 *
 * La referencia Python usa exactamente la misma heurística que
 * select_guide_frame (samples.py): mediana por componente (np.median) y
 * distancia L1 con desempate a la primera ocurrencia (np.argmin).
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import { createGuideFrameSelector } from '../../frontend/js/guideFrame.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : dflt;
}

const casesPath = path.resolve(arg('cases', ''));
if (!casesPath) {
  console.error('Uso: node guideParity.mjs --cases <cases.json>');
  process.exit(2);
}

let cases;
try {
  cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
} catch (e) {
  console.error('FAIL — no se pudo leer cases: ' + e.message);
  process.exit(1);
}

let failCount = 0;
const check = (cond, msg) => {
  if (cond) { console.log('  PASS — ' + msg); } else { console.error('  FAIL — ' + msg); failCount += 1; }
};

const featureSize = cases.feature_size;
const minFrames = cases.min_frames;

for (let w = 0; w < cases.windows.length; w++) {
  const entry = cases.windows[w];
  const g = createGuideFrameSelector({ featureSize, minFrames });
  let selected = null;
  for (const frame of entry.frames) {
    selected = g.push(frame);
  }
  const name = `ventana ${w} (${entry.frames.length} frames)`;
  check(selected !== null, `${name}: selección no nula`);
  if (selected === null) continue;

  const jsIndex = entry.frames.indexOf(selected);
  check(jsIndex === entry.expected_index,
    `${name}: índice JS=${jsIndex} == PY=${entry.expected_index}`);
  check(selected.length === featureSize, `${name}: salida [1,${featureSize}]`);

  let identical = selected.length === entry.expected_vector.length;
  if (identical) {
    for (let i = 0; i < selected.length; i++) {
      if (selected[i] !== entry.expected_vector[i]) { identical = false; break; }
    }
  }
  check(identical, `${name}: vector seleccionado idéntico al de Python`);
}

// Comportamiento de arranque: ventana por debajo de min_frames -> null
{
  const g = createGuideFrameSelector({ featureSize, minFrames });
  const warmup = cases.windows[0].frames.slice(0, minFrames - 1);
  for (const frame of warmup) g.push(frame);
  check(g.size === warmup.length && !g.isReady,
    `arranque: con ${warmup.length} frames no se predice (null)`);
}

console.log(failCount === 0
  ? 'PASS — Python ↔ JS guide-frame parity'
  : `FAIL — ${failCount} checks fallidos`);
process.exit(failCount === 0 ? 0 : 1);