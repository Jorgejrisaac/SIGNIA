#!/usr/bin/env node
/**
 * PARIDAD Python <-> JavaScript (Fase 8) — SIGNIA.
 *
 * Carga el modelo TF.js EXACTAMENTE como lo hará el navegador:
 *   leer model.json + concat shards → tf.loadLayersModel(fromMemory).
 *
 * Uso:
 *   node parity.mjs --model-dir <dir> --cases <cases.json> [--max-abs-tol 1e-3]
 *
 * cases.json (generado por tests/test_export.py con las mismas entradas
 * usadas en Python):
 *   {
 *     "expected_classes": [...],
 *     "normalization_cases": [{"name", "left", "right", "expected_features": [126]}],
 *     "cases": [{"name", "features": [126], "expected": [probs de Python]}]
 *   }
 *
 * Salida:
 *   PASS — Python ↔ TF.js parity        (exit 0)
 *   FAIL — maximum probability difference: X  (exit 1)
 */

import * as tf from '@tensorflow/tfjs';
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const feats = require('../../frontend/js/features.js');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : dflt;
}

const modelDir = path.resolve(arg('model-dir', ''));
const casesPath = path.resolve(arg('cases', ''));
const maxAbsTol = parseFloat(arg('max-abs-tol', '1e-3'));
const maeTol = parseFloat(arg('mae-tol', '1e-4'));
const normTol = parseFloat(arg('norm-tol', '1e-5'));

if (!modelDir || !casesPath) {
  console.error('Uso: node parity.mjs --model-dir <dir> --cases <cases.json>');
  process.exit(2);
}

let failCount = 0;
const check = (cond, msg) => {
  if (cond) { console.log('  PASS — ' + msg); } else { console.error('  FAIL — ' + msg); failCount += 1; }
};

// --- 1. Carga del modelo TF.js (model.json + shards, igual que el navegador) ---
let model;
try {
  const modelJson = JSON.parse(fs.readFileSync(path.join(modelDir, 'model.json'), 'utf8'));
  const weightsManifest = modelJson.weightsManifest;
  const chunks = [];
  for (const group of weightsManifest) {
    for (const wPath of group.paths) {
      chunks.push(fs.readFileSync(path.join(modelDir, wPath)));
    }
  }
  const weightData = Buffer.concat(chunks.map(c => new Uint8Array(c))).buffer;
  model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: modelJson.modelTopology,
    weightSpecs: weightsManifest.flatMap(g => g.weights),
    weightData,
  }));
  console.log('Modelo TF.js cargado: ' + path.join(modelDir, 'model.json'));
} catch (e) {
  console.error('FAIL — no se pudo cargar el modelo TF.js: ' + e.message);
  process.exit(1);
}

// --- 2. Metadata y formas ---------------------------------------------------
let metadata;
try {
  metadata = JSON.parse(fs.readFileSync(path.join(modelDir, 'metadata.json'), 'utf8'));
} catch (e) {
  console.error('FAIL — no se pudo leer metadata.json: ' + e.message);
  process.exit(1);
}

const inShape = model.inputs[0].shape;
const outShape = model.output.shape;
check(inShape[1] === metadata.n_features,
  `input shape ${JSON.stringify(inShape)} == (null, ${metadata.n_features})`);
check(outShape[1] === metadata.n_classes,
  `output shape ${JSON.stringify(outShape)} == (null, ${metadata.n_classes})`);
check(Array.isArray(metadata.classes) && metadata.classes.length === metadata.n_classes,
  `clases en metadata: ${metadata.n_classes}`);

// --- 3. Orden de clases: una sola fuente (metadata.json) --------------------
const cases = JSON.parse(fs.readFileSync(casesPath, 'utf8'));
check(JSON.stringify(metadata.classes) === JSON.stringify(cases.expected_classes),
  `orden de clases compartido: ${metadata.classes.join(', ')}`);

// --- 4. Paridad de normalización (features.js vs features.py) --------------
for (const nc of cases.normalization_cases) {
  const jsFeatures = Array.from(feats.handsToFeatures(nc.left ?? null, nc.right ?? null));
  const expected = nc.expected_features;
  let maxDiff = 0; let mae = 0;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(jsFeatures[i] - expected[i]);
    maxDiff = Math.max(maxDiff, d); mae += d;
  }
  mae /= expected.length;
  check(maxDiff <= normTol,
    `normalización "${nc.name}": maxDiff=${maxDiff.toExponential(3)} <= ${normTol}`);
}

// --- 5. Paridad de predicción (probabilidades completas) -------------------
let globalMaxDiff = 0; let globalMae = 0;
for (const c of cases.cases) {
  const x = tf.tensor2d([c.features], [1, metadata.n_features]);
  const probs = Array.from(await model.predict(x).data());
  x.dispose();

  let maxDiff = 0; let mae = 0;
  for (let i = 0; i < probs.length; i++) {
    const d = Math.abs(probs[i] - c.expected[i]);
    maxDiff = Math.max(maxDiff, d); mae += d;
  }
  mae /= probs.length;
  globalMaxDiff = Math.max(globalMaxDiff, maxDiff);
  globalMae += mae;

  const argmaxJs = probs.indexOf(Math.max(...probs));
  const argmaxPy = c.expected.indexOf(Math.max(...c.expected));
  check(argmaxJs === argmaxPy,
    `argmax "${c.name}": JS=${argmaxJs} (${metadata.classes[argmaxJs]}) == PY=${argmaxPy} (${metadata.classes[argmaxPy]})`);
  check(maxDiff <= maxAbsTol,
    `probabilidades "${c.name}": maxAbsDiff=${maxDiff.toExponential(3)} <= ${maxAbsTol}`);
}
globalMae /= cases.cases.length;
check(globalMae <= maeTol, `MAE global: ${globalMae.toExponential(3)} <= ${maeTol}`);

// --- 6. Resultado -----------------------------------------------------------
console.log('\nJS — maxAbsDiff global = ' + globalMaxDiff.toExponential(4) +
            ' | MAE global = ' + globalMae.toExponential(4));
if (failCount === 0) {
  console.log('PASS — Python ↔ TF.js parity');
  process.exit(0);
}
console.log(`FAIL — maximum probability difference: ${globalMaxDiff.toExponential(4)} (${failCount} checks)`);
process.exit(1);