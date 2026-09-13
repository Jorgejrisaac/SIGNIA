/**
 * Guía temporal (Solución A, F15) — createGuideFrameSelector.
 * La ventana estabiliza el frame representativo; el modelo sigue recibiendo
 * [1,126]. Módulo puro: misma filosofía que select_guide_frame (samples.py).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  createGuideFrameSelector,
  GUIDE_FRAME_CONFIG,
} from '../../frontend/js/guideFrame.js';

const D = GUIDE_FRAME_CONFIG.featureSize; // 126

/** Vector válido de D componentes con valor constante v. */
function vec(v) {
  return Array.from({ length: D }, () => v);
}

function small(v, n = 4) {
  return Array.from({ length: n }, () => v);
}

test('config por defecto: windowSize 10-15, minFrames seguro', () => {
  assert.ok(GUIDE_FRAME_CONFIG.windowSize >= 10 && GUIDE_FRAME_CONFIG.windowSize <= 15);
  assert.ok(GUIDE_FRAME_CONFIG.minFrames >= 1 && GUIDE_FRAME_CONFIG.minFrames < GUIDE_FRAME_CONFIG.windowSize);
  assert.equal(GUIDE_FRAME_CONFIG.featureSize, 126);
});

test('ventana vacía: push de frame inválido devuelve null y no contamina', () => {
  const g = createGuideFrameSelector();
  assert.equal(g.isReady, false);
  assert.equal(g.size, 0);
  assert.equal(g.push(null), null);
  assert.equal(g.push(undefined), null);
  assert.equal(g.push([1, 2, 3]), null);          // longitud != 126
  assert.equal(g.push(vec(1).map((x) => String(x))), null); // no numéricos
  assert.equal(g.push(vec(NaN)), null);           // no finitos
  assert.equal(g.size, 0);
});

test('menos frames de los necesarios: comportamiento seguro (null)', () => {
  const g = createGuideFrameSelector({ minFrames: 3 });
  assert.equal(g.push(vec(0)), null);
  assert.equal(g.push(vec(1)), null);
  assert.equal(g.isReady, false);
  assert.equal(g.size, 2);
  const sel = g.push(vec(2)); // tercer frame válido: ya se puede seleccionar
  assert.ok(sel !== null);
  assert.equal(g.isReady, true);
});

test('selección: frame más cercano (L1) a la mediana por componente', () => {
  // featureSize=4: [1,2,2] -> mediana 2 -> v2 (dist 0). El outlier nunca gana.
  const g = createGuideFrameSelector({ featureSize: 4, minFrames: 3, windowSize: 5 });
  const v1 = small(1);
  const v2 = small(2);
  const v100 = small(100);
  assert.equal(g.push(v1), null);          // 1 frame: aún no se predice
  assert.equal(g.push(v2), null);          // 2 frames: aún no se predice
  const sel = g.push(v2);                  // [1,2,2] -> mediana 2 -> v2
  assert.deepEqual(sel, v2);
  const sel2 = g.push(v100);               // [1,2,2,100] -> mediana 2 -> v2
  assert.deepEqual(sel2, v2);
  assert.equal(g.size, 4);
});

test('selección con conteo impar: mediana = valor central', () => {
  const g = createGuideFrameSelector({ featureSize: 4, minFrames: 3, windowSize: 5 });
  g.push(small(1));
  g.push(small(10));
  const sel = g.push(small(5)); // mediana = 5 -> el propio 5 gana
  assert.deepEqual(sel, small(5));
});

test('la salida es exactamente el vector [1,126] seleccionado', () => {
  const g = createGuideFrameSelector();
  // target con valores > 3 en TODAS las componentes: con ventana [3,t,t]
  // la mediana por componente es t -> el propio target gana (dist 0).
  const target = Array.from({ length: D }, (_, i) => (i % 5) + 4);
  g.push(vec(3));
  g.push(target);
  const sel = g.push(target);
  assert.ok(Array.isArray(sel));
  assert.equal(sel.length, 126);
  assert.deepEqual(sel, target);
});

test('manos perdidas / frames no válidos no contaminan la ventana', () => {
  const g = createGuideFrameSelector({ featureSize: 4, minFrames: 2, windowSize: 4 });
  const good = small(7);
  g.push(good);
  g.push(small(7));
  // frames inválidos entre válidos: ignorados
  assert.equal(g.push(null), null);
  assert.equal(g.push(small(7).slice(0, 2)), null); // longitud incorrecta
  assert.equal(g.size, 2);
  // reset tras mano perdida: ventana limpia y arranque seguro de nuevo
  g.reset();
  assert.equal(g.size, 0);
  assert.equal(g.isReady, false);
  assert.equal(g.push(small(7)), null);
});

test('FIFO: la ventana mantiene a lo sumo windowSize frames', () => {
  const g = createGuideFrameSelector({ featureSize: 4, minFrames: 1, windowSize: 4 });
  g.push(small(1));
  g.push(small(2));
  g.push(small(3));
  g.push(small(4));
  assert.equal(g.size, 4);
  // 1er push(5) -> [2,3,4,5]: mediana 3.5 -> empate 3/4 -> primera ocurrencia -> 3
  assert.deepEqual(g.push(small(5)), small(3));
  // 2º push(5) -> [3,4,5,5]: mediana 4.5 -> empate 4/5/5 -> primera ocurrencia -> 4
  assert.deepEqual(g.push(small(5)), small(4));
  assert.equal(g.size, 4);
});

test('determinismo: misma secuencia -> misma selección', () => {
  const build = () => {
    const g = createGuideFrameSelector({ featureSize: 4, minFrames: 2, windowSize: 5 });
    const seq = [small(1), small(9), small(5), small(7)];
    const sel = seq.map((v) => g.push(v)).pop();
    return sel;
  };
  assert.deepEqual(build(), build());
});

test('REGRESIÓN F15: acepta Float32Array (tipo real de handsToFeatures)', () => {
  // handsToFeatures() devuelve Float32Array (features.js:60); el selector
  // NO debe rechazarlo (rompía todo el flujo de predicción).
  const g = createGuideFrameSelector();
  const typed = new Float32Array(D).fill(1.5);
  assert.equal(g.push(typed), null);           // 1er frame: calentamiento
  assert.equal(g.size, 1);                     // SÍ entró en la ventana
  g.push(typed);
  const sel = g.push(typed);
  assert.ok(sel !== null);
  assert.ok(sel instanceof Float32Array);
  assert.equal(sel.length, 126);
  assert.deepEqual(Array.from(sel), Array.from(typed));
});

test('REGRESIÓN F15: flujo real handsToFeatures -> guideFrame -> [1,126]', async () => {
  // Réplica del flujo de app.js:356-361 (sin DOM, solo los módulos puros).
  const { createRequire } = await import('node:module');
  const { fileURLToPath } = await import('node:url');
  const path = await import('node:path');
  const require = createRequire(import.meta.url);
  const feats = require(path.join(path.dirname(fileURLToPath(import.meta.url)),
    '..', '..', 'frontend', 'js', 'features.js'));

  const hand = Array.from({ length: 21 }, (_, i) => [
    0.4 + 0.03 * (i % 5), 0.5 - 0.02 * i, 0.01 * (i % 3),
  ]);
  const features = feats.handsToFeatures(null, hand);
  assert.ok(features instanceof Float32Array);
  assert.equal(features.length, 126);

  const g = createGuideFrameSelector();
  for (let i = 0; i < 2; i++) g.push(features);      // calentamiento
  const selected = g.push(features);                 // 3er frame
  assert.ok(selected !== null);
  assert.equal(selected.length, 126);                // contrato [1,126]
  assert.ok(Array.from(selected).every(Number.isFinite));
});