/**
 * Estabilidad temporal (F10) — TemporalSmoother.
 * La capa NO toca el modelo: trabaja sobre arrays de probabilidades.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { TemporalSmoother, SMOOTHING_CONFIG } from '../../frontend/js/temporalSmoother.js';

// probs fuertes por clase (muy por encima del umbral 0.70)
const P0 = [0.95, 0.05]; // clase 0 dominante
const P1 = [0.05, 0.95]; // clase 1 dominante
const WEAK = [0.55, 0.45]; // ninguna supera 0.70

function repeat(value, n) {
  return Array.from({ length: n }, () => value);
}

test('EMA: primer frame = entrada (sin historial)', () => {
  const s = new TemporalSmoother();
  const r = s.step(P0);
  assert.equal(r.emaProbabilities[0], 0.95);
  assert.equal(r.emaProbabilities[1], 0.05);
});

test('EMA: fórmula correcta alpha=0.35 (frames consecutivos)', () => {
  const s = new TemporalSmoother({ stableFrames: 1 });
  // frame2: 0.35*P0 + 0.65*frame1
  s.step(P0);
  const r2 = s.step(P0);
  const expected0 = 0.35 * 0.95 + 0.65 * 0.95;
  const expected1 = 0.35 * 0.05 + 0.65 * 0.05;
  assert.ok(Math.abs(r2.emaProbabilities[0] - expected0) < 1e-12);
  assert.ok(Math.abs(r2.emaProbabilities[1] - expected1) < 1e-12);
});

test('EMA: converge al valor sostenido', () => {
  const s = new TemporalSmoother({ stableFrames: 1 });
  let r;
  for (let i = 0; i < 50; i++) r = s.step(P0);
  assert.ok(Math.abs(r.emaProbabilities[0] - 0.95) < 1e-6);
});

test('primeras N frames: ninguna clase confirmada (stabilizing)', () => {
  const s = new TemporalSmoother(); // stableFrames=5
  for (let i = 0; i < 4; i++) {
    const r = s.step(P0);
    assert.equal(r.reason, 'stabilizing');
    assert.equal(r.label, null);
    assert.equal(r.recognized, false);
  }
});

test('1) un frame aislado NO cambia la clase mostrada', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);      // clase 0 confirmada
  const before = s.step(P0);
  assert.equal(before.label, 0);
  // un único frame de clase 1
  const noisy = s.step(P1);
  assert.equal(noisy.label, 0, 'sigue mostrando clase 0');
  assert.equal(noisy.reason, 'ok');
});

test('2) una nueva clase estable SÍ reemplaza a la anterior', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);      // HOLA confirmada
  for (let i = 0; i < 5; i++) s.step(P1);      // GRACIAS 5 frames
  const r = s.step(P1);
  assert.equal(r.label, 1, 'clase 1 reemplaza tras ser estable');
  assert.equal(r.reason, 'ok');
});

test('3) ausencia de manos limpia TODO el estado', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0); // clase 0 confirmada
  assert.equal(s.noHand(), null);
  // tras la mano ausente, la primera tanda de clase 1 NO se contamina con clase 0
  let r;
  for (let i = 0; i < 5; i++) r = s.step(P1);
  assert.equal(r.label, 1, 'arranca desde cero con la nueva seña');
  assert.equal(r.reason, 'ok');
});

test('4) predicción bajo el umbral NO genera una seña', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(WEAK); // nada supera 0.70
  const r = s.step(WEAK);
  assert.equal(r.label, null);
  assert.equal(r.reason, 'stabilizing');
  assert.equal(r.recognized, false);
});

test('4b) bajo el umbral tras confirmar: "Confianza insuficiente"', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);
  const dipped = s.step(WEAK);
  assert.notEqual(dipped.label, null, 'la clase confirmada no desaparece');
  assert.equal(dipped.reason, 'low_confidence');
  assert.equal(dipped.recognized, false);
});

test('5) el smoothing NO modifica las probabilidades RAW', () => {
  const s = new TemporalSmoother();
  const input = P0.slice();
  let r;
  for (let i = 0; i < 5; i++) r = s.step(input);
  assert.deepEqual(input, P0, 'el array de entrada no se muta');
  // y las RAW reportadas son las del modelo (argmax de la entrada)
  assert.equal(r.rawLabel, 0);
  assert.ok(Math.abs(r.rawConfidence - 0.95) < 1e-12);
});

test('cambio de seña: HOLA -> GRACIAS se abandona tras estabilidad', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);
  // menos de 5 frames de clase 1: se mantiene clase 0
  for (let i = 0; i < 4; i++) {
    const r = s.step(P1);
    assert.equal(r.label, 0, 'clase 0 aún mostrada durante la transición');
  }
  const r = s.step(P1);
  assert.equal(r.label, 1);
});

test('frame aislado de otra clase no rompe la secuencia estable', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);
  s.step(P1); // ruido 1 frame
  for (let i = 0; i < 5; i++) {
    const r = s.step(P0);
    assert.equal(r.label, 0, 'la clase 0 sigue siendo la mostrada');
  }
});

test('configuración: stableFrames y alpha personalizados', () => {
  const s = new TemporalSmoother({ stableFrames: 2, alpha: 0.5 });
  assert.equal(s.step(P0).reason, 'stabilizing');
  assert.equal(s.step(P0).reason, 'ok');        // 2 frames: confirmada
  assert.ok(s.step(P0).emaProbabilities[0] > 0.95 + 1e-9
    ? false : s.step(P0).emaProbabilities[0] > 0.9, 'EMA con alpha 0.5 responde más');
});

test('configuración: enabled=false -> passthrough RAW con umbral', () => {
  const s = new TemporalSmoother({ enabled: false });
  const r = s.step(WEAK);
  assert.equal(r.label, 0);                       // RAW argmax directo
  assert.equal(r.reason, 'low_confidence');       // pero bajo umbral
  assert.equal(r.recognized, false);
});

test('comportamiento determinista: misma secuencia -> mismos resultados', () => {
  const seq = [P0, P0, P0, P0, P1, P0, P0, P1, P1, P1, P1, P1, WEAK, WEAK];
  const run = () => {
    const s = new TemporalSmoother();
    const out = [];
    for (const p of seq) out.push(s.step(p));
    return out;
  };
  const a = run();
  const b = run();
  assert.deepEqual(a.map(x => [x.label, x.reason, x.consecutiveFrames]),
                   b.map(x => [x.label, x.reason, x.consecutiveFrames]));
});

test('reset() vuelve al estado inicial', () => {
  const s = new TemporalSmoother();
  for (let i = 0; i < 5; i++) s.step(P0);
  s.reset();
  const r = s.step(P0);
  assert.equal(r.label, null, 'nada confirmado tras reset');
});

test('configuración por defecto documentada en SMOOTHING_CONFIG', () => {
  assert.equal(SMOOTHING_CONFIG.enabled, true);
  assert.equal(SMOOTHING_CONFIG.alpha, 0.35);
  assert.equal(SMOOTHING_CONFIG.minConfidence, 0.70);
  assert.equal(SMOOTHING_CONFIG.stableFrames, 5);
});

test('requisito 15: 5 frames NO se afirman óptimos (valores editables)', () => {
  const s = new TemporalSmoother({ stableFrames: 3, minConfidence: 0.6, alpha: 0.9 });
  assert.equal(s.config.stableFrames, 3);
  assert.equal(s.config.minConfidence, 0.6);
  assert.equal(s.config.alpha, 0.9);
});