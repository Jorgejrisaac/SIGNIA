/**
 * Integración F11: SOLO las SEÑAS CONFIRMADAS de F10 generan
 * historial + voz. RAW y SMOOTHED por sí solos no.
 *
 * Simula el pipeline completo sin DOM ni cámara:
 *   TemporalSmoother (F10) → flow (F11) → History + Speech stub.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { TemporalSmoother } from '../../frontend/js/temporalSmoother.js';
import { History } from '../../frontend/js/history.js';
import { createConfirmedFlow } from '../../frontend/js/flow.js';

const P0 = [0.95, 0.05];
const P1 = [0.05, 0.95];

function pipeline({ onConfirm = () => {} } = {}) {
  const smoother = new TemporalSmoother();
  const history = new History();
  const flow = createConfirmedFlow({ history, onConfirm });
  const confirmations = [];
  return {
    smoother,
    history,
    confirmations,
    frame(probs, handPresent = true) {
      const smoothed = handPresent
        ? smoother.step(probs)
        : smoother.noHand();
      const entry = flow.onFrame(smoothed);
      if (entry) confirmations.push(entry);
      if (!handPresent) flow.separate();
      return smoothed;
    },
  };
}

test('RAW por sí sola (frames sueltos) NO genera historial', () => {
  const p = pipeline();
  p.frame(P0); // 1 frame: stabilizing
  assert.equal(p.history.size, 0);
  assert.equal(p.confirmations.length, 0);
});

test('SMOOTHED sin confirmar NO genera historial (transición)', () => {
  const p = pipeline();
  for (let i = 0; i < 5; i++) p.frame(P0); // clase 0 confirmada
  p.frame(P1);                              // transición (aún clase 0)
  assert.equal(p.confirmations.length, 1, 'solo la confirmación inicial');
  assert.equal(p.history.size, 1);
});

test('solo la CONFIRMACIÓN genera historial + voz (una vez)', () => {
  const p = pipeline();
  let speakCount = 0;
  const q = pipeline({ onConfirm: () => { speakCount += 1; } });
  q.frame(P0);
  q.frame(P0);
  q.frame(P0);
  q.frame(P0);
  q.frame(P0); // 5º: confirmación
  q.frame(P0); // frames siguientes: sin duplicado
  q.frame(P0);
  assert.equal(speakCount, 1, 'voz una sola vez por confirmación');
  assert.equal(q.history.size, 1);
  assert.equal(q.history.list()[0].label, 0);
});

test('misma seña durante 100 frames: un solo registro', () => {
  const p = pipeline();
  for (let i = 0; i < 100; i++) p.frame(P0);
  assert.equal(p.history.size, 1);
  assert.equal(p.confirmations.length, 1);
});

test('mano desaparece y vuelve: la misma seña se registra de nuevo', () => {
  const p = pipeline();
  for (let i = 0; i < 5; i++) p.frame(P0);
  assert.equal(p.history.size, 1);
  p.frame(P0, false); // mano ausente -> noHand + separate
  for (let i = 0; i < 5; i++) p.frame(P0);
  assert.equal(p.history.size, 2, 'HOLA, HOLA: repetición permitida');
});

test('cambio de seña HOLA -> GRACIAS registra ambas', () => {
  const p = pipeline();
  for (let i = 0; i < 5; i++) p.frame(P0);
  for (let i = 0; i < 5; i++) p.frame(P1);
  assert.deepEqual(p.history.list().map((e) => e.label), [0, 1]);
});

test('confianza baja no genera confirmación ni historial', () => {
  const p = pipeline();
  for (let i = 0; i < 10; i++) p.frame([0.55, 0.45]); // bajo 0.70
  assert.equal(p.confirmations.length, 0);
  assert.equal(p.history.size, 0);
});

test('historial guarda confianza EMA de la confirmación', () => {
  const p = pipeline();
  for (let i = 0; i < 5; i++) p.frame(P0);
  const entry = p.confirmations[0];
  assert.equal(entry.label, 0);
  assert.ok(entry.confidence >= 0.7 && entry.confidence <= 1.0);
  assert.equal(typeof entry.timestamp, 'number');
});

test('un frame aislado durante la seña no genera evento extra', () => {
  const p = pipeline();
  for (let i = 0; i < 5; i++) p.frame(P0);
  p.frame(P1); // ruido
  assert.equal(p.confirmations.length, 1);
  assert.equal(p.history.size, 1);
});

test('convierte el índice confirmado a etiqueta textual para historial y voz', () => {
  const history = new History();
  const flow = createConfirmedFlow({
    history,
    resolveLabel: (index) => ['HOLA', 'SI', 'NO'][index],
  });
  const entry = flow.onFrame({
    reason: 'ok',
    isNewConfirmation: true,
    label: 1,
    confidence: 0.8,
  });
  assert.equal(entry.label, 'SI');
  assert.equal(history.last().label, 'SI');
});
