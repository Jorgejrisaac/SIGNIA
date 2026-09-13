/**
 * Historial de señas confirmadas (F11).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { History, HISTORY_CONFIG } from '../../frontend/js/history.js';

test('agregar una confirmación', () => {
  const h = new History();
  const entry = h.pushIfNew('HOLA', 0.91);
  assert.ok(entry);
  assert.equal(entry.label, 'HOLA');
  assert.equal(entry.confidence, 0.91);
  assert.equal(typeof entry.timestamp, 'number');
  assert.equal(h.size, 1);
});

test('evitar duplicados continuos', () => {
  const h = new History();
  assert.ok(h.pushIfNew('HOLA', 0.9));
  assert.equal(h.pushIfNew('HOLA', 0.92), null, 'misma seña mantenida: no duplicar');
  assert.equal(h.pushIfNew('HOLA', 0.88), null);
  assert.equal(h.size, 1);
});

test('permitir repetir la misma seña tras separate() (mano ausente)', () => {
  const h = new History();
  h.pushIfNew('HOLA', 0.9);
  h.separate();
  assert.ok(h.pushIfNew('HOLA', 0.9), 'tras separación se registra de nuevo');
  assert.equal(h.size, 2);
});

test('cambio de seña entra directamente (sin separate)', () => {
  const h = new History();
  h.pushIfNew('HOLA', 0.9);
  assert.ok(h.pushIfNew('GRACIAS', 0.87), 'seña distinta no es duplicado');
  assert.equal(h.size, 2);
});

test('límite máximo: se elimina la entrada más antigua', () => {
  const h = new History({ maxEntries: 3 });
  h.pushIfNew('A', 0.9);
  h.pushIfNew('B', 0.9);
  h.pushIfNew('C', 0.9);
  h.pushIfNew('D', 0.9);
  assert.equal(h.size, 3);
  assert.deepEqual(h.list().map((e) => e.label), ['B', 'C', 'D']);
});

test('limpiar', () => {
  const h = new History();
  h.pushIfNew('HOLA', 0.9);
  h.pushIfNew('GRACIAS', 0.9);
  h.clear();
  assert.equal(h.size, 0);
  assert.equal(h.last(), null);
  // tras limpiar, la misma seña vuelve a registrarse
  assert.ok(h.pushIfNew('HOLA', 0.9));
});

test('orden cronológico (inserción)', () => {
  const h = new History();
  const t0 = h.pushIfNew('HOLA', 0.9).timestamp;
  const t1 = h.pushIfNew('GRACIAS', 0.9).timestamp;
  const t2 = h.pushIfNew('AYUDA', 0.9).timestamp;
  assert.ok(t0 <= t1 && t1 <= t2);
  assert.deepEqual(h.list().map((e) => e.label), ['HOLA', 'GRACIAS', 'AYUDA']);
});

test('confianza y timestamp se conservan', () => {
  const h = new History();
  const entry = h.pushIfNew('SI', 0.833333);
  assert.ok(Math.abs(entry.confidence - 0.833333) < 1e-9);
  assert.equal(typeof entry.timestamp, 'number');
  assert.ok(entry.timestamp > 0);
});

test('last() devuelve la más reciente', () => {
  const h = new History();
  h.pushIfNew('HOLA', 0.9);
  h.pushIfNew('GRACIAS', 0.9);
  assert.equal(h.last().label, 'GRACIAS');
});

test('restore ignora datos corruptos o vacíos', () => {
  const h = new History();
  h.restore(null);
  assert.equal(h.size, 0);
  h.restore('{no es json');
  assert.equal(h.size, 0);
  h.restore('[{"label":"HOLA","confidence":0.9,"timestamp":1}]');
  assert.equal(h.size, 1);
  assert.equal(h.list()[0].label, 'HOLA');
});

test('restore respeta el límite máximo', () => {
  const h = new History({ maxEntries: 2 });
  const raw = JSON.stringify([
    { label: 'A', confidence: 0.9, timestamp: 1 },
    { label: 'B', confidence: 0.9, timestamp: 2 },
    { label: 'C', confidence: 0.9, timestamp: 3 },
  ]);
  h.restore(raw);
  assert.equal(h.size, 2);
  assert.deepEqual(h.list().map((e) => e.label), ['B', 'C']);
});

test('pushIfNew ignora labels nulos', () => {
  const h = new History();
  assert.equal(h.pushIfNew(null, 0.9), null);
  assert.equal(h.pushIfNew(undefined, 0.9), null);
  assert.equal(h.size, 0);
});

test('HISTORY_CONFIG por defecto: límite 50', () => {
  assert.equal(HISTORY_CONFIG.maxEntries, 50);
  const h = new History();
  for (let i = 0; i < 60; i++) {
    h.separate(); // cada una separada para poder repetir
    h.pushIfNew(`S${i}`, 0.9);
  }
  assert.equal(h.size, 50);
});