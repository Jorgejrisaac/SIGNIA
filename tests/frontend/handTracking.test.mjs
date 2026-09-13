/**
 * MediaPipe -> formato features.js: mapeo de manos y mano ausente (F9).
 * mapHandsResult() es puro: no requiere MediaPipe ni cámara.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { mapHandsResult, eyesStatus } from '../../frontend/js/handTracking.js';

const require = createRequire(import.meta.url);
const feats = require('../../frontend/js/features.js');

function lm(value) {
  return { x: value, y: value, z: value };
}

function hands(leftCount, rightCount) {
  const landmarks = [];
  const handedness = [];
  for (let i = 0; i < leftCount; i++) {
    landmarks.push(Array.from({ length: 21 }, (_, j) => lm(j * 0.01)));
    handedness.push([{ categoryName: 'Left', score: 0.99 - i * 0.01 }]);
  }
  for (let i = 0; i < rightCount; i++) {
    landmarks.push(Array.from({ length: 21 }, (_, j) => lm(0.5 + j * 0.01)));
    handedness.push([{ categoryName: 'Right', score: 0.9 - i * 0.01 }]);
  }
  return { landmarks, handedness };
}

test('sin manos: cuenta 0, ambas manos null', () => {
  const r = mapHandsResult({ landmarks: [], handedness: [] });
  assert.equal(r.hands, 0);
  assert.equal(r.left, null);
  assert.equal(r.right, null);
});

test('resultado nulo o malformado no rompe', () => {
  const r = mapHandsResult(null);
  assert.equal(r.hands, 0);
  assert.deepEqual(mapHandsResult({}), { hands: 0, left: null, right: null });
});

test('mano derecha sola -> bloque derecho', () => {
  const r = mapHandsResult(hands(0, 1));
  assert.equal(r.hands, 1);
  assert.equal(r.left, null);
  assert.ok(r.right, 'debe haber mano derecha');
  assert.equal(r.right.length, 21);
  assert.deepEqual(r.right[0], [0.5, 0.5, 0.5]);
});

test('mano izquierda sola -> bloque izquierdo', () => {
  const r = mapHandsResult(hands(1, 0));
  assert.equal(r.hands, 1);
  assert.ok(r.left, 'debe haber mano izquierda');
  assert.equal(r.right, null);
  assert.deepEqual(r.left[0], [0, 0, 0]);
});

test('dos manos: left y right asignadas', () => {
  const r = mapHandsResult(hands(1, 1));
  assert.equal(r.hands, 2);
  assert.ok(r.left && r.right, 'ambas manos presentes');
});

test('etiquetas duplicadas: fallback por orden de aparición', () => {
  const result = hands(2, 0); // dos manos etiquetadas "Left"
  const r = mapHandsResult(result);
  assert.equal(r.hands, 2);
  assert.ok(r.left && r.right, 'las dos manos se asignan a bloques distintos');
});

test('feature vector: izquierda en 0..62, derecha en 63..125, ausente en ceros', () => {
  const r = mapHandsResult(hands(1, 0)); // solo izquierda
  const f = feats.handsToFeatures(r.left, r.right);
  assert.equal(f.length, 126);
  const leftBlock = Array.from(f.slice(0, 63));
  const rightBlock = Array.from(f.slice(63));
  assert.ok(leftBlock.some(v => v !== 0), 'bloque izquierdo tiene valores');
  assert.ok(rightBlock.every(v => v === 0), 'bloque derecho en ceros (mano ausente)');
});

test('eyesStatus: 0/1/2 manos', () => {
  assert.equal(eyesStatus(0), 'NO_HAND');
  assert.equal(eyesStatus(1), 'ONE_HAND');
  assert.equal(eyesStatus(2), 'TWO_HANDS');
});