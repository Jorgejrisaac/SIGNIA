/**
 * Estados del sistema SIGNIA — tests de la máquina de estados (F9).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  SYSTEM_STATES,
  STATE_MESSAGES,
  STATE_ICONS,
  ERROR_STATES,
  SIGNIA_EXPECTED_CLASSES,
  isTestModel,
  TEST_MODEL_WARNING,
} from '../../frontend/js/states.js';

test('existen los 10 estados de F9', () => {
  const expected = [
    'INITIALIZING', 'CAMERA_PERMISSION', 'CAMERA_ACTIVE',
    'MODEL_LOADING', 'MODEL_READY', 'NO_HAND', 'ONE_HAND',
    'TWO_HANDS', 'ERROR_CAMERA', 'ERROR_MODEL',
  ];
  assert.deepEqual(Object.keys(SYSTEM_STATES).sort(), expected.sort());
});

test('todo estado tiene mensaje legible e icono', () => {
  for (const key of Object.keys(SYSTEM_STATES)) {
    const state = SYSTEM_STATES[key];
    assert.ok(STATE_MESSAGES[state], `mensaje para ${state}`);
    assert.ok(STATE_ICONS[state] !== undefined, `icono para ${state}`);
    assert.ok(STATE_MESSAGES[state].length > 3, `mensaje no vacío para ${state}`);
  }
});

test('estados de error identificados', () => {
  assert.deepEqual(ERROR_STATES, ['ERROR_CAMERA', 'ERROR_MODEL']);
});

test('mensajes de error son explícitos (no ambiguos)', () => {
  assert.ok(STATE_MESSAGES.ERROR_CAMERA.includes('Error de cámara'));
  assert.ok(STATE_MESSAGES.ERROR_MODEL.includes('Modelo no cargado'));
  assert.ok(STATE_MESSAGES.NO_HAND.includes('Coloca tu mano'));
});

test('clases esperadas del modelo real = 10 (config.py)', () => {
  assert.equal(SIGNIA_EXPECTED_CLASSES, 10);
});

test('isTestModel: metadata ausente o desconocida', () => {
  assert.equal(isTestModel(null), false);
  assert.equal(isTestModel({}), false);
});

test('isTestModel: modelo de prueba con pocas clases', () => {
  assert.equal(isTestModel({ n_classes: 2, classes: ['HOLA', 'ADIOS'] }), true);
});

test('isTestModel: modelo real con 10 clases', () => {
  assert.equal(isTestModel({ n_classes: 10, classes: Array(10).fill('X') }), false);
});

test('isTestModel: flag explícito is_test_model', () => {
  assert.equal(isTestModel({ n_classes: 10, is_test_model: true }), true);
});

test('advertencia de modelo de prueba está definida', () => {
  assert.ok(TEST_MODEL_WARNING.includes('MODELO DE PRUEBA'));
  assert.ok(TEST_MODEL_WARNING.includes('NO REPRESENTA RECONOCIMIENTO LSC'));
});