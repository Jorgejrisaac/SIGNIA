/**
 * Cámara SIGNIA — clasificación de errores y flujo sin hardware (F9).
 * La apertura real de cámara es prueba MANUAL (ver docs/prueba_manual_f9.md).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  openCamera,
  stopStream,
  classifyCameraError,
  cameraErrorMessage,
  CAMERA_ERRORS,
  CAMERA_ERROR_MESSAGES,
} from '../../frontend/js/camera.js';

const cases = [
  ['NotAllowedError', CAMERA_ERRORS.PERMISSION_DENIED],
  ['PermissionDeniedError', CAMERA_ERRORS.PERMISSION_DENIED],
  ['NotFoundError', CAMERA_ERRORS.NOT_FOUND],
  ['DevicesNotFoundError', CAMERA_ERRORS.NOT_FOUND],
  ['NotReadableError', CAMERA_ERRORS.BUSY],
  ['TrackStartError', CAMERA_ERRORS.BUSY],
  ['OverconstrainedError', CAMERA_ERRORS.OVERCONSTRAINED],
  ['RareErrorXyz', CAMERA_ERRORS.UNKNOWN],
];

test('clasificación de errores de cámara', () => {
  for (const [name, expected] of cases) {
    assert.equal(classifyCameraError({ name }), expected, `name=${name}`);
  }
});

test('cada error tiene mensaje explicativo distinto', () => {
  const msgs = new Set(Object.values(CAMERA_ERROR_MESSAGES));
  assert.equal(msgs.size, Object.keys(CAMERA_ERROR_MESSAGES).length);
  for (const [name] of cases) {
    const msg = cameraErrorMessage({ name });
    assert.ok(msg.length > 10, `mensaje para ${name}`);
  }
});

test('sin mediaDevices (o sin navegador) -> estado NOT_FOUND claro', async () => {
  const result = await openCamera();
  assert.equal(result.ok, false);
  assert.equal(result.error, CAMERA_ERRORS.NOT_FOUND);
  assert.ok(result.message.length > 10);
});

test('stopStream tolera null y streams vacíos', () => {
  assert.doesNotThrow(() => stopStream(null));
  assert.doesNotThrow(() => stopStream({ getTracks: () => [] }));
});

test('stopStream detiene todas las pistas', () => {
  const stopped = [];
  const track = () => ({ stop: () => stopped.push(1) });
  stopStream({ getTracks: () => [track(), track()] });
  assert.equal(stopped.length, 2);
});