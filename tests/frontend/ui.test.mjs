/**
 * Tests F12 — lógica de interfaz (módulo puro frontend/js/ui.js).
 * Sin DOM, sin píxeles: se prueba la DECISIÓN de presentación.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  UI_TEXT,
  predictionPlaceholder,
  modelStatusInfo,
  historyLineText,
  confidencePercent,
  voiceButtonText,
  cameraErrorTitle,
} from '../../frontend/js/ui.js';

// ---------------------------------------------------------------------------
// Placeholders: la UI NUNCA inventa etiquetas.
// ---------------------------------------------------------------------------
test('sin modelo: placeholder honesto de modelo no disponible', () => {
  assert.equal(predictionPlaceholder({ modelReady: false, hands: 0, reason: null }),
    'Modelo no disponible');
  assert.equal(predictionPlaceholder({ modelReady: false, hands: 2, reason: 'ok' }),
    'Modelo no disponible');
});

test('sin mano: "Coloca tu mano frente a la cámara"', () => {
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 0, reason: null }),
    'Coloca tu mano frente a la cámara');
});

test('stabilizing: "Esperando una seña..." (aún sin confirmar)', () => {
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 1, reason: 'stabilizing' }),
    'Esperando una seña...');
});

test('confianza insuficiente: mensaje explícito, sin etiqueta', () => {
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 1, reason: 'low_confidence' }),
    'Confianza insuficiente');
});

test('solo reason ok devuelve null (la seña confirmada se muestra)', () => {
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 1, reason: 'ok' }), null);
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 2, reason: 'ok' }), null);
});

test('reason desconocido: se trata como "en espera", no como seña', () => {
  assert.equal(predictionPlaceholder({ modelReady: true, hands: 1, reason: 'algo_raro' }),
    'Esperando una seña...');
});

// ---------------------------------------------------------------------------
// Estado del modelo (ausente / prueba / real)
// ---------------------------------------------------------------------------
test('sin modelo (summary no listo): "Modelo no disponible" + instrucción', () => {
  const info = modelStatusInfo(null);
  assert.equal(info.kind, 'missing');
  assert.equal(info.title, 'Modelo no disponible');
  assert.match(info.detail, /Necesitas cargar el modelo SIGNIA/);
});

test('summary sin listar (ready:false): tratado como modelo ausente', () => {
  const info = modelStatusInfo({ ready: false });
  assert.equal(info.kind, 'missing');
});

test('modelo de prueba (2 clases): badge de advertencia, sin versión', () => {
  const info = modelStatusInfo({
    ready: true, classes: ['A', 'B'], isTestModel: true, modelVersion: 'x',
  });
  assert.equal(info.kind, 'test');
  assert.equal(info.title, 'MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC');
  assert.equal(info.detail, null);
});

test('modelo real de 10 clases: "Modelo SIGNIA" + versión', () => {
  const info = modelStatusInfo({
    ready: true, classes: Array(10).fill('x'), isTestModel: false, modelVersion: '1.2.0',
  });
  assert.equal(info.kind, 'real');
  assert.equal(info.title, 'Modelo SIGNIA');
  assert.equal(info.detail, 'versión 1.2.0');
});

test('modelo real sin versión: detail null (no se inventa)', () => {
  const info = modelStatusInfo({
    ready: true, classes: Array(10).fill('x'), isTestModel: false, modelVersion: undefined,
  });
  assert.equal(info.kind, 'real');
  assert.equal(info.detail, null);
});

// ---------------------------------------------------------------------------
// Historial: la palabra domina, la confianza acompaña (opcional).
// ---------------------------------------------------------------------------
test('línea de historial con confianza: "HOLA · 91%"', () => {
  assert.equal(historyLineText({ label: 'HOLA', confidence: 0.9123 }, true), 'HOLA · 91%');
});

test('línea de historial sin confianza: solo la palabra', () => {
  assert.equal(historyLineText({ label: 'HOLA', confidence: 0.91 }, false), 'HOLA');
  assert.equal(historyLineText({ label: 'HOLA' }, true), 'HOLA');
});

test('entrada nula: cadena vacía', () => {
  assert.equal(historyLineText(null, true), '');
});

test('confianza redondeada con Math.round (0.5 -> 50%)', () => {
  assert.equal(confidencePercent(0.5), 50);
  assert.equal(confidencePercent(0.9999), 100);
});

// ---------------------------------------------------------------------------
// Voz
// ---------------------------------------------------------------------------
test('texto del botón de voz según estado (icono + texto)', () => {
  assert.equal(voiceButtonText(true), '🔊 Voz activada');
  assert.equal(voiceButtonText(false), '🔇 Voz desactivada');
});

// ---------------------------------------------------------------------------
// Cámara: títulos compactos y legibles por tipo de error.
// ---------------------------------------------------------------------------
test('título de error de cámara por tipo', () => {
  assert.equal(cameraErrorTitle('PERMISSION_DENIED'), 'Permiso de cámara denegado');
  assert.equal(cameraErrorTitle('NOT_FOUND'), 'Sin cámara disponible');
  assert.equal(cameraErrorTitle('BUSY'), 'Cámara ocupada');
  assert.equal(cameraErrorTitle('OVERCONSTRAINED'), 'Cámara no soportada');
});

test('error desconocido: título genérico, nunca un stack trace', () => {
  assert.equal(cameraErrorTitle('???'), 'Error de cámara');
  assert.doesNotMatch(cameraErrorTitle('???'), /Error:|at /);
});

// ---------------------------------------------------------------------------
// Anuncios aria-live: SOLO eventos significativos.
// ---------------------------------------------------------------------------
test('anuncio de seña confirmada (único evento aria-live de la predicción)', () => {
  assert.equal(UI_TEXT.signConfirmedAnnouncement('HOLA'), 'Seña confirmada: HOLA');
});

test('no existen textos que anuncien RAW, FPS o frames', () => {
  const all = JSON.stringify(UI_TEXT);
  assert.doesNotMatch(all, /FPS|RAW|frame|fotograma/i);
});