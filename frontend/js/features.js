/**
 * Normalización y empaquetado de landmarks — ES 100% DEL PROYECTO.
 *
 * Réplica EXACTA de ai/preprocessing/features.py (Python).
 * La paridad se verifica automáticamente (tests/test_export_numérica + tests/js_parity).
 *
 * NO dupliques esta lógica en otro archivo del frontend: este es el único
 * lugar que traduce landmarks → vector de features en el navegador.
 *
 * Formato del vector (126):
 *   [0..62]   mano física IZQUIERDA : 21 landmarks x (x, y, z)
 *   [63..125] mano física DERECHA   : 21 landmarks x (x, y, z)
 *   Mano ausente → su bloque en ceros.
 *
 * Normalización por mano:
 *   1. Origen en muñeca (índice 0).
 *   2. Escala = distancia muñeca → MCP medio (índice 9).
 *   3. landmark = (landmark - muñeca) / escala.
 */

'use strict';

const ANCHOR_LANDMARK = 0;   // muñeca
const SCALE_LANDMARK = 9;    // MCP medio (tamaño de palma)
const EPSILON = 1e-6;
const LANDMARKS_PER_HAND = 21;
const COORDS_PER_LANDMARK = 3;
const HANDS = 2;
const FEATURES_SIZE = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK * HANDS;
const HAND_CONNECTIONS = new Set([
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],
  [11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
].map(pair => pair.join(',')));

// MediaPipe: muñeca = 0, MCP medio = 9, puntas de dedos = 4,8,12,16,20

function normalizeHand(landmarks) {
  // landmarks: array plano o anidado de 21 puntos [x, y, z] (normalizados 0..1)
  if (landmarks.length === 63) {
    landmarks = Array.from({ length: 21 }, (_, i) => landmarks.slice(i * 3, i * 3 + 3));
  }
  const wrist = landmarks[ANCHOR_LANDMARK];
  let scale = 0.0;
  const base = landmarks[SCALE_LANDMARK];
  for (let i = 0; i < 3; i++) {
    const d = base[i] - wrist[i];
    scale += d * d;
  }
  scale = Math.sqrt(scale);
  if (scale < EPSILON) scale = EPSILON;
  return landmarks.map(p => [
    (p[0] - wrist[0]) / scale,
    (p[1] - wrist[1]) / scale,
    (p[2] - wrist[2]) / scale,
  ]);
}

function handsToFeatures(leftHand, rightHand) {
  // leftHand / rightHand: array de 21 puntos [x,y,z] o null si la mano no está.
  const features = new Float32Array(FEATURES_SIZE); // ceros por defecto
  if (leftHand) {
    const flat = normalizeHand(leftHand).flat(2);   // 63 valores
    features.set(new Float32Array(flat), 0);
  }
  if (rightHand) {
    const flat = normalizeHand(rightHand).flat(2);
    features.set(new Float32Array(flat), LANDMARKS_PER_HAND * COORDS_PER_LANDMARK);
  }
  return features;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ANCHOR_LANDMARK, SCALE_LANDMARK, EPSILON,
    LANDMARKS_PER_HAND, COORDS_PER_LANDMARK, HANDS, FEATURES_SIZE,
    HAND_CONNECTIONS, normalizeHand, handsToFeatures,
  };
}