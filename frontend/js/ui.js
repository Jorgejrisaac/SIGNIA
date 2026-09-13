/**
 * SIGNIA — lógica de interfaz F12 (módulo PURO, sin DOM, testeable en Node).
 *
 * Centraliza los textos y decisiones de presentación de la UI final:
 * placeholders de predicción, estado del modelo (ausente/prueba/real),
 * mensajes de cámara, historial y voz. Sin estilos ni DOM: la presentación
 * real vive en app.js + CSS.
 *
 * La UI NUNCA inventa etiquetas: sin modelo o sin confirmación solo se
 * muestran placeholders honestos.
 */

export const UI_TEXT = Object.freeze({
  NO_MODEL_TITLE: 'Modelo no disponible',
  NO_MODEL_DETAIL:
    'Necesitas cargar el modelo SIGNIA para iniciar el reconocimiento.',
  NO_MODEL_RETRY: 'Reintentar',
  TEST_MODEL_BADGE: 'MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC',
  REAL_MODEL_LABEL: 'Modelo SIGNIA',
  PREDICTION_NO_MODEL: 'Modelo no disponible',
  PREDICTION_NO_HAND: 'Coloca tu mano frente a la cámara',
  PREDICTION_WAITING: 'Esperando una seña...',
  PREDICTION_LOW_CONFIDENCE: 'Confianza insuficiente',
  CONFIDENCE_LABEL: (percent) => `Confianza ${percent}%`,
  VOICE_ON: '🔊 Voz activada',
  VOICE_OFF: '🔇 Voz desactivada',
  VOICE_UNSUPPORTED_TITLE: 'La voz no está disponible en este navegador',
  REPLAY: 'Reproducir última',
  HISTORY_EMPTY: 'Aún no hay señas confirmadas.',
  CLEAR_HISTORY: 'Limpiar historial',
  PRIVACY_NOTE:
    'Tu cámara se procesa localmente. El video no se envía a servidores.',
  LSC_PENDING: 'Reconocimiento LSC real: PENDIENTE DE VALIDACIÓN.',
  /** Anuncio aria-live: SOLO eventos significativos (seña confirmada). */
  signConfirmedAnnouncement: (label) => `Seña confirmada: ${label}`,
});

/**
 * Placeholder honesto para la predicción principal.
 * @param {{modelReady: boolean, hands: number, reason: string|null}} ctx
 * @returns {string|null} null si hay una seña confirmada que mostrar.
 */
export function predictionPlaceholder({ modelReady, hands, reason }) {
  if (!modelReady) return UI_TEXT.PREDICTION_NO_MODEL;
  if (!hands) return UI_TEXT.PREDICTION_NO_HAND;
  switch (reason) {
    case 'ok':
      return null;
    case 'low_confidence':
      return UI_TEXT.PREDICTION_LOW_CONFIDENCE;
    default: // stabilizing: aún sin clase confirmada
      return UI_TEXT.PREDICTION_WAITING;
  }
}

/**
 * Estado visual del modelo para la tarjeta de traducción.
 * @param {object|null} summary resultado de predictor.summary()
 * @returns {{kind: 'missing'|'test'|'real', title: string, detail: string|null}}
 */
export function modelStatusInfo(summary) {
  if (!summary || !Array.isArray(summary.classes)) {
    return {
      kind: 'missing',
      title: UI_TEXT.NO_MODEL_TITLE,
      detail: UI_TEXT.NO_MODEL_DETAIL,
    };
  }
  if (summary.isTestModel) {
    return { kind: 'test', title: UI_TEXT.TEST_MODEL_BADGE, detail: null };
  }
  return {
    kind: 'real',
    title: UI_TEXT.REAL_MODEL_LABEL,
    detail: summary.modelVersion ? `versión ${summary.modelVersion}` : null,
  };
}

/** Línea de una entrada del historial: la palabra domina, la confianza acompaña. */
export function historyLineText(entry, showConfidence) {
  if (!entry) return '';
  if (showConfidence && typeof entry.confidence === 'number') {
    return `${entry.label} · ${Math.round(entry.confidence * 100)}%`;
  }
  return entry.label;
}

export function confidencePercent(confidence) {
  return Math.round(confidence * 100);
}

export function voiceButtonText(enabled) {
  return enabled ? UI_TEXT.VOICE_ON : UI_TEXT.VOICE_OFF;
}

/**
 * Título compacto del banner según el error de cámara (camera.js).
 * El detalle legible vive en CAMERA_ERROR_MESSAGES; aquí solo el encabezado.
 */
export function cameraErrorTitle(error) {
  switch (error) {
    case 'PERMISSION_DENIED':
      return 'Permiso de cámara denegado';
    case 'NOT_FOUND':
      return 'Sin cámara disponible';
    case 'BUSY':
      return 'Cámara ocupada';
    case 'OVERCONSTRAINED':
      return 'Cámara no soportada';
    case 'INSECURE_CONTEXT':
      return 'Cámara móvil requiere HTTPS';
    default:
      return 'Error de cámara';
  }
}
