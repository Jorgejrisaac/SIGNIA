/**
 * Flujo de confirmación (F11) — orquestación PURA y testeable.
 *
 * Conecta TemporalSmoother (F10) con History y Speech:
 *
 *   SEÑA CONFIRMADA (isNewConfirmation + reason ok)
 *     → history.pushIfNew()            (sin duplicados continuos)
 *     → onConfirm callback             (voz + UI en app.js)
 *
 * RAW prediction o SMOOTHED por sí solos NUNCA generan historial ni voz:
 * solo las confirmaciones de F10 entran aquí.
 */

export function createConfirmedFlow({ history, onConfirm }) {
  return {
    /**
     * Procesa el resultado de TemporalSmoother.step().
     * @param {object} smoothed resultado de F10
     * @returns {object|null} la entrada de historial agregada o null.
     */
    onFrame(smoothed) {
      if (!smoothed) return null;
      const confirmed =
        smoothed.reason === 'ok' &&
        smoothed.isNewConfirmation === true &&
        smoothed.label !== null;
      if (!confirmed) return null;

      const entry = history.pushIfNew(smoothed.label, smoothed.confidence);
      if (!entry) return null; // duplicado continuo: no registrar ni hablar
      onConfirm?.(smoothed.label, entry);
      return entry;
    },

    /** Separación (mano ausente): permite repetir la misma seña. */
    separate() {
      history.separate();
    },
  };
}