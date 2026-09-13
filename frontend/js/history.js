/**
 * Historial de señas CONFIRMADAS (F11) — módulo puro, sin DOM.
 *
 * Solo recibe señas confirmadas por TemporalSmoother (F10); nunca
 * predicciones RAW ni por-frame.
 *
 * Reglas:
 *   - pushIfNew(): no repite la misma seña de forma continua.
 *   - separate(): marca una separación (mano ausente): a partir de ahí
 *     la MISMA seña vuelve a ser registrable (permite construir frases).
 *   - Cambio de seña: entra directamente (no es duplicado continuo).
 *   - Límite configurable (MAX_HISTORY): al alcanzarlo se elimina la
 *     entrada más antigua.
 *   - Orden cronológico (inserción). Timestamp en ms.
 *
 * La persistencia en localStorage es OPCIONAL y se controla desde la UI:
 * solo guarda {label, confidence, timestamp} — nunca imágenes ni video.
 */

export const HISTORY_CONFIG = Object.freeze({
  maxEntries: 50, // límite: no crecimiento ilimitado
});

export class History {
  constructor(config = {}) {
    this.config = { ...HISTORY_CONFIG, ...config };
    this.entries = [];
    this._separated = true; // inicialmente permite registrar
  }

  /** La mano desapareció (o reset): la misma seña puede repetirse. */
  separate() {
    this._separated = true;
  }

  /**
   * Registra una confirmación si no es duplicado continuo.
   * @returns {object|null} la entrada agregada, o null si era duplicado.
   */
  pushIfNew(label, confidence) {
    if (label === undefined || label === null) return null;
    const last = this.last();
    if (!this._separated && last && last.label === label) {
      return null; // misma seña mantenida: no duplicar
    }
    this._separated = false;
    return this.add(label, confidence);
  }

  /** Agrega siempre (uso interno y restauración). */
  add(label, confidence) {
    const entry = {
      label,
      confidence: Number(confidence),
      timestamp: Date.now(),
    };
    this.entries.push(entry);
    if (this.entries.length > this.config.maxEntries) {
      this.entries.shift(); // elimina la más antigua
    }
    return entry;
  }

  last() {
    return this.entries.length ? this.entries[this.entries.length - 1] : null;
  }

  /** Copia cronológica (primera confirmada primero). */
  list() {
    return this.entries.map((e) => ({ ...e }));
  }

  get size() {
    return this.entries.length;
  }

  clear() {
    this.entries = [];
    this._separated = true;
  }

  // ------------------------------------------------------------------
  // Persistencia opcional (localStorage del navegador; nada sensible).
  // ------------------------------------------------------------------
  serialize() {
    return JSON.stringify(this.entries);
  }

  restore(raw) {
    if (!raw) return;
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return;
      const keep = parsed
        .filter((e) => e && typeof e.label === 'string' && typeof e.timestamp === 'number')
        .slice(-this.config.maxEntries);
      this.entries = keep;
      this._separated = true;
    } catch {
      /* datos corruptos: se ignora (historial empieza vacío) */
    }
  }
}