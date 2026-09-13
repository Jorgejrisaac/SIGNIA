/**
 * Estabilidad temporal (F10) — capa INDEPENDIENTE sobre la predicción.
 *
 * NO modifica el modelo ni sus probabilidades: recibe el vector de
 * probabilidades del predictor (RAW) y devuelve la predicción estabilizada.
 * Las probabilidades RAW se devuelven intactas (rawLabel, rawConfidence).
 *
 * Esto NO es una mejora de accuracy: solo evita cambios bruscos de
 * predicción entre frames consecutivos (ESTABILIDAD TEMPORAL).
 *
 * Reglas (deterministas, configurables en SMOOTHING_CONFIG):
 *   1. EMA por clase: smoothed[t][c] = alpha*cur[t][c] + (1-alpha)*smoothed[t-1][c].
 *   2. La clase "dominante" es el argmax de las probabilidades RAW del modelo.
 *   3. Una clase solo se CONFIRMA tras `stableFrames` frames consecutivos
 *      siendo dominante Y con maxProbability >= minConfidence.
 *   4. El umbral (maxProbability < minConfidence) se evalúa sobre las RAW:
 *      si no se alcanza, la salida es "Confianza insuficiente" — nunca se
 *      inventa una clase.
 *   5. Durante transiciones la confianza mostrada se congela: un frame
 *      ruidoso no produce "Confianza insuficiente" ni cambia la clase.
 *
 * Sin tensores: trabaja solo con arrays JS.
 */

/** Parámetros iniciales (NO óptimos): se ajustan sin tocar la lógica. */
export const SMOOTHING_CONFIG = Object.freeze({
  enabled: true,        // false -> la capa pasa la predicción RAW tal cual
  alpha: 0.35,          // EMA: 0 = nada cambia, 1 = sin suavizado
  minConfidence: 0.70,  // por debajo -> "Confianza insuficiente" (no inventar clase)
  stableFrames: 5,      // frames consecutivos para confirmar un cambio de clase
});

const REASONS = Object.freeze({
  STABILIZING: 'stabilizing',       // aún no hay clase confirmada
  OK: 'ok',                         // clase confirmada con confianza suficiente
  LOW_CONFIDENCE: 'low_confidence', // maxProbability bajo el umbral (RAW)
});

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

export class TemporalSmoother {
  constructor(config = {}) {
    this.config = { ...SMOOTHING_CONFIG, ...config };
    this.reset();
  }

  /**
   * Reinicia TODO el estado temporal (también tras ausencia de manos).
   * No arrastra probabilidades de una seña anterior.
   */
  reset() {
    this.ema = null;
    this.consecutive = 0;
    this.lastRawLabel = null;
    this.currentLabel = null;       // clase confirmada (índice) o null
    this.currentConfidence = null;  // confianza mostrada (EMA, congelada en transiciones)
  }

  /**
   * Ausencia de manos: limpia/pausa el estado. Devuelve null.
   * Cuando la mano vuelva, step() arranca de cero.
   */
  noHand() {
    this.reset();
    return null;
  }

  /**
   * @param {number[]} probs probabilidades RAW del modelo (sin mutar).
   * @returns {{
   *   reason: string, label: number|null, confidence: number|null,
   *   recognized: boolean, isStable: boolean,
   *   rawLabel: number, rawConfidence: number,
   *   consecutiveFrames: number, emaProbabilities: number[]
   * }}
   */
  step(probs) {
    const raw = Array.from(probs); // copia: nunca mutamos la entrada
    const rawIndex = argmax(raw);
    const rawConfidence = raw[rawIndex];
    const aboveThreshold = rawConfidence >= this.config.minConfidence;

    if (!this.config.enabled) {
      return {
        reason: aboveThreshold ? REASONS.OK : REASONS.LOW_CONFIDENCE,
        label: rawIndex,
        confidence: rawConfidence,
        recognized: aboveThreshold,
        isStable: true,
        isNewConfirmation: false, // sin F10 no hay SEÑA CONFIRMADA (F11)
        rawLabel: rawIndex,
        rawConfidence,
        consecutiveFrames: 1,
        emaProbabilities: raw,
      };
    }

    // EMA por clase (solo influye en la confianza mostrada, no en la clase)
    if (this.ema === null) {
      this.ema = raw.slice();
    } else {
      const a = this.config.alpha;
      for (let i = 0; i < raw.length; i++) {
        this.ema[i] = a * raw[i] + (1 - a) * this.ema[i];
      }
    }

    const prevConfirmed = this.currentLabel;

    // Dominancia: argmax RAW del modelo. Confirmación solo tras stableFrames
    // consecutivos Y con evidencia por encima del umbral.
    this.consecutive = rawIndex === this.lastRawLabel ? this.consecutive + 1 : 1;
    this.lastRawLabel = rawIndex;

    if (this.currentLabel === null) {
      if (this.consecutive >= this.config.stableFrames && aboveThreshold) {
        this.currentLabel = rawIndex;
        this.currentConfidence = this.ema[rawIndex];
      }
    } else if (rawIndex === this.currentLabel) {
      this.currentConfidence = this.ema[rawIndex];
    } else if (this.consecutive >= this.config.stableFrames && aboveThreshold) {
      // Nueva clase suficientemente estable: la anterior se abandona.
      this.currentLabel = rawIndex;
      this.currentConfidence = this.ema[rawIndex];
    }
    // Si no: transición — se sigue mostrando la clase confirmada y su
    // confianza queda congelada (un frame aislado no cambia nada).

    // F11: UNA sola vez por confirmación (no cada frame).
    const isNewConfirmation =
      this.currentLabel !== null && this.currentLabel !== prevConfirmed;

    if (this.currentLabel === null) {
      return {
        reason: REASONS.STABILIZING,
        label: null,
        confidence: null,
        recognized: false,
        isStable: false,
        isNewConfirmation: false,
        rawLabel: rawIndex,
        rawConfidence,
        consecutiveFrames: this.consecutive,
        emaProbabilities: this.ema.slice(),
      };
    }

    return {
      reason: aboveThreshold ? REASONS.OK : REASONS.LOW_CONFIDENCE,
      label: this.currentLabel,
      confidence: this.currentConfidence,
      recognized: aboveThreshold,
      isStable: true,
      isNewConfirmation,
      rawLabel: rawIndex,
      rawConfidence,
      consecutiveFrames: this.consecutive,
      emaProbabilities: this.ema.slice(),
    };
  }
}