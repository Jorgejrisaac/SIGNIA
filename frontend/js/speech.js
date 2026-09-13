/**
 * Voz SIGNIA (F11) — SpeechSynthesis local (window.speechSynthesis).
 *
 * - Solo recibe SEÑAS CONFIRMADAS (la orquestación la hace app.js/flow.js).
 * - Sin APIs externas, sin envío de texto a servidores.
 * - Idioma: es-CO → es-ES → cualquier voz es-* (nunca asume una voz).
 * - Política anti-acumulación: con `replaceOnNew` (por defecto) cada nueva
 *   seña confirmada hace cancel() + speak(); la voz representa SIEMPRE la
 *   última confirmación.
 * - Etiquetas internas del modelo → texto hablado: labelToSpeechText().
 *
 * Inyectable para tests de Node: { synthesis, Utterance }.
 */

export const SPEECH_CONFIG = Object.freeze({
  enabled: true,          // preferencia base (UI + localStorage la ajusta)
  replaceOnNew: true,     // anti-colas: cancel() antes de speak() en nueva seña
  rate: 1.0,
  pitch: 1.0,
  languagePriority: ['es-CO', 'es-ES', 'es-*'],
});

/**
 * Etiqueta interna (sin tildes, mayúsculas) → texto a pronunciar.
 * Corregir aquí NO implica tocar el modelo ni metadata.
 */
const LABEL_TO_TEXT = {
  HOLA: 'Hola',
  ADIOS: 'Adiós',
  GRACIAS: 'Gracias',
  POR_FAVOR: 'Por favor',
  SI: 'Sí',
  NO: 'No',
  AYUDA: 'Ayuda',
  BIEN: 'Bien',
  MAL: 'Mal',
  AMIGO: 'Amigo',
};

export function labelToSpeechText(label) {
  const direct = LABEL_TO_TEXT[label];
  if (direct) return direct;
  if (typeof label !== 'string' || !label) return '';
  // fallback seguro para etiquetas desconocidas: primera letra en mayúscula
  return label.toLowerCase().replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());
}

export function speechSupported(synthesis, Utterance) {
  return Boolean(
    synthesis &&
    Utterance &&
    typeof synthesis.speak === 'function' &&
    typeof synthesis.getVoices === 'function'
  );
}

export class SpeechController {
  /**
   * @param {object} deps synthesis (speechSynthesis), Utterance, config.
   */
  constructor({ synthesis, Utterance, config = {} } = {}) {
    this.synthesis = synthesis ?? globalThis.speechSynthesis ?? null;
    this.Utterance = Utterance ?? globalThis.SpeechSynthesisUtterance ?? null;
    this.config = { ...SPEECH_CONFIG, ...config };
    this.supported = speechSupported(this.synthesis, this.Utterance);
    this.enabled = this.config.enabled && this.supported;
    this._voices = null;
    this._bindVoicesChanged();
  }

  _bindVoicesChanged() {
    if (!this.synthesis) return;
    this._reloadVoices();
    try {
      this.synthesis.addEventListener?.('voiceschanged', () => this._reloadVoices());
    } catch {
      /* navegadores antiguos: onvoiceschanged */
    }
    if (this.synthesis.onvoiceschanged === null || this.synthesis.onvoiceschanged === undefined) {
      try {
        this.synthesis.onvoiceschanged = () => this._reloadVoices();
      } catch {
        /* sin soporte: se reintenta en cada speak() */
      }
    }
  }

  _reloadVoices() {
    try {
      this._voices = Array.from(this.synthesis.getVoices?.() ?? []);
    } catch {
      this._voices = [];
    }
  }

  /**
   * Voces en español por prioridad: es-CO, es-ES, cualquier es-*.
   * @returns {{voice: object|null, lang: string|null}}
   */
  resolveSpanishVoice() {
    const voices = this._voices ?? [];
    const es = (p) => voices.find((v) =>
      String(v.lang ?? '').toLowerCase().startsWith(p.toLowerCase()));
    for (const pref of this.config.languagePriority) {
      if (pref === 'es-*') {
        const any = es('es');
        if (any) return { voice: any, lang: any.lang };
      } else {
        const exact = es(pref);
        if (exact) return { voice: exact, lang: exact.lang };
      }
    }
    return { voice: null, lang: null };
  }

  /**
   * Habla una seña CONFIRMADA (recibe la etiqueta interna del modelo).
   * @returns {boolean} true si se encoló la locución.
   */
  speak(label) {
    if (!this.enabled) return false;
    const text = labelToSpeechText(label);
    if (!text) return false;
    if (this._voices === null || (Array.isArray(this._voices) && this._voices.length === 0)) {
      this._reloadVoices(); // reintento: voces pueden tardar en cargar
    }
    if (this.config.replaceOnNew) this.cancel();
    try {
      // Safari/iOS and some Android browsers can leave synthesis paused after
      // a previous utterance or after the page was backgrounded.
      this.synthesis.resume?.();
      const utter = new this.Utterance(text);
      const { voice, lang } = this.resolveSpanishVoice();
      if (voice) utter.voice = voice;
      utter.lang = voice?.lang ?? lang ?? 'es-CO';
      utter.rate = this.config.rate;
      utter.pitch = this.config.pitch;
      this.synthesis.speak(utter);
      return true;
    } catch {
      return false; // manejo de error: la app nunca debe romper por la voz
    }
  }

  cancel() {
    try {
      this.synthesis?.cancel?.();
    } catch {
      /* sin voz */
    }
  }

  /** Preferencia de voz (persistible en localStorage por la UI). */
  setEnabled(value) {
    this.enabled = Boolean(value) && speechSupported(this.synthesis, this.Utterance);
    if (!this.enabled) this.cancel();
    return this.enabled;
  }
}
