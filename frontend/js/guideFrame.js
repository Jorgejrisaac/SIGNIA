/**
 * Guía temporal (Solución A, F15) — módulo PURO (sin DOM), testeable en Node.
 *
 * Replica la filosofía de select_guide_frame (ai/preprocessing/samples.py):
 * entre los frames de una ventana de frames VÁLIDOS (mano detectada), se
 * selecciona el vector de 126 dimensiones MÁS CERCANO (distancia L1) a la
 * mediana componente a componente de la configuración de la ventana.
 *
 * La ventana NO es un modelo temporal: solo estabiliza el frame representativo
 * que se envía al modelo actual. El contrato de entrada del modelo NO cambia:
 * sigue siendo [1, 126] (un único vector por predicción).
 *
 * Selección determinista, igual que en Python:
 *   1. mediana = mediana por componente de los vectores de la ventana
 *      (con conteo par se promedian los dos valores centrales, como np.median)
 *   2. distancias L1 de cada vector a la mediana
 *   3. argmin con desempate a la PRIMERA ocurrencia (como np.argmin)
 */

export const GUIDE_FRAME_CONFIG = Object.freeze({
  windowSize: 12,   // 10-15: estabiliza la elección sin añadir latencia
  minFrames: 3,     // con menos frames válidos NO se predice (arranque seguro)
  featureSize: 126, // contrato SIGNIA (dimensión fija)
});

export function createGuideFrameSelector(config = {}) {
  const windowSize = config.windowSize ?? GUIDE_FRAME_CONFIG.windowSize;
  const minFrames = config.minFrames ?? GUIDE_FRAME_CONFIG.minFrames;
  const featureSize = config.featureSize ?? GUIDE_FRAME_CONFIG.featureSize;

  let window = []; // solo vectores válidos (mano detectada, 126 números finitos)

  function isValid(features) {
    // handsToFeatures() devuelve Float32Array (features.js); los tests y la
    // paridad usan arrays planos. El contrato es la LONGITUD y los valores,
    // no el tipo concreto del contenedor.
    return (Array.isArray(features) || ArrayBuffer.isView(features)) &&
      features.length === featureSize &&
      features.every((v) => typeof v === 'number' && Number.isFinite(v));
  }

  /** Mediana por componente (mismo convenio que np.median). */
  function medianOf(vectors) {
    const n = vectors.length;
    const median = new Array(featureSize);
    for (let i = 0; i < featureSize; i++) {
      const col = new Array(n);
      for (let j = 0; j < n; j++) col[j] = vectors[j][i];
      col.sort((a, b) => a - b);
      const mid = n >> 1;
      median[i] = (n % 2 === 1) ? col[mid] : (col[mid - 1] + col[mid]) / 2;
    }
    return median;
  }

  /** Distancia L1 entre dos vectores de featureSize componentes. */
  function l1Distance(a, b) {
    let d = 0;
    for (let i = 0; i < featureSize; i++) d += Math.abs(a[i] - b[i]);
    return d;
  }

  /** Vector más cercano (L1) a la mediana; null si faltan frames. */
  function select() {
    if (window.length < minFrames) return null;
    const median = medianOf(window);
    let bestIndex = 0;
    let bestDistance = l1Distance(window[0], median);
    for (let i = 1; i < window.length; i++) {
      const d = l1Distance(window[i], median);
      if (d < bestDistance) { // desempate: primera ocurrencia (np.argmin)
        bestDistance = d;
        bestIndex = i;
      }
    }
    return window[bestIndex];
  }

  return {
    /**
     * Añade un frame válido a la ventana (FIFO, tamaño máximo windowSize)
     * y devuelve el vector guía seleccionado, o null si:
     *  - el frame NO es válido (no contamina la ventana), o
     *  - la ventana aún no tiene minFrames válidos (arranque seguro).
     */
    push(features) {
      if (!isValid(features)) return null;
      window.push(features);
      if (window.length > windowSize) window.shift();
      return select();
    },

    /** Limpia la ventana (mano perdida: los frames viejos no deben persistir). */
    reset() {
      window = [];
    },

    get size() {
      return window.length;
    },

    get isReady() {
      return window.length >= minFrames;
    },
  };
}