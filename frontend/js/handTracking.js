/**
 * Seguimiento de manos con MediaPipe — SIGNIA (F9).
 *
 * Envuelve @mediapipe/tasks-vision (HandLandmarker, ejecutado LOCALMENTE
 * en WASM: ningún frame sale del dispositivo).
 *
 * Salida compatible con features.js (F8):
 *   - left  -> bloque 0..62   (mano física IZQUIERDA, según handedness)
 *   - right -> bloque 63..125 (mano física DERECHA)
 *   - mano ausente -> null (features.js rellena con ceros)
 *
 * `mapHandsResult()` es una función PURA y testeable en Node sin MediaPipe;
 * `HandTracker.create()` se encarga solo de cargar la librería y el modelo.
 */

export const MEDIAPIPE_FILES = Object.freeze({
  // import() ESM se resuelve contra la URL del MÓDULO (frontend/js/), por eso
  // el specifier debe ser relativo (../vendor/...). Lo demás (wasm/task) lo
  // cargan fetch() internos de MediaPipe, que se resuelven contra el DOCUMENTO
  // (index.html en frontend/): ahí las rutas quedan relativas al documento.
  bundleUrl: '../vendor/mediapipe/vision_bundle.mjs',
  wasmDirUrl: 'vendor/mediapipe/wasm',
  modelUrl: 'vendor/mediapipe/hand_landmarker.task',
});

/**
 * Mapea el resultado de HandLandmarker al formato de features.js.
 * @param {object} result  {landmarks: [[{x,y,z}...]], handedness: [[{categoryName, score}]]}
 * @returns {{hands: number, left: number[][]|null, right: number[][]|null}}
 */
export function mapHandsResult(result) {
  const out = { hands: 0, left: null, right: null };
  if (!result || !Array.isArray(result.landmarks)) return out;
  out.hands = result.landmarks.length;
  let leftTaken = false;
  let rightTaken = false;
  result.landmarks.forEach((lmArray, i) => {
    const label = result.handedness?.[i]?.[0]?.categoryName ?? '';
    const pts = lmArray.map(p => [p.x, p.y, p.z]);
    if (label === 'Left' && !leftTaken) {
      out.left = pts;
      leftTaken = true;
    } else if (label === 'Right' && !rightTaken) {
      out.right = pts;
      rightTaken = true;
    } else if (!leftTaken) {
      // Etiqueta ausente o duplicada: fallback por orden de aparición.
      out.left = pts;
      leftTaken = true;
    } else {
      out.right = pts;
      rightTaken = true;
    }
  });
  return out;
}

export function eyesStatus(hands) {
  return hands >= 2 ? 'TWO_HANDS' : hands === 1 ? 'ONE_HAND' : 'NO_HAND';
}

export class HandTracker {
  /** @param {object} detector instancia de HandLandmarker (solo browser). */
  constructor(detector) {
    this.detector = detector;
    this._closed = false;
    // Estadísticas de detección en tiempo real (panel de diagnóstico).
    this.detectCalls = 0;
    this.lastTimestamp = null;
    this.lastDurationMs = null;
    this.lastHands = 0;
    this.lastError = null;
  }

  /**
   * Carga MediaPipe + el modelo de landmarks desde el directorio local
   * vendor/ (sin CDN). Intenta GPU y cae a CPU si WebGL no está disponible.
   * @param {object} opts override de MEDIAPIPE_FILES
   */
  static async create(opts = {}) {
    const files = { ...MEDIAPIPE_FILES, ...opts };
    let vision;
    try {
      vision = await import(files.bundleUrl);
    } catch (err) {
      throw new Error(`No se pudo cargar MediaPipe local: ${err.message}`);
    }
    const wasmResolver = await vision.FilesetResolver.forVisionTasks(files.wasmDirUrl);
    let detector = null;
    for (const delegate of ['GPU', 'CPU']) {
      try {
        detector = await vision.HandLandmarker.createFromOptions(wasmResolver, {
          baseOptions: {
            modelAssetPath: files.modelUrl,
            delegate,
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });
        break;
      } catch (err) {
        if (delegate === 'CPU') {
          throw new Error(`MediaPipe no pudo iniciar: ${err.message}`);
        }
      }
    }
    return new HandTracker(detector);
  }

  /**
   * Detecta manos en un frame de video (modo VIDEO).
   * @returns {Promise<{hands:number, left:number[][]|null, right:number[][]|null,
   *                    landmarks: [], handedness: []}>}
   */
  async detect(video, timestampMs) {
    if (this._closed) return mapHandsResult(null);
    this.detectCalls += 1;
    this.lastTimestamp = timestampMs;
    const start = performance.now();
    let result;
    try {
      // detectForVideo() es SÍNCRONO en tasks-vision (modo VIDEO). Si falla,
      // el error NUNCA se oculta: se registra en consola y el panel de
      // diagnóstico muestra lastError.
      result = this.detector.detectForVideo(video, timestampMs);
    } catch (err) {
      this.lastHands = 0;
      this.lastDurationMs = performance.now() - start;
      this.lastError = String(err?.message ?? err);
      console.error(
        `detectForVideo: llamada ${this.detectCalls}, ts=${timestampMs}:`,
        err
      );
      return mapHandsResult(null);
    }
    const mapped = mapHandsResult(result);
    const lm = Array.isArray(result?.landmarks) ? result.landmarks : [];
    mapped.landmarks = lm;
    mapped.handedness = Array.isArray(result?.handedness) ? result.handedness : [];
    this.lastHands = mapped.hands;
    this.lastDurationMs = performance.now() - start;
    this.lastError = null;
    return mapped;
  }

  /** Estado en vivo del bucle de detección (para el panel de diagnóstico). */
  stats() {
    return {
      detectCalls: this.detectCalls,
      lastTimestamp: this.lastTimestamp,
      lastDurationMs: this.lastDurationMs,
      lastHands: this.lastHands,
      lastError: this.lastError,
    };
  }

  close() {
    this._closed = true;
    try {
      this.detector?.close();
    } catch {
      /* ya cerrado */
    }
  }
}