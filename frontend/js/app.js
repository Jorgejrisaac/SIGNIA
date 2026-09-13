/**
 * SIGNIA — aplicación principal.
 * F9: webcam + MediaPipe + modelo TF.js + estados.
 * F10: estabilidad temporal (señas CONFIRMADAS).
 * F11: historial + voz (SpeechSynthesis local).
 * F12: UI final del MVP (jerarquía, accesibilidad, responsive, diagnóstico
 * oculto con Alt+D, voz/historial/confianza como controles explícitos).
 *
 * Privacidad: toda la detección y la inferencia ocurren localmente; ningún
 * frame sale del dispositivo (MediaPipe y TF.js se cargan desde vendor/,
 * sin CDN). La voz usa SpeechSynthesis local; el historial solo guarda
 * etiquetas + confianza + timestamp (persistencia opcional, nada sensible).
 *
 * aria-live SOLO anuncia eventos significativos (seña confirmada), nunca
 * fotogramas, RAW ni FPS.
 */

import { openCamera, stopStream, CAMERA_ERROR_MESSAGES } from './camera.js';
import { HandTracker, eyesStatus } from './handTracking.js';
import { mapLandmarkToOverlay } from './overlay.js';
import { makePredictor, PredictError } from './predictor.js';
import { SYSTEM_STATES, STATE_MESSAGES, STATE_ICONS } from './states.js';
import { TemporalSmoother, SMOOTHING_CONFIG } from './temporalSmoother.js';
import { createGuideFrameSelector } from './guideFrame.js';
import { History } from './history.js';
import { SpeechController } from './speech.js';
import { createConfirmedFlow } from './flow.js';
import {
  predictionPlaceholder,
  modelStatusInfo,
  historyLineText,
  confidencePercent,
  voiceButtonText,
  cameraErrorTitle,
  UI_TEXT,
} from './ui.js';

// ---------------------------------------------------------------------------
// Configuración (rutas relativas a index.html; servir frontend/ como raíz).
// ---------------------------------------------------------------------------
const MODEL_BASE = 'models/signia/';
const MODEL_URL = MODEL_BASE + 'model.json';
const METADATA_URL = MODEL_BASE + 'metadata.json';

const FEATURES_SIZE = 126;
const LIVE_MIN_CONFIDENCE = 0.55;
const DIAG_ENABLED = new URLSearchParams(globalThis.location?.search ?? '').has('debug');

// --- LIVE VECTOR DIAGNOSTIC (instrumentation, no behavior change) ---
const DIAG_CENTROIDS = {
  si: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.244374,-0.136372,-0.124914,0.45897,-0.449381,-0.185711,0.47517,-0.754774,-0.235633,0.372584,-0.950198,-0.258007,0.280576,-0.966121,-0.070255,0.389152,-1.03,-0.20494,0.377434,-0.722121,-0.259079,0.326517,-0.667184,-0.281206,0.093598,-0.985546,-0.06654,0.216362,-0.971808,-0.198619,0.221138,-0.613693,-0.211619,0.168552,-0.626678,-0.194877,-0.081147,-0.925677,-0.087151,0.041432,-0.86531,-0.235659,0.070079,-0.545462,-0.193454,0.020794,-0.553278,-0.134843,-0.25273,-0.826972,-0.117119,-0.122315,-0.774076,-0.206629,-0.076616,-0.549197,-0.166791,-0.117414,-0.530038,-0.114711],
  no: [0,0,0,-0.018861,-0.021618,-0.011114,-0.0283,-0.054221,-0.017267,-0.016556,-0.083536,-0.023709,-0.000411,-0.104139,-0.029308,-0.026031,-0.101154,-0.00619,-0.027785,-0.144221,-0.017611,-0.027303,-0.169533,-0.024913,-0.025833,-0.192902,-0.030115,-0.007394,-0.099395,-0.00748,-0.006641,-0.127853,-0.024341,-0.010339,-0.096125,-0.027345,-0.011429,-0.077175,-0.024968,0.009407,-0.091924,-0.010834,0.010079,-0.112488,-0.026409,0.003798,-0.081534,-0.021093,0.002607,-0.066324,-0.01294,0.025236,-0.079346,-0.015513,0.026539,-0.103601,-0.023447,0.019663,-0.084682,-0.01775,0.017054,-0.072353,-0.011151,0,0,0,0.18151,-0.19848,-0.087378,0.307729,-0.482122,-0.157739,0.278487,-0.703946,-0.23246,0.141946,-0.80372,-0.3033,0.08903,-0.932242,-0.09469,0.097883,-1.286985,-0.193453,0.106288,-1.519406,-0.248162,0.099832,-1.716896,-0.285937,-0.058727,-0.884967,-0.117983,0.059498,-1.043662,-0.27402,0.137705,-0.781092,-0.309446,0.13096,-0.629734,-0.297262,-0.168284,-0.778445,-0.150486,-0.020657,-0.854695,-0.306776,0.061127,-0.609157,-0.278605,0.047028,-0.495513,-0.220496,-0.256393,-0.630732,-0.190091,-0.108334,-0.701919,-0.286448,-0.030089,-0.540317,-0.252466,-0.043504,-0.452838,-0.202842],
};

// The generated SI reference had one extra zero at the Left/Right boundary.
// Keep the diagnostic reference on the same 126-dimensional contract.
if (DIAG_CENTROIDS.si.length === FEATURES_SIZE + 1 &&
    DIAG_CENTROIDS.si[66] === 0 && DIAG_CENTROIDS.si[67] === 0.244374) {
  DIAG_CENTROIDS.si.splice(66, 1);
}

const DIAG_REFERENCE_OK = [DIAG_CENTROIDS.si, DIAG_CENTROIDS.no].every(
  (reference) => reference.length === FEATURES_SIZE &&
    reference.every((value) => Number.isFinite(value))
);

function diagDistL1(a, b) {
  if (a.length !== b.length || !DIAG_REFERENCE_OK) return Number.NaN;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}
function diagBlockL2(vec, start, end) { let s = 0; for (let i = start; i < end; i++) s += vec[i] * vec[i]; return Math.sqrt(s); }
let diagFrameCount = 0;
const diagRecords = [];
const diagThresholdStats = {
  threshold: LIVE_MIN_CONFIDENCE,
  currentClass: null,
  currentRun: 0,
  maxRunByClass: {},
  framesAboveByClass: {},
};

if (DIAG_ENABLED) {
  globalThis.__SIGNIA_LIVE_DIAG__ = {
    reference: {
      featureSize: FEATURES_SIZE,
      siLength: DIAG_CENTROIDS.si.length,
      noLength: DIAG_CENTROIDS.no.length,
      valid: DIAG_REFERENCE_OK,
    },
    threshold: diagThresholdStats,
    latest: null,
    records: diagRecords,
  };
}

function updateDiagThreshold(raw, rawIndex) {
  const above = raw.confidence >= diagThresholdStats.threshold;
  if (!above) {
    diagThresholdStats.currentClass = null;
    diagThresholdStats.currentRun = 0;
    return false;
  }

  if (diagThresholdStats.currentClass === raw.label) {
    diagThresholdStats.currentRun += 1;
  } else {
    diagThresholdStats.currentClass = raw.label;
    diagThresholdStats.currentRun = 1;
  }

  diagThresholdStats.framesAboveByClass[raw.label] =
    (diagThresholdStats.framesAboveByClass[raw.label] ?? 0) + 1;
  diagThresholdStats.maxRunByClass[raw.label] = Math.max(
    diagThresholdStats.maxRunByClass[raw.label] ?? 0,
    diagThresholdStats.currentRun
  );
  return true;
}

function resetDiagThreshold() {
  diagThresholdStats.currentClass = null;
  diagThresholdStats.currentRun = 0;
}

// Preferencias locales (solo voz/historial; NUNCA cámara ni landmarks).
const LS_VOICE = 'SIGNIA_VOICE_ENABLED';
const LS_HISTORY = 'SIGNIA_HISTORY';
const LS_HISTORY_PERSIST = 'SIGNIA_HISTORY_PERSIST';

const $ = (id) => document.getElementById(id);

const dom = {
  video: $('video'),
  canvas: $('overlay'),
  stagePlaceholder: $('stage-placeholder'),
  predictionLabel: $('prediction-label'),
  predictionConfidence: $('prediction-confidence'),
  confidenceText: $('confidence-text'),
  confidenceFill: $('confidence-fill'),
  cameraChip: $('chip-camera'),
  modelChip: $('chip-model'),
  handsChip: $('chip-hand'),
  banner: $('error-banner'),
  bannerTitle: $('error-banner-title'),
  bannerText: $('error-banner-text'),
  retryCamera: $('btn-retry-camera'),
  modelStatus: $('model-status'),
  modelStatusTitle: $('model-status-title'),
  modelStatusDetail: $('model-status-detail'),
  retryModel: $('btn-retry-model'),
  liveRegion: $('live-region'),
  historyList: $('history-list'),
  historyEmpty: $('history-empty'),
  historyLive: $('history-live'),
  btnClearHistory: $('btn-clear-history'),
  btnToggleVoice: $('btn-toggle-voice'),
  btnReplay: $('btn-replay'),
  toggleConfidence: $('toggle-confidence'),
  togglePersist: $('toggle-persist'),
  btnDiagnostics: $('btn-diagnostics'),
  diagCamera: $('diag-camera'),
  diagMediapipe: $('diag-mediapipe'),
  diagModel: $('diag-model'),
  diagInputShape: $('diag-input-shape'),
  diagOutputShape: $('diag-output-shape'),
  diagModelVersion: $('diag-model-version'),
  diagHands: $('diag-hands'),
  diagFps: $('diag-fps'),
  diagLoop: $('diag-loop'),
  diagDetectCalls: $('diag-detect-calls'),
  diagLastTs: $('diag-last-ts'),
  diagDetectResult: $('diag-detect-result'),
  diagModelInfo: $('diag-model-info'),
  diagRaw: $('diag-raw'),
  diagSmoothed: $('diag-smoothed'),
  diagSmoothing: $('diag-smoothing'),
  diagPanel: $('diagnostics'),
};

let tracker = null;
let predictor = null;
const smoother = new TemporalSmoother({ minConfidence: LIVE_MIN_CONFIDENCE });
const guideFrame = createGuideFrameSelector();
const history = new History();
const speech = new SpeechController();
let flow = null;
let stream = null;
let rafId = 0;
let fps = 0;
let showConfidence = true;
let frames = 0;
let fpsWindowStart = performance.now();

const chipStates = {
  camera: SYSTEM_STATES.INITIALIZING,
  model: SYSTEM_STATES.INITIALIZING,
  hands: SYSTEM_STATES.NO_HAND,
};

function setChip(which, state, detail = '') {
  chipStates[which] = state;
  const chip = dom[which + 'Chip'];
  const icon = STATE_ICONS[state] ?? '•';
  const text = detail || STATE_MESSAGES[state] || state;
  chip.dataset.state = state;
  chip.querySelector('.chip-icon').textContent = icon;
  chip.querySelector('.chip-text').textContent = text;
  chip.setAttribute('aria-label', text);
}

/**
 * Anuncio aria-live SOLO de eventos significativos: cámara y modelo cambian
 * pocas veces; las manos cambian por fotograma y NUNCA se anuncian aquí
 * (las confirmaciones de seña se anuncian en onConfirm).
 */
function announce() {
  const msg = [
    STATE_MESSAGES[chipStates.camera],
    STATE_MESSAGES[chipStates.model],
  ].join('. ');
  if (dom.liveRegion.textContent !== msg) {
    dom.liveRegion.textContent = msg;
  }
}

function showBanner(title, message) {
  dom.bannerTitle.textContent = title;
  dom.bannerText.textContent = message;
  dom.banner.hidden = false;
}

function hideBanner() {
  dom.banner.hidden = true;
}

function setHandState(state) {
  setChip('hands', state);
  dom.diagHands.textContent = state === 'NO_HAND' ? '0' : state === 'ONE_HAND' ? '1' : '2';
}

/** Estado en vivo del bucle de detección (llamadas, timestamp, errores). */
function updateDetectionStats() {
  if (!tracker) {
    dom.diagLoop.textContent = 'STOPPED';
    dom.diagDetectCalls.textContent = '0';
    dom.diagLastTs.textContent = '—';
    dom.diagDetectResult.textContent = 'no tracker';
    return;
  }
  const s = tracker.stats();
  dom.diagLoop.textContent = 'RUNNING';
  dom.diagDetectCalls.textContent = String(s.detectCalls);
  dom.diagLastTs.textContent =
    s.lastTimestamp === null ? '—' : `${s.lastTimestamp.toFixed(1)} ms`;
  dom.diagDetectResult.textContent =
    s.lastError === null && s.detectCalls > 0
      ? `OK (${(s.lastDurationMs ?? 0).toFixed(1)} ms)`
      : s.lastError === null ? 'OK' : `ERROR — ${s.lastError}`;
}

// ---------------------------------------------------------------------------
// Predicción principal (SOLO resultado confirmado; placeholders honestos).
// ---------------------------------------------------------------------------
function renderPrediction(result, modelReady) {
  const placeholder = predictionPlaceholder({
    modelReady,
    hands: result === null ? 0 : 1,
    reason: result?.reason ?? null,
  });

  if (placeholder !== null) {
    dom.predictionLabel.textContent = placeholder;
    dom.predictionLabel.classList.add('placeholder');
    dom.predictionLabel.classList.remove('pulse');
    dom.predictionConfidence.hidden = true;
    return;
  }

  const label = predictor.classes[result.label];
  const pct = confidencePercent(result.confidence);
  dom.predictionLabel.textContent = label;
  dom.predictionLabel.classList.remove('placeholder');
  dom.confidenceText.textContent = UI_TEXT.CONFIDENCE_LABEL(pct);
  dom.confidenceFill.style.width = `${Math.min(100, pct)}%`;
  dom.predictionConfidence.hidden = false;
}

/** Microinteracción solo al CONFIRMAR una seña (nunca por fotograma). */
function pulsePrediction() {
  dom.predictionLabel.classList.remove('pulse');
  void dom.predictionLabel.offsetWidth; // reinicia la animación
  dom.predictionLabel.classList.add('pulse');
}

function updateDiagPrediction(raw, smoothed, selected) {
  if (!raw || !predictor?.isReady) {
    dom.diagRaw.textContent = '—';
    dom.diagSmoothed.textContent = '—';
    return;
  }
  dom.diagRaw.textContent =
    `${raw.label} — ${raw.confidence.toFixed(3)}`;
  if (smoothed.label !== null) {
    dom.diagSmoothed.textContent =
      `${predictor.classes[smoothed.label]} — ` +
      `${(smoothed.confidence ?? 0).toFixed(3)} (${smoothed.reason})`;
  } else {
    dom.diagSmoothed.textContent = `— (${smoothed.reason})`;
  }
  // --- VECTOR DIAGNOSTIC DISPLAY ---
  if (selected && DIAG_ENABLED) {
    const leftL2 = diagBlockL2(selected, 0, 63);
    const rightL2 = diagBlockL2(selected, 63, 126);
    const distSI = diagDistL1(selected, DIAG_CENTROIDS.si);
    const distNO = diagDistL1(selected, DIAG_CENTROIDS.no);
    const closer = distSI < distNO ? 'SI' : 'NO';
    const rawIdx = raw.probabilities.indexOf(Math.max(...raw.probabilities));
    // Show in console for F12 inspection
    if (diagFrameCount % 10 === 1) {
      console.log(`[VECTOR] idx=${rawIdx} ${raw.label}(${raw.confidence.toFixed(3)}) L_L2=${leftL2.toFixed(3)} R_L2=${rightL2.toFixed(3)} dSI=${distSI.toFixed(1)} dNO=${distNO.toFixed(1)} closer=${closer}`);
    }
  }
}

// ---------------------------------------------------------------------------
// Bucle principal (A PRUEBA DE MUERTE SILENCIOSA)
// ---------------------------------------------------------------------------
let overlayMirrored = true; // el video se espeja por CSS (scaleX(-1))

/**
 * El espejo del overlay se detecta del CSS REAL del video (matriz de
 * transformación): única fuente de verdad. El canvas NUNCA se transforma;
 * el espejo se aplica en el mapeo de coordenadas (x = ancho - x).
 */
function detectVideoMirror() {
  try {
    const t = getComputedStyle(dom.video).transform;
    if (t && t !== 'none') {
      const m = /matrix\(([^,]+)/.exec(t);
      overlayMirrored = !m || Number(m[1]) < 0;
    }
  } catch {
    /* mantiene el valor por defecto */
  }
}

/**
 * Ajusta el canvas a su CAJA real (la misma que el <video>: inset:0,
 * width/height 100%). Nunca se usa video.videoWidth aquí: el CSS recorta
 * el frame (object-fit: cover) y el mapeo compensa ese recorte.
 * Devuelve la caja en CSS px o null si no hay caja todavía.
 */
function ensureOverlaySize() {
  const rect = dom.canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  const dpr = window.devicePixelRatio || 1;
  const pxW = Math.round(rect.width * dpr);
  const pxH = Math.round(rect.height * dpr);
  if (dom.canvas.width !== pxW || dom.canvas.height !== pxH) {
    dom.canvas.width = pxW;
    dom.canvas.height = pxH;
  }
  return { w: rect.width, h: rect.height, dpr };
}

function drawOverlay(result) {
  const box = ensureOverlaySize();
  if (!box) return;
  const ctx = dom.canvas.getContext('2d');
  ctx.setTransform(box.dpr, 0, 0, box.dpr, 0, 0);
  ctx.clearRect(0, 0, box.w, box.h);

  const hands = result?.landmarks ?? [];
  const videoSize = { w: dom.video.videoWidth, h: dom.video.videoHeight };
  const elementSize = { w: box.w, h: box.h };

  for (let i = 0; i < hands.length; i++) {
    const pts = hands[i];
    if (!Array.isArray(pts) || pts.length < 21) continue; // nunca romper el loop
    const label = result.handedness?.[i]?.[0]?.categoryName ?? '';
    const isLeft = label === 'Left';
    const color = isLeft ? '#60a5fa' : '#22c55e';
    // Único punto de conversión de coordenadas (espejo + cover + escala):
    const mapped = pts.map((p) => mapLandmarkToOverlay(
      p, videoSize, elementSize, { mirrored: overlayMirrored, fit: 'cover' }
    ));

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(2, box.w / 500);
    ctx.lineCap = 'round';

    for (const pair of HAND_CONNECTIONS) {
      const [a, b] = pair.split(',').map(Number);
      const pa = mapped[a];
      const pb = mapped[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    }
    for (const p of mapped) {
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, Math.max(2, box.w / 300), 0, Math.PI * 2);
      ctx.fill();
    }
    if (label) {
      ctx.font = `${Math.max(14, box.w / 45)}px system-ui, sans-serif`;
      ctx.fillStyle = '#0f172a';
      ctx.fillText(label, mapped[0].x + 6, mapped[0].y + 6);
    }
  }
}

// ---------------------------------------------------------------------------
// Bucle principal (A PRUEBA DE MUERTE SILENCIOSA)
// ---------------------------------------------------------------------------
/**
 * El bucle se auto-reprograma SIEMPRE (try/finally): ninguna excepción puede
 * detener la detección de manos para siempre. Cualquier error queda en
 * consola + panel de diagnóstico, y el loop sigue corriendo.
 */
async function loop(now) {
  try {
    if (!tracker || dom.video.readyState < 2) {
      return;
    }
    const ts = performance.now();
    const result = await tracker.detect(dom.video, ts);

    drawOverlay(result);
    setHandState(eyesStatus(result.hands));

    frames += 1;
    if (ts - fpsWindowStart >= 1000) {
      fps = frames / ((ts - fpsWindowStart) / 1000);
      frames = 0;
      fpsWindowStart = ts;
      dom.diagFps.textContent = fps.toFixed(1);
    }

    updateDetectionStats();

    if (predictor?.isReady && result.hands > 0) {
      const features = handsToFeatures(result.left, result.right);
      if (features.length === FEATURES_SIZE) {
        // Solución A (F15): el modelo sigue recibiendo [1,126], pero el vector
        // es el frame de la ventana temporal más cercano a su mediana
        // (réplica en vivo de select_guide_frame). La ventana solo acumula
        // frames con mano; el gate de arriba ya descarta hand_count == 0.
        const selected = guideFrame.push(features);
        if (selected === null) {
          // Calentamiento (minFrames) o frame inválido: sin predicción aún.
          return;
        }
        try {
          const raw = predictor.predict(selected);       // probabilidades RAW
          const smoothed = smoother.step(raw.probabilities); // F10: capa independiente
          flow.onFrame(smoothed);      // F11: SOLO confirmadas -> historial + voz
          renderPrediction(smoothed, true);
          updateDiagPrediction(raw, smoothed, selected);
          if (DIAG_ENABLED) {
            diagFrameCount++;
            const diagLeftL2 = diagBlockL2(selected, 0, 63);
            const diagRightL2 = diagBlockL2(selected, 63, 126);
            const diagDistSI = diagDistL1(selected, DIAG_CENTROIDS.si);
            const diagDistNO = diagDistL1(selected, DIAG_CENTROIDS.no);
            const diagRawIdx = raw.probabilities.indexOf(Math.max(...raw.probabilities));
            const diagAboveThreshold = updateDiagThreshold(raw, diagRawIdx);
            const diagRecord = {
              frame: diagFrameCount,
              rawIndex: diagRawIdx,
              rawClass: raw.label,
              rawConfidence: raw.confidence,
              rawAboveThreshold: diagAboveThreshold,
              rawConsecutiveAboveThreshold: diagThresholdStats.currentRun,
              smoothedIndex: smoothed.label,
              smoothedClass: smoothed.label === null
                ? null : predictor.classes[smoothed.label],
              smoothedConfidence: smoothed.confidence,
              smoothedReason: smoothed.reason,
              selectedVector: Array.from(selected),
              windowSize: guideFrame.size,
              leftPresent: result.left !== null,
              rightPresent: result.right !== null,
              leftL2: diagLeftL2,
              rightL2: diagRightL2,
              l1ToSICentroid: diagDistSI,
              l1ToNOCentroid: diagDistNO,
              closerTo: diagDistSI < diagDistNO ? 'SI' : 'NO',
            };
            diagRecords.push(diagRecord);
            if (diagRecords.length > 30) diagRecords.shift();
            globalThis.__SIGNIA_LIVE_DIAG__.latest = diagRecord;
            if (diagAboveThreshold && diagThresholdStats.currentRun === 5) {
              console.log(`[THRESHOLD] ${raw.label} reached 5 consecutive RAW frames >= ${diagThresholdStats.threshold.toFixed(2)}`);
            }
            console.log(`[DIAG #${diagFrameCount}] RAW idx=${diagRawIdx} class=${raw.label} conf=${raw.confidence.toFixed(3)} | LeftL2=${diagLeftL2.toFixed(3)} RightL2=${diagRightL2.toFixed(3)} | distSI=${diagDistSI.toFixed(2)} distNO=${diagDistNO.toFixed(2)} | closer=${diagDistSI < diagDistNO ? 'SI' : 'NO'} | hands: L=${result.left !== null} R=${result.right !== null}`);
          }
        } catch (err) {
          console.error('Predicción:', err);
          renderPrediction(null, true);
        }
        return;
      }
    }
    // Sin manos (o modelo no listo): limpiar/pausar el smoother y separar
    // el historial (permite repetir la misma seña cuando la mano vuelva).
    guideFrame.reset();          // los frames viejos no deben sobrevivir
    smoother.noHand();
    resetDiagThreshold();
    flow.separate();
    updateDiagPrediction(null, null);
    renderPrediction(null, predictor?.isReady);
  } catch (err) {
    console.error('Detection loop (sigue corriendo):', err);
    dom.diagLoop.textContent = 'ERROR — ' + String(err?.message ?? err);
  } finally {
    rafId = requestAnimationFrame(loop);
  }
}

// ---------------------------------------------------------------------------
// Cámara
// ---------------------------------------------------------------------------
/**
 * Espera a que el video reciba frames REALES. "Cámara activa" solo significa
 * que getUserMedia devolvió un stream: un video verde o sin imagen significa
 * que no llegan frames decodificados (p. ej. cámara virtual/DroidCam).
 */
function waitForVideoFrames(video, timeoutMs) {
  return new Promise((resolve) => {
    const start = performance.now();
    const t0 = video.currentTime;
    const poll = () => {
      const frames = video.readyState >= 2 &&
        video.videoWidth > 0 && video.videoHeight > 0 &&
        !video.paused && video.currentTime > t0;
      if (frames || performance.now() - start > timeoutMs) {
        resolve(frames);
        return;
      }
      setTimeout(poll, 250);
    };
    poll();
  });
}

async function startCamera() {
  stopStream(stream);
  stream = null;
  hideBanner();
  dom.retryCamera.hidden = true;
  dom.stagePlaceholder.hidden = false;
  dom.stagePlaceholder.textContent = 'Inicializando cámara...';
  setChip('camera', SYSTEM_STATES.CAMERA_PERMISSION);
  const result = await openCamera();
  if (!result.ok) {
    setChip('camera', SYSTEM_STATES.ERROR_CAMERA, result.message);
    dom.diagCamera.textContent = `ERROR — ${result.error}`;
    showBanner(cameraErrorTitle(result.error), result.message);
    dom.stagePlaceholder.textContent =
      CAMERA_ERROR_MESSAGES[result.error] ?? result.message;
    dom.retryCamera.hidden = false;
    announce();
    return;
  }
  stream = result.stream;
  dom.video.srcObject = stream;
  try {
    await dom.video.play();
  } catch {
    /* autoplay bloqueado: el botón de retry lo reanuda */
  }
  const hasFrames = await waitForVideoFrames(dom.video, 5000);
  if (!hasFrames) {
    // Stream vivo pero SIN frames decodificados (video verde/negro).
    setChip('camera', SYSTEM_STATES.CAMERA_ACTIVE, 'Cámara activa');
    dom.diagCamera.textContent =
      `SIN FRAMES — readyState=${dom.video.readyState} ` +
      `${dom.video.videoWidth}x${dom.video.videoHeight}`;
    showBanner(
      'La cámara no está entregando imagen',
      'El stream está activo pero no llegan frames decodificados. ' +
      'Verifica DroidCam en la aplicación Cámara de Windows y, si el video ' +
      'se ve verde, desactiva la aceleración por hardware de Chrome ' +
      '(chrome://settings/system) y recarga.'
    );
    dom.stagePlaceholder.textContent =
      'Sin imagen: el stream no entrega frames. Revisa DroidCam y la ' +
      'aceleración por hardware de Chrome.';
    announce();
    return;
  }
  dom.stagePlaceholder.hidden = true;
  dom.diagCamera.textContent =
    `OK — ${dom.video.videoWidth}x${dom.video.videoHeight}`;
  setChip('camera', SYSTEM_STATES.CAMERA_ACTIVE);
  dom.retryCamera.hidden = true;
  announce();
  if (!rafId) rafId = requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// MediaPipe
// ---------------------------------------------------------------------------
async function initTracker() {
  dom.diagMediapipe.textContent = 'CARGANDO';
  try {
    tracker = await HandTracker.create();
    dom.diagMediapipe.textContent = 'READY';
  } catch (err) {
    // El error ORIGINAL (import, wasm, task o delegado GPU/CPU) siempre se
    // muestra en el panel de diagnóstico: no se esconde tras un mensaje genérico.
    dom.diagMediapipe.textContent = `ERROR — ${err.message}`;
    dom.diagLoop.textContent = 'STOPPED';
    console.error('MediaPipe:', err);
    showBanner(
      'Detección de manos no disponible',
      'El módulo de detección de manos no pudo iniciar. La cámara sigue activa, pero no se detectarán manos.'
    );
  }
}

// ---------------------------------------------------------------------------
// Modelo
// ---------------------------------------------------------------------------
function renderModelStatus() {
  const info = modelStatusInfo(predictor?.summary?.() ?? null);
  dom.modelStatus.dataset.kind = info.kind;
  dom.modelStatusTitle.textContent = info.title;
  dom.modelStatusDetail.textContent = info.detail ?? '';
  dom.modelStatusDetail.hidden = !info.detail;
  dom.retryModel.hidden = info.kind !== 'missing';
}

async function loadModel() {
  setChip('model', SYSTEM_STATES.MODEL_LOADING);
  renderModelStatus();
  try {
    predictor = makePredictor();
    await predictor.prepare({ modelUrl: MODEL_URL, metadataUrl: METADATA_URL });
    const s = predictor.summary();
    dom.diagInputShape.textContent = JSON.stringify(s.inputShape);
    dom.diagOutputShape.textContent = JSON.stringify(s.outputShape);
    dom.diagModel.textContent = 'READY';
    dom.diagModelVersion.textContent = s.modelVersion ?? '?';
    dom.diagModelInfo.textContent =
      `entrenado ${s.trainingDate ?? '?'} · ${s.classes.length} clases`;
    setChip('model', SYSTEM_STATES.MODEL_READY);
    renderModelStatus();
  } catch (err) {
    const message = err instanceof PredictError ? err.message : `Error inesperado: ${err.message}`;
    console.error('Modelo:', err);
    setChip('model', SYSTEM_STATES.ERROR_MODEL, UI_TEXT.NO_MODEL_TITLE);
    dom.diagModel.textContent = 'ERROR';
    dom.diagModelInfo.textContent = message;
    renderModelStatus();
  }
  announce();
}

// ---------------------------------------------------------------------------
// Historial (F11): solo señas CONFIRMADAS por F10.
// ---------------------------------------------------------------------------
function renderHistory() {
  const items = history.list();
  dom.historyEmpty.hidden = items.length > 0;
  dom.historyList.replaceChildren(
    ...items.map((entry) => {
      const li = document.createElement('li');
      li.className = 'history-item';
      li.textContent = historyLineText(entry, showConfidence);
      return li;
    })
  );
  if (dom.togglePersist.checked) {
    try {
      localStorage.setItem(LS_HISTORY, history.serialize());
    } catch {
      /* almacenamiento no disponible */
    }
  }
}

function setVoiceButton() {
  const on = speech.enabled;
  dom.btnToggleVoice.textContent = voiceButtonText(on);
  dom.btnToggleVoice.setAttribute('aria-pressed', String(on));
  dom.btnToggleVoice.disabled = !speech.supported;
  if (!speech.supported) {
    dom.btnToggleVoice.title = UI_TEXT.VOICE_UNSUPPORTED_TITLE;
  }
}

function restorePreferences() {
  try {
    const voicePref = localStorage.getItem(LS_VOICE);
    if (voicePref !== null) speech.setEnabled(voicePref === '1');
    const persist = localStorage.getItem(LS_HISTORY_PERSIST) === '1';
    dom.togglePersist.checked = persist;
    if (persist) {
      history.restore(localStorage.getItem(LS_HISTORY));
    }
  } catch {
    /* sin localStorage */
  }
}

// ---------------------------------------------------------------------------
// Eventos y arranque
// ---------------------------------------------------------------------------
function setDiagnosticsOpen(open) {
  dom.diagPanel.hidden = !open;
  dom.btnDiagnostics.setAttribute('aria-expanded', String(open));
}

function bindEvents() {
  dom.retryCamera.addEventListener('click', startCamera);

  dom.retryModel.addEventListener('click', () => {
    dom.retryModel.hidden = true;
    loadModel();
  });

  dom.btnDiagnostics.addEventListener('click', () => {
    setDiagnosticsOpen(dom.diagPanel.hidden);
  });

  dom.btnClearHistory.addEventListener('click', () => {
    history.clear();
    try {
      localStorage.removeItem(LS_HISTORY);
    } catch { /* sin localStorage */ }
    renderHistory();
  });

  dom.btnToggleVoice.addEventListener('click', () => {
    const now = speech.setEnabled(!speech.enabled);
    try {
      localStorage.setItem(LS_VOICE, now ? '1' : '0');
    } catch { /* sin localStorage */ }
    setVoiceButton();
  });

  dom.btnReplay.addEventListener('click', () => {
    const last = history.last();
    if (last && predictor && !predictor.isTestModel) {
      speech.speak(last.label);
    }
  });

  dom.toggleConfidence.addEventListener('change', (e) => {
    showConfidence = e.target.checked;
    renderHistory();
  });

  dom.togglePersist.addEventListener('change', (e) => {
    try {
      if (e.target.checked) {
        localStorage.setItem(LS_HISTORY_PERSIST, '1');
        localStorage.setItem(LS_HISTORY, history.serialize());
      } else {
        localStorage.removeItem(LS_HISTORY_PERSIST);
        localStorage.removeItem(LS_HISTORY);
      }
    } catch { /* sin localStorage */ }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key.toLowerCase() === 'd' && e.altKey) {
      setDiagnosticsOpen(dom.diagPanel.hidden);
    }
  });
}

function init() {
  bindEvents();
  if (typeof handsToFeatures === 'undefined' || typeof HAND_CONNECTIONS === 'undefined') {
    showBanner('Error interno', 'frontend/js/features.js no se cargó. Recarga la página.');
    return;
  }
  // F11: flujo confirmación -> historial -> voz. La voz se bloquea con un
  // modelo de prueba (nunca presentar las palabras sintéticas como LSC real).
  flow = createConfirmedFlow({
    history,
    onConfirm: (label) => {
      renderHistory();
      dom.historyLive.textContent = UI_TEXT.signConfirmedAnnouncement(label);
      pulsePrediction();
      if (predictor && !predictor.isTestModel) {
        speech.speak(label);
      }
    },
  });
  restorePreferences();
  setVoiceButton();
  renderHistory();
  dom.diagSmoothing.textContent =
    `α=${SMOOTHING_CONFIG.alpha} N=${SMOOTHING_CONFIG.stableFrames} ` +
    `umbral=${LIVE_MIN_CONFIDENCE} activo=${SMOOTHING_CONFIG.enabled}`;
  detectVideoMirror();
  updateDetectionStats();
  renderModelStatus();
  loadModel();
  initTracker();
  startCamera();
}

document.addEventListener('DOMContentLoaded', init);
