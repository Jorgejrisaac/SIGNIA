/**
 * Mapeo de landmarks MediaPipe (normalizados 0..1) a coordenadas del overlay.
 *
 * El canvas y el <video> comparten caja (inset:0, width/height 100%,
 * object-fit: cover). El video interno (p. ej. 1280x720) se muestra
 * RECORTADO dentro de la caja (cover) y ESPEJADO por CSS (scaleX(-1)).
 * MediaPipe procesa el frame SIN espejo y SIN recorte: hay que reproducir
 * ambas transformaciones matemáticamente para que el esqueleto quede
 * pegado a la mano visible.
 *
 * Estrategia de espejo (UNA sola, documentada): el canvas NUNCA lleva
 * transform CSS; el espejo se aplica aquí en X (x = elementWidth - x),
 * replicando exactamente el scaleX(-1) del video. NO hacer ambas cosas.
 *
 * Módulo PURO (sin DOM): testeable en Node.
 */

/**
 * Rectángulo visible del contenido del video dentro de la caja del elemento
 * (según object-fit).
 * @param {number} videoW resolución interna (videoWidth)
 * @param {number} videoH resolución interna (videoHeight)
 * @param {number} elW ancho de la caja del elemento (CSS px)
 * @param {number} elH alto de la caja del elemento (CSS px)
 * @param {'cover'|'contain'} fit
 * @returns {{scale:number, offsetX:number, offsetY:number,
 *            viewW:number, viewH:number}|null}
 */
export function videoViewport(videoW, videoH, elW, elH, fit = 'cover') {
  if (!(videoW > 0 && videoH > 0 && elW > 0 && elH > 0)) return null;
  const scale = fit === 'contain'
    ? Math.min(elW / videoW, elH / videoH)
    : Math.max(elW / videoW, elH / videoH);
  const viewW = videoW * scale;
  const viewH = videoH * scale;
  return {
    scale,
    offsetX: (elW - viewW) / 2,
    offsetY: (elH - viewH) / 2,
    viewW,
    viewH,
  };
}

/**
 * Convierte un landmark normalizado a coordenadas de la caja del elemento
 * (CSS px). El canvas se dibuja 1:1 con su caja (ver app.js: drawOverlay),
 * por lo que el punto resultante cae exactamente sobre la articulación.
 * @param {{x:number, y:number}} landmark coordenadas MediaPipe 0..1
 * @param {{w:number, h:number}} videoSize resolución interna del video
 * @param {{w:number, h:number}} elementSize caja visible (CSS px)
 * @param {{mirrored?:boolean, fit?:'cover'|'contain'}} [opts]
 * @returns {{x:number, y:number}|null}
 */
export function mapLandmarkToOverlay(landmark, videoSize, elementSize, opts = {}) {
  if (!landmark || !Number.isFinite(landmark.x) || !Number.isFinite(landmark.y)) {
    return null;
  }
  const { mirrored = true, fit = 'cover' } = opts;
  const vp = videoViewport(videoSize.w, videoSize.h, elementSize.w, elementSize.h, fit);
  if (!vp) return null;
  const rawX = vp.offsetX + landmark.x * vp.viewW;
  const rawY = vp.offsetY + landmark.y * vp.viewH;
  return {
    x: mirrored ? elementSize.w - rawX : rawX,
    y: rawY,
  };
}
