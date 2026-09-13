/**
 * Cámara SIGNIA — envoltura de navigator.mediaDevices.getUserMedia().
 *
 * Módulo sin DOM en el import: toda interacción con el DOM la hace quien
 * lo consume (app.js). Maneja cada fallo con un estado explícito:
 *   - permiso denegado        (NotAllowedError)
 *   - cámara inexistente      (NotFoundError)
 *   - cámara ocupada          (NotReadableError)
 *   - restricciones inviables (OverconstrainedError)
 *   - error inesperado        (cualquier otro)
 *
 * No solicita audio: video únicamente, resolución razonable.
 */

export const CAMERA_ERRORS = Object.freeze({
  PERMISSION_DENIED: 'PERMISSION_DENIED',
  NOT_FOUND: 'NOT_FOUND',
  BUSY: 'BUSY',
  OVERCONSTRAINED: 'OVERCONSTRAINED',
  INSECURE_CONTEXT: 'INSECURE_CONTEXT',
  UNKNOWN: 'UNKNOWN',
});

export const CAMERA_ERROR_MESSAGES = Object.freeze({
  [CAMERA_ERRORS.PERMISSION_DENIED]:
    'Permiso de cámara denegado. Permite el acceso en la barra del navegador y vuelve a intentarlo.',
  [CAMERA_ERRORS.NOT_FOUND]:
    'No se encontró ninguna cámara en este dispositivo.',
  [CAMERA_ERRORS.BUSY]:
    'La cámara está siendo usada por otra aplicación. Ciérrala e intenta de nuevo.',
  [CAMERA_ERRORS.OVERCONSTRAINED]:
    'La cámara no soporta la configuración solicitada.',
  [CAMERA_ERRORS.INSECURE_CONTEXT]:
    'En el teléfono, la cámara del navegador requiere HTTPS. Usa una URL HTTPS; http://localhost solo funciona en el mismo dispositivo.',
  [CAMERA_ERRORS.UNKNOWN]:
    'Error inesperado al abrir la cámara. Revisa el panel de diagnóstico.',
});

const DEFAULT_CONSTRAINTS = {
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    facingMode: 'user', // cámara frontal (selfie): correcto para señas
  },
  audio: false, // sin audio: no se piden permisos innecesarios
};

/**
 * @returns {Promise<{ok: true, stream: MediaStream} |
 *                   {ok: false, error: string, message: string}>}
 */
export async function openCamera(constraints = DEFAULT_CONSTRAINTS) {
  if (typeof window !== 'undefined' && !window.isSecureContext &&
      !isLocalhost(window.location?.hostname)) {
    return {
      ok: false,
      error: CAMERA_ERRORS.INSECURE_CONTEXT,
      message: CAMERA_ERROR_MESSAGES[CAMERA_ERRORS.INSECURE_CONTEXT],
    };
  }
  if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
    return {
      ok: false,
      error: CAMERA_ERRORS.NOT_FOUND,
      message: CAMERA_ERROR_MESSAGES[CAMERA_ERRORS.NOT_FOUND],
    };
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    return { ok: true, stream };
  } catch (err) {
    return { ok: false, error: classifyCameraError(err), message: cameraErrorMessage(err) };
  }
}

function isLocalhost(hostname) {
  return hostname === 'localhost' || hostname === '127.0.0.1' ||
    hostname === '[::1]';
}

export function classifyCameraError(err) {
  const name = err?.name ?? '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError': // Safari
      return CAMERA_ERRORS.PERMISSION_DENIED;
    case 'NotFoundError':
    case 'DevicesNotFoundError': // Safari
      return CAMERA_ERRORS.NOT_FOUND;
    case 'NotReadableError':
    case 'TrackStartError': // Safari
      return CAMERA_ERRORS.BUSY;
    case 'OverconstrainedError':
      return CAMERA_ERRORS.OVERCONSTRAINED;
    default:
      return CAMERA_ERRORS.UNKNOWN;
  }
}

export function cameraErrorMessage(err) {
  return CAMERA_ERROR_MESSAGES[classifyCameraError(err)];
}

/** Libera todas las pistas del stream (apaga la luz de la cámara). */
export function stopStream(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
}
