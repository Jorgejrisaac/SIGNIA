/**
 * Estados del sistema SIGNIA — módulo PURO (sin DOM), testeable en Node.
 *
 * Cada estado se refleja en la interfaz con texto + icono, no solo color.
 * F9: estados de cámara, modelo y manos. F10+ añadirá más si hace falta.
 */

export const SYSTEM_STATES = Object.freeze({
  INITIALIZING: 'INITIALIZING',
  CAMERA_PERMISSION: 'CAMERA_PERMISSION',
  CAMERA_ACTIVE: 'CAMERA_ACTIVE',
  MODEL_LOADING: 'MODEL_LOADING',
  MODEL_READY: 'MODEL_READY',
  NO_HAND: 'NO_HAND',
  ONE_HAND: 'ONE_HAND',
  TWO_HANDS: 'TWO_HANDS',
  ERROR_CAMERA: 'ERROR_CAMERA',
  ERROR_MODEL: 'ERROR_MODEL',
});

/** Mensajes legibles por estado (para la interfaz). */
export const STATE_MESSAGES = Object.freeze({
  [SYSTEM_STATES.INITIALIZING]: 'Inicializando SIGNIA...',
  [SYSTEM_STATES.CAMERA_PERMISSION]: 'Esperando permiso de cámara',
  [SYSTEM_STATES.CAMERA_ACTIVE]: 'Cámara activa',
  [SYSTEM_STATES.MODEL_LOADING]: 'Cargando modelo...',
  [SYSTEM_STATES.MODEL_READY]: 'Modelo listo',
  [SYSTEM_STATES.NO_HAND]: 'Coloca tu mano frente a la cámara',
  [SYSTEM_STATES.ONE_HAND]: 'Mano detectada',
  [SYSTEM_STATES.TWO_HANDS]: 'Dos manos detectadas',
  [SYSTEM_STATES.ERROR_CAMERA]: 'Error de cámara',
  [SYSTEM_STATES.ERROR_MODEL]: 'Modelo no cargado',
});

/** Iconos simples (no dependen del color para transmitir el estado). */
export const STATE_ICONS = Object.freeze({
  [SYSTEM_STATES.INITIALIZING]: '…',
  [SYSTEM_STATES.CAMERA_PERMISSION]: '📷',
  [SYSTEM_STATES.CAMERA_ACTIVE]: '●',
  [SYSTEM_STATES.MODEL_LOADING]: '…',
  [SYSTEM_STATES.MODEL_READY]: '●',
  [SYSTEM_STATES.NO_HAND]: '✋',
  [SYSTEM_STATES.ONE_HAND]: '✋',
  [SYSTEM_STATES.TWO_HANDS]: '🤚',
  [SYSTEM_STATES.ERROR_CAMERA]: '✖',
  [SYSTEM_STATES.ERROR_MODEL]: '✖',
});

/**
 * Estados que representan un fallo: la interfaz debe mostrarlos con
 * clase de error y mensaje explicativo.
 */
export const ERROR_STATES = Object.freeze([
  SYSTEM_STATES.ERROR_CAMERA,
  SYSTEM_STATES.ERROR_MODEL,
]);

/** Número de clases esperado del modelo real (config.py: CLASES). */
export const SIGNIA_EXPECTED_CLASSES = 10;

/**
 * Un modelo es "de prueba" (MODELO DE PRUEBA — NO ES LSC) si no tiene la
 * cantidad de clases del modelo real o si su metadata lo declara.
 * La producción sustituye metadata.json/model.json sin tocar el código.
 */
export function isTestModel(metadata) {
  if (!metadata) return false;
  if (metadata.is_test_model === true) return true;
  if (typeof metadata.n_classes === 'number') {
    return metadata.n_classes !== SIGNIA_EXPECTED_CLASSES;
  }
  return false;
}

/** Mensaje visible cuando el modelo presente es un modelo de prueba. */
export const TEST_MODEL_WARNING = 'MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC';