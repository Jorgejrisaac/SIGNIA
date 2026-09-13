/**
 * Predictor SIGNIA — carga el modelo TF.js exportado en F8 y predice.
 *
 * NO entrena ni crea modelos desde JavaScript: únicamente carga
 * frontend/models/signia/model.json (+ shards) y metadata.json.
 *
 * Antes de declarar el modelo READY valida (contra metadata.json):
 *   - input shape (None, 126)
 *   - output shape (None, n_clases)
 *   - n_classes == metadata.n_classes == metadata.classes.length
 * Si algo no coincide -> ERROR_MODEL con mensaje explicativo.
 *
 * Dependencias inyectables para tests en Node (sin DOM ni red):
 *   makePredictor({ tf, loadLayersModel, fetchJson })
 * En el navegador usa los valores por defecto (tf global + fetch).
 */

import { isTestModel, TEST_MODEL_WARNING } from './states.js';

export class PredictError extends Error {}

function browserFetchJson(url) {
  return fetch(url).then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status} para ${url}`);
    return res.json();
  });
}

export function makePredictor(deps = {}) {
  return new Predictor({
    tf: deps.tf ?? globalThis.tf,
    loadLayersModel: deps.loadLayersModel ?? ((url) => deps.tf?.loadLayersModel?.(url) ?? globalThis.tf.loadLayersModel(url)),
    fetchJson: deps.fetchJson ?? browserFetchJson,
  });
}

export class Predictor {
  constructor({ tf, loadLayersModel, fetchJson }) {
    if (!tf) throw new PredictError('TensorFlow.js no está disponible (tf.min.js)');
    this.tf = tf;
    this.loadLayersModel = loadLayersModel;
    this.fetchJson = fetchJson;
    this.model = null;
    this.metadata = null;
    this.classes = [];
    this.inputShape = null;
    this.outputShape = null;
    this.isTestModel = false;
    this._preparePromise = null;
  }

  get isReady() {
    return this.model !== null;
  }

  /**
   * Carga y valida el modelo. Se puede llamar varias veces sin fugas
   * (descarta el modelo previo). Cachea la promesa para no cargar dos
   * veces el mismo modelo en paralelo.
   */
  prepare({ modelUrl, metadataUrl }) {
    if (!this._preparePromise) {
      this._preparePromise = this._prepare(modelUrl, metadataUrl)
        .catch((err) => {
          this._preparePromise = null; // reintento permitido
          this.model = null;
          throw err;
        });
    }
    return this._preparePromise;
  }

  async _prepare(modelUrl, metadataUrl) {
    let metadata;
    try {
      metadata = await this.fetchJson(metadataUrl);
    } catch (err) {
      throw new PredictError(
        `Modelo no cargado: falta metadata.json (${err.message}). ` +
        'Exporta el modelo con python ai/training/export_tfjs.py'
      );
    }

    const nFeatures = 126;
    const inputShape = metadata.input_shape;
    const outputShape = metadata.output_shape;
    const classes = Array.isArray(metadata.classes) ? metadata.classes : [];
    const nClasses = metadata.n_classes;

    if (!inputShape || inputShape[1] !== nFeatures) {
      throw new PredictError(
        `ERROR_MODEL: input_shape esperado (null, ${nFeatures}), recibido ${JSON.stringify(inputShape)}`
      );
    }
    if (!outputShape || outputShape[1] !== nClasses) {
      throw new PredictError(
        `ERROR_MODEL: output_shape (${JSON.stringify(outputShape)}) no coincide con n_classes (${nClasses})`
      );
    }
    if (classes.length !== nClasses) {
      throw new PredictError(
        `ERROR_MODEL: metadata.classes (${classes.length}) != n_classes (${nClasses}). ` +
        'El orden de clases es una sola fuente (metadata.json).'
      );
    }

    let model;
    try {
      model = await this.loadLayersModel(modelUrl);
    } catch (err) {
      throw new PredictError(
        `Modelo no cargado: no se pudo leer ${modelUrl} (${err.message}). ` +
        '¿Exportaste el modelo con export_tfjs.py?'
      );
    }

    const modelIn = model.inputs[0].shape;
    const modelOut = model.output.shape;
    if (modelIn[1] !== nFeatures || modelOut[1] !== nClasses) {
      model.dispose?.();
      throw new PredictError(
        `ERROR_MODEL: el modelo TF.js tiene shapes in=${JSON.stringify(modelIn)} ` +
        `out=${JSON.stringify(modelOut)}; metadata espera (${nFeatures}, ${nClasses}). ` +
        'Reexporta el modelo con el pipeline SIGNIA.'
      );
    }

    if (this.model) this.model.dispose?.();
    this.model = model;
    this.metadata = metadata;
    this.classes = classes;
    this.inputShape = modelIn;
    this.outputShape = modelOut;
    this.isTestModel = isTestModel(metadata);
    return this;
  }

  /**
   * Predice sobre el vector normalizado de features (126).
   * tf.tidy: los tensores intermedios se liberan automáticamente.
   * @returns {{label: string, confidence: number, probabilities: number[]}}
   */
  predict(features) {
    if (!this.isReady) {
      throw new PredictError('Modelo no cargado: no se puede predecir.');
    }
    const probs = this.tf.tidy(() => {
      const x = this.tf.tensor2d([Array.from(features)], [1, 126]);
      const out = this.model.predict(x);
      return Array.from(out.dataSync());
    });
    let argmax = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[argmax]) argmax = i;
    }
    return {
      label: this.classes[argmax],
      confidence: probs[argmax],
      probabilities: probs,
    };
  }

  summary() {
    if (!this.isReady) return { ready: false };
    return {
      ready: true,
      inputShape: this.inputShape,
      outputShape: this.outputShape,
      classes: this.classes,
      isTestModel: this.isTestModel,
      testModelWarning: this.isTestModel ? TEST_MODEL_WARNING : null,
      modelVersion: this.metadata.model_version,
      trainingDate: this.metadata.training_date,
    };
  }
}