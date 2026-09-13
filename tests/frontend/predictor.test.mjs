/**
 * Predictor SIGNIA — carga, validación de shapes/metadata y predicción (F9).
 * Usa un stub de tf: valida la LÓGICA del predictor (estados, errores,
 * formato de salida), no la aritmética (eso lo cubre la paridad F8).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { makePredictor, PredictError } from '../../frontend/js/predictor.js';
import { TEST_MODEL_WARNING } from '../../frontend/js/states.js';

const CLASSES_10 = Array.from({ length: 10 }, (_, i) => `CLASE_${i}`);
const METADATA_OK = {
  model_name: 'SIGNIA',
  model_version: 1,
  input_shape: [null, 126],
  output_shape: [null, 10],
  n_features: 126,
  n_classes: 10,
  classes: CLASSES_10,
};

function tfStub(model = null) {
  return {
    tidy(fn) { return fn(); },
    tensor2d(data, shape) {
      assert.equal(shape[1], 126, 'el tensor debe ser de 126 features');
      return { data: Array.from(data[0]) };
    },
    model,
  };
}

function makeModel(probs) {
  return {
    inputs: [{ shape: [null, 126] }],
    output: { shape: [null, probs.length] },
    predict() {
      return { dataSync: () => Float32Array.from(probs) };
    },
    dispose() {},
  };
}

function build({ metadata = METADATA_OK, probs = null, loadFails = false } = {}) {
  const model = probs ? makeModel(probs) : null;
  return makePredictor({
    tf: tfStub(),
    fetchJson: async () => {
      if (metadata === null) throw new Error('HTTP 404');
      return metadata;
    },
    loadLayersModel: async () => {
      if (loadFails) throw new Error('file not found');
      return model;
    },
  });
}

test('carga correcta con metadata válida -> READY', async () => {
  const p = build({ probs: [0.9, 0.05, ...Array(8).fill(0.005)] });
  await p.prepare({ modelUrl: 'x', metadataUrl: 'y' });
  assert.equal(p.isReady, true);
  assert.deepEqual(p.inputShape, [null, 126]);
  assert.deepEqual(p.outputShape, [null, 10]);
  assert.equal(p.isTestModel, false);
  assert.equal(p.classes.length, 10);
});

test('metadata ausente -> ERROR_MODEL claro', async () => {
  const p = build({ metadata: null });
  await assert.rejects(
    () => p.prepare({ modelUrl: 'x', metadataUrl: 'y' }),
    (err) => {
      assert.ok(err instanceof PredictError);
      assert.match(err.message, /Modelo no cargado/);
      assert.match(err.message, /falta metadata\.json/);
      return true;
    }
  );
  assert.equal(p.isReady, false);
});

test('input_shape incorrecto -> ERROR_MODEL', async () => {
  const p = build({ metadata: { ...METADATA_OK, input_shape: [null, 64] } });
  await assert.rejects(
    () => p.prepare({ modelUrl: 'x', metadataUrl: 'y' }),
    (err) => {
      assert.ok(err instanceof PredictError);
      assert.match(err.message, /input_shape esperado/);
      return true;
    }
  );
});

test('classes != n_classes -> ERROR_MODEL (orden de clases único)', async () => {
  const p = build({ metadata: { ...METADATA_OK, classes: CLASSES_10.slice(0, 3) } });
  await assert.rejects(
    () => p.prepare({ modelUrl: 'x', metadataUrl: 'y' }),
    (err) => {
      assert.ok(err instanceof PredictError);
      assert.match(err.message, /!= n_classes|no coincide/);
      return true;
    }
  );
});

test('shapes del modelo TF.js no coinciden con metadata -> ERROR_MODEL', async () => {
  const p = makePredictor({
    tf: tfStub(),
    fetchJson: async () => METADATA_OK,
    loadLayersModel: async () => ({
      inputs: [{ shape: [null, 126] }],
      output: { shape: [null, 3] }, // metadata dice 10
      predict() {},
      dispose() {},
    }),
  });
  await assert.rejects(
    () => p.prepare({ modelUrl: 'x', metadataUrl: 'y' }),
    (err) => {
      assert.ok(err instanceof PredictError);
      assert.match(err.message, /shapes|no coincide/);
      return true;
    }
  );
});

test('modelo no cargable -> ERROR_MODEL', async () => {
  const p = build({ loadFails: true });
  await assert.rejects(
    () => p.prepare({ modelUrl: 'x', metadataUrl: 'y' }),
    (err) => {
      assert.ok(err instanceof PredictError);
      assert.match(err.message, /no se pudo leer/);
      return true;
    }
  );
});

test('predict sin modelo -> error claro (nunca inventa)', () => {
  const p = build();
  assert.throws(() => p.predict(new Float32Array(126)), PredictError);
  try {
    p.predict(new Float32Array(126));
    assert.fail('debió lanzar');
  } catch (err) {
    assert.match(err.message, /Modelo no cargado/);
  }
});

test('predict: label, confidence y probabilities correctos', async () => {
  const p = build({ probs: [0.02, 0.9, ...Array(8).fill(0.01)] });
  await p.prepare({ modelUrl: 'x', metadataUrl: 'y' });
  const out = p.predict(new Float32Array(126));
  assert.equal(out.label, CLASSES_10[1]);
  assert.ok(Math.abs(out.confidence - 0.9) < 1e-6);
  assert.equal(out.probabilities.length, 10);
  const sum = out.probabilities.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1) < 1e-4, 'probabilidades suman 1 (softmax)');
});

test('modelo de prueba (2 clases) se marca y muestra advertencia', async () => {
  const meta = {
    ...METADATA_OK,
    n_classes: 2,
    output_shape: [null, 2],
    classes: ['HOLA', 'ADIOS'],
  };
  const p = build({ metadata: meta, probs: [0.6, 0.4] });
  await p.prepare({ modelUrl: 'x', metadataUrl: 'y' });
  assert.equal(p.isTestModel, true);
  const s = p.summary();
  assert.equal(s.testModelWarning, TEST_MODEL_WARNING);
});