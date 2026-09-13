/**
 * Voz (F11) — SpeechController con SpeechSynthesis inyectado (sin navegador).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  SpeechController,
  SPEECH_CONFIG,
  labelToSpeechText,
  speechSupported,
} from '../../frontend/js/speech.js';

function fakeSynthesis(voices = []) {
  const listeners = {};
  const spoken = [];
  return {
    voices,
    spoken,
    getVoices() { return this.voices; },
    speak(u) { this.spoken.push(u); return true; },
    cancel() {
      this.spoken.length = 0;
      this.cancelCount = (this.cancelCount ?? 0) + 1;
    },
    addEventListener(ev, fn) {
      (listeners[ev] ??= []).push(fn);
    },
    listeners,
    onvoiceschanged: null,
  };
}

function fakeUtterance() {
  return class FakeUtterance {
    constructor(text) {
      this.text = text;
      this.lang = null;
      this.voice = null;
      this.rate = 1;
      this.pitch = 1;
    }
  };
}

function make({ voices = [], config = {}, synthesis = null } = {}) {
  const ss = synthesis ?? fakeSynthesis(voices);
  return new SpeechController({ synthesis: ss, Utterance: fakeUtterance(), config });
}

const VOICE = (lang, name = `Voz ${lang}`) => ({ lang, name });

test('voz desactivada: no habla', () => {
  const c = make({ config: { enabled: false } });
  assert.equal(c.speak('HOLA'), false);
  assert.equal(c.synthesis.spoken.length, 0);
});

test('setEnabled(false) desactiva y cancela', () => {
  const c = make();
  c.speak('HOLA');
  assert.equal(c.setEnabled(false), false);
  assert.ok(c.synthesis.cancelCount >= 1);
  assert.equal(c.speak('GRACIAS'), false);
});

test('speak recibe etiqueta interna y produce texto mapeado', () => {
  const c = make();
  assert.equal(c.speak('HOLA'), true);
  assert.equal(c.synthesis.spoken.length, 1);
  assert.equal(c.synthesis.spoken[0].text, 'Hola');
  assert.equal(c.synthesis.spoken[0].lang, 'es-CO');
});

test('labelToSpeechText: mapeo de acentos/formato', () => {
  assert.equal(labelToSpeechText('ADIOS'), 'Adiós');
  assert.equal(labelToSpeechText('GRACIAS'), 'Gracias');
  assert.equal(labelToSpeechText('POR_FAVOR'), 'Por favor');
  assert.equal(labelToSpeechText('SI'), 'Sí');
  assert.equal(labelToSpeechText('NO'), 'No');
  assert.equal(labelToSpeechText('HOLA'), 'Hola');
});

test('labelToSpeechText: etiqueta desconocida no rompe', () => {
  assert.equal(labelToSpeechText('NUEVA_SECCION'), 'Nueva seccion');
  assert.equal(labelToSpeechText(null), '');
  assert.equal(labelToSpeechText(''), '');
});

test('selección de voz: es-CO preferida', () => {
  const c = make({ voices: [VOICE('es-ES'), VOICE('es-CO'), VOICE('en-US')] });
  c.speak('HOLA');
  assert.equal(c.synthesis.spoken[0].voice.lang, 'es-CO');
});

test('fallback: es-ES cuando no hay es-CO', () => {
  const c = make({ voices: [VOICE('es-ES'), VOICE('en-US'), VOICE('fr-FR')] });
  c.speak('HOLA');
  assert.equal(c.synthesis.spoken[0].voice.lang, 'es-ES');
});

test('fallback: cualquier es-*', () => {
  const c = make({ voices: [VOICE('es-MX'), VOICE('en-US')] });
  c.speak('HOLA');
  assert.equal(c.synthesis.spoken[0].voice.lang, 'es-MX');
});

test('sin voces en español: lang por defecto es-CO y sin fallo', () => {
  const c = make({ voices: [VOICE('en-US'), VOICE('de-DE')] });
  assert.equal(c.speak('HOLA'), true);
  assert.equal(c.synthesis.spoken[0].voice, null);
  assert.equal(c.synthesis.spoken[0].lang, 'es-CO');
});

test('ausencia de SpeechSynthesis: supported=false y speak=false', () => {
  const c = new SpeechController({ synthesis: undefined, Utterance: undefined });
  assert.equal(c.supported, false);
  assert.equal(c.enabled, false);
  assert.equal(c.speak('HOLA'), false);
  assert.doesNotThrow(() => c.cancel());
});

test('speechSupported rechaza objetos incompletos', () => {
  assert.equal(speechSupported(null, null), false);
  assert.equal(speechSupported({ speak() {} }, fakeUtterance()), false);
});

test('manejo de error: speak() que lanza no rompe', () => {
  const ss = fakeSynthesis();
  ss.speak = () => { throw new Error('not allowed'); };
  const c = new SpeechController({ synthesis: ss, Utterance: fakeUtterance() });
  assert.equal(c.speak('HOLA'), false);
  assert.equal(c.enabled, true, 'el controlador sigue operativo');
});

test('no acumulación de cola: replaceOnNew cancela la anterior', () => {
  const c = make();
  c.speak('HOLA');
  assert.equal(c.synthesis.spoken.length, 1);
  c.speak('GRACIAS'); // cancel() limpia + speak() agrega -> una sola en cola
  assert.equal(c.synthesis.spoken.length, 1);
  assert.equal(c.synthesis.spoken[0].text, 'Gracias');
});

test('replaceOnNew=false: se encolan (política alternativa)', () => {
  const c = make({ config: { replaceOnNew: false } });
  c.speak('HOLA');
  c.speak('GRACIAS');
  assert.equal(c.synthesis.spoken.length, 2);
  assert.equal(c.synthesis.spoken[0].text, 'Hola');
  assert.equal(c.synthesis.spoken[1].text, 'Gracias');
});

test('voiceschanged: voces tardías se recargan en el siguiente speak', () => {
  const ss = fakeSynthesis([]); // inicialmente vacío
  const c = new SpeechController({ synthesis: ss, Utterance: fakeUtterance() });
  c.speak('HOLA');
  assert.equal(ss.spoken[0].voice, null);
  // llegan las voces y se dispara el evento
  ss.voices.push(VOICE('es-CO'));
  ss.listeners.voiceschanged?.forEach((fn) => fn());
  c.speak('GRACIAS');
  // política replaceOnNew: la cola conserva SOLO la última locución
  assert.equal(ss.spoken.length, 1);
  assert.equal(ss.spoken[0].voice.lang, 'es-CO');
});

test('SPEECH_CONFIG: valores iniciales documentados', () => {
  assert.equal(SPEECH_CONFIG.enabled, true);
  assert.equal(SPEECH_CONFIG.replaceOnNew, true);
  assert.deepEqual(SPEECH_CONFIG.languagePriority, ['es-CO', 'es-ES', 'es-*']);
});