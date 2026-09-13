import { test } from 'node:test';
import assert from 'node:assert/strict';
import { videoViewport, mapLandmarkToOverlay } from '../../frontend/js/overlay.js';

const close = (got, want, tol = 1e-6) => assert.ok(
  Math.abs(got - want) <= tol,
  `esperado ${want}, obtenido ${got}`
);

test('videoViewport: cover sin recorte cuando la caja tiene el mismo aspect', () => {
  const vp = videoViewport(1280, 720, 640, 360, 'cover');
  assert.deepEqual(vp, { scale: 0.5, offsetX: 0, offsetY: 0, viewW: 640, viewH: 360 });
});

test('videoViewport: cover en caja 4:3 recorta los lados (crop horizontal)', () => {
  const vp = videoViewport(1280, 720, 800, 600, 'cover');
  close(vp.scale, 600 / 720);
  close(vp.viewH, 600);
  close(vp.viewW, 1280 * vp.scale);
  close(vp.offsetX, (800 - vp.viewW) / 2);
  close(vp.offsetY, 0);
});

test('videoViewport: cover en caja 3:4 vertical recorta los lados', () => {
  const vp = videoViewport(1280, 720, 360, 480, 'cover');
  close(vp.scale, 480 / 720);
  close(vp.offsetX, (360 - vp.viewW) / 2);
  close(vp.offsetY, 0);
});

test('videoViewport: contain genera letterboxing vertical', () => {
  const vp = videoViewport(1280, 720, 800, 600, 'contain');
  close(vp.scale, 800 / 1280);
  close(vp.viewW, 800);
  close(vp.viewH, 450);
  close(vp.offsetX, 0);
  close(vp.offsetY, 75);
});

test('videoViewport: tamaños inválidos devuelven null', () => {
  assert.equal(videoViewport(0, 720, 640, 360), null);
  assert.equal(videoViewport(1280, 0, 640, 360), null);
  assert.equal(videoViewport(1280, 720, 0, 360), null);
});

test('1280x720 -> canvas 640x360 (sin espejo): mapeo 1:1 y límites', () => {
  const v = { w: 1280, h: 720 };
  const e = { w: 640, h: 360 };
  let p = mapLandmarkToOverlay({ x: 0, y: 0 }, v, e, { mirrored: false });
  close(p.x, 0); close(p.y, 0);
  p = mapLandmarkToOverlay({ x: 0.5, y: 0.5 }, v, e, { mirrored: false });
  close(p.x, 320); close(p.y, 180);
  p = mapLandmarkToOverlay({ x: 1, y: 1 }, v, e, { mirrored: false });
  close(p.x, 640); close(p.y, 360);
});

test('1280x720 -> canvas 800x600 (cover, crop lateral): centro y recorte', () => {
  const v = { w: 1280, h: 720 };
  const e = { w: 800, h: 600 };
  const scale = 600 / 720;
  const viewW = 1280 * scale;
  const off = (800 - viewW) / 2;
  let p = mapLandmarkToOverlay({ x: 0.5, y: 0.5 }, v, e, { mirrored: false });
  close(p.x, 400); close(p.y, 300);
  p = mapLandmarkToOverlay({ x: 0, y: 0 }, v, e, { mirrored: false });
  close(p.x, off); close(p.y, 0);
  p = mapLandmarkToOverlay({ x: 1, y: 1 }, v, e, { mirrored: false });
  close(p.x, off + viewW); close(p.y, 600);
});

test('1280x720 -> contenedor 360x480 (cover, móvil vertical): crop fuerte', () => {
  const v = { w: 1280, h: 720 };
  const e = { w: 360, h: 480 };
  let p = mapLandmarkToOverlay({ x: 0.5, y: 0.5 }, v, e, { mirrored: false });
  close(p.x, 180); close(p.y, 240);
  p = mapLandmarkToOverlay({ x: 0.5, y: 1 }, v, e, { mirrored: false });
  close(p.y, 480);
  close(p.x, 180);
});

test('espejo activado: x se invierte alrededor del centro (y no cambia)', () => {
  const v = { w: 1280, h: 720 };
  const e = { w: 800, h: 600 };
  const scale = 600 / 720;
  const off = (800 - 1280 * scale) / 2;
  const rawX0 = off; // x=0 sin espejo
  let p = mapLandmarkToOverlay({ x: 0, y: 0.25 }, v, e, { mirrored: true });
  close(p.x, 800 - rawX0); close(p.y, 0.25 * 600);
  p = mapLandmarkToOverlay({ x: 0.5, y: 0.5 }, v, e, { mirrored: true });
  close(p.x, 400); close(p.y, 300); // centro invariante
  p = mapLandmarkToOverlay({ x: 1, y: 0.75 }, v, e, { mirrored: true });
  close(p.x, 800 - (off + 1280 * scale)); close(p.y, 0.75 * 600);
});

test('límites x=0/0.5/1 e y=0/0.5/1 con espejo en 640x360', () => {
  const v = { w: 1280, h: 720 };
  const e = { w: 640, h: 360 };
  const xs = { 0: 640, 0.5: 320, 1: 0 };
  for (const [lx, wantX] of Object.entries(xs)) {
    const p = mapLandmarkToOverlay({ x: Number(lx), y: 0.5 }, v, e, { mirrored: true });
    close(p.x, wantX); close(p.y, 180);
  }
  const ys = { 0: 0, 0.5: 180, 1: 360 };
  for (const [ly, wantY] of Object.entries(ys)) {
    const p = mapLandmarkToOverlay({ x: 0.5, y: Number(ly) }, v, e, { mirrored: true });
    close(p.x, 320); close(p.y, wantY);
  }
});

test('landmark inválido o caja inválida devuelven null sin romper', () => {
  assert.equal(mapLandmarkToOverlay({ x: 0.5, y: 0.5 }, { w: 0, h: 720 }, { w: 100, h: 100 }), null);
  assert.equal(mapLandmarkToOverlay(null, { w: 1280, h: 720 }, { w: 100, h: 100 }), null);
});
