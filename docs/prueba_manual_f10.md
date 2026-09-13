# PRUEBA MANUAL — Fase 10 (estabilidad temporal)

Requiere hardware (webcam o celular con HTTPS). Sin webcam: prueba la
lógica en Node (`node --test "tests/frontend/**/*.test.mjs"`), que cubre
los mismos comportamientos de forma determinista.

## Configuración

```powershell
node frontend/server.js     # http://localhost:8000
```

Parámetros (NO óptimos, editables sin tocar lógica) en
`frontend/js/temporalSmoother.js` → `SMOOTHING_CONFIG`:
`alpha=0.35`, `minConfidence=0.70`, `stableFrames=5`.

## Checklist

- [ ] Predicción estable: con la mano quieta haciendo una seña, la
      etiqueta mostrada no parpadea entre clases.
- [ ] Frame aislado no provoca cambio: mover la mano 1 frame a otra
      posición no cambia la seña mostrada (debe tardar N=5 frames
      consecutivos en confirmar un cambio).
- [ ] Cambio real de seña: HOLA → GRACIAS cambia la etiqueta tras unos
      frames estables (el diagnóstico SMOOTHED muestra el cambio).
- [ ] Mano desaparece: quitar la mano → "Coloca tu mano frente a la
      cámara"; el panel de diagnóstico limpia RAW/SMOOTHED.
- [ ] Mano vuelve: al volver, arranca desde cero (no arrastra la seña
      anterior: debe tardar N frames en confirmar la nueva).
- [ ] Confianza insuficiente: con la mano en una posición ambigua (o
      modelo de prueba), si el máximo de probabilidad baja de 0.70 →
      "Confianza insuficiente" (nunca se inventa una clase).
- [ ] Modelo sintético correctamente identificado: con el modelo de
      2 clases, el badge "MODELO DE PRUEBA — NO REPRESENTA
      RECONOCIMIENTO LSC" es visible.
- [ ] FPS razonable: panel de diagnóstico, FPS ≈ fluido (el smoothing
      no añade tensores; solo arrays JS por frame).

## RAW vs SMOOTHED (panel de diagnóstico)

- [ ] RAW muestra la predicción directa del modelo (p. ej. HOLA — 0.83).
- [ ] SMOOTHED muestra el resultado tras estabilidad (p. ej. HOLA — 0.79).
- [ ] La interfaz normal muestra SOLO el resultado final (nunca RAW).

## Verificación de rendimiento

- [ ] Sin crecimiento de memoria: DevTools → Memory (o Task Manager)
      durante ~1 min con la mano activa: la memoria se mantiene plana
      (el modelo se carga 1 sola vez; sin fugas de tensores).