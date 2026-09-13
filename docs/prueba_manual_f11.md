# Prueba manual F11 — Historial + voz (SpeechSynthesis)

Prerrequisito: F9 (cámara + landmarks + modelo TF.js) y F10 (smoothing temporal)
funcionando. Servidor: `node frontend/server.js` → http://localhost:8000.
Webcam + micrófono/salida de audio del sistema.

## Checklist

- [ ] 1. Sin modelo exportado: la app muestra "Modelo no cargado" y el panel
      TRADUCCIÓN aparece vacío con el cartel "Aún no hay señas confirmadas".
- [ ] 2. Con modelo: al confirmar una seña (F10, ~5 frames estables con
      confianza ≥ 0.70), aparece UNA entrada en el historial con la etiqueta y
      la hora.
- [ ] 3. Mantener la misma seña durante varios segundos: NO se agregan
      entradas duplicadas consecutivas (una sola por confirmación).
- [ ] 4. Cambiar de seña: se agrega la nueva confirmación a continuación.
- [ ] 5. Quitar la mano de cuadro y volver a hacer la MISMA seña: se registra
      de nuevo (mano ausente = separación, permite repetir para "HOY, GRACIAS").
- [ ] 6. Voz activa (botón 🔊): se escucha la palabra en español UNA sola vez
      por confirmación, inmediatamente al confirmar.
- [ ] 7. Cambio rápido de seña: la nueva locución reemplaza a la anterior
      (no se acumulan en cola).
- [ ] 8. Botón 🔊/🔇 alterna y muestra estado con `aria-pressed`; la
      preferencia se conserva al recargar (localStorage).
- [ ] 9. Con el modelo de prueba cargado (ModeloNoEntrenado.class): el botón
      "Reproducir última seña" está deshabilitado y la voz NO suena al
      confirmar (seña sintética identificada).
- [ ] 10. Botón "Limpiar historial" vacía la lista y el cartel de vacío vuelve.
- [ ] 11. Más de 50 señas: las más antiguas se descartan (máx. 50 guardadas).
- [ ] 12. "Guardar historial en este dispositivo": al recargar la página el
      historial persiste (solo etiquetas y horas; nunca imágenes ni video).
- [ ] 13. "Mostrar confianza": cada entrada muestra `ETIQUETA · NN%`.
- [ ] 14. Sin SpeechSynthesis (o fallo) en el navegador: el botón de voz está
      deshabilitado y la app funciona igual.
- [ ] 15. La voz se elige con prioridad es-CO → es-ES → cualquier es-*.
- [ ] 16. Sin internet: todo funciona (voz local, sin CDN ni servidores de voz).
- [ ] 17. Narrador/lector: solo se anuncia la adición de una seña confirmada
      (región aria-live), no ruido por fotograma.

## Resultados

| Ítem | Estado | Notas |
|------|--------|-------|
| ...  | ✅/❌   | ...   |