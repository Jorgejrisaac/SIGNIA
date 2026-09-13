# Prueba manual F12 — UI final del MVP

Prerrequisito: F9–F11 funcionando. `node frontend/server.js` →
http://localhost:8000. Idealmente con las DevTools abiertas para cambiar
el tamaño de pantalla y el emulador de dispositivo.

## Checklist

### Responsive
- [ ] Desktop 1440px: dos columnas (Cámara | Traducción) centradas; controles e historial abajo; nada cortado.
- [ ] Desktop 1024px: misma estructura, sin scroll horizontal.
- [ ] Tablet 768px: cámara | traducción lado a lado; controles e historial apilados.
- [ ] Mobile 430px: orden HEADER → CÁMARA → TRADUCCIÓN → CONTROLES → HISTORIAL.
- [ ] Mobile 390px: sin desbordes ni texto cortado (etiqueta larga con `overflow-wrap`).
- [ ] Mobile 375px: idem; botones y toggles ≥ 48px de alto.
- [ ] Mobile 320px: idem; header envuelve sin romper; sin scroll horizontal en NINGÚN ancho.
- [ ] Cámara 3/4 en móvil y sin deformación (espejo intacto); 4/3 en escritorio; 16/9 en landscape móvil.
- [ ] En teléfono, la app se abre mediante HTTPS; por HTTP muestra un error explícito de contexto inseguro.

### Estados y errores
- [ ] Sin cámara: chip rojo con icono ✖ + banner "Sin cámara disponible"; botón "Reintentar cámara".
- [ ] Cámara denegada: banner "Permiso de cámara denegado" con el detalle legible (sin stack traces).
- [ ] Modelo ausente: tarjeta roja "Modelo no disponible" + "Necesitas cargar el modelo SIGNIA..." + botón Reintentar; la cámara y MediaPipe siguen funcionando (diagnóstico).
- [ ] Modelo sintético (2 clases): tarjeta ámbar "MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC"; voz y "Reproducir última" bloqueadas.
- [ ] Modelo de 10 clases: tarjeta verde "Modelo SIGNIA" + "versión X" (sin declarar precisión).

### Estados del sistema (header)
- [ ] Chips compactos con icono + texto: "● Cámara activa", "● IA lista", "✋ Mano detectada"; el color por sí solo nunca es la única señal.

### Predicción y confirmación
- [ ] Sin mano: "Coloca tu mano frente a la cámara".
- [ ] En transición: "Esperando una seña...".
- [ ] Confianza baja: "Confianza insuficiente" (nunca una etiqueta falsa).
- [ ] Seña confirmada: palabra dominante en grande + "Confianza NN%" con barra; pulso breve al confirmar.
- [ ] Sin modelo: la predicción nunca muestra nada (placeholder honesto).

### Historial y voz
- [ ] "TRADUCCIONES RECIENTES" numeradas (01, 02...); la palabra domina; confianza opcional al lado.
- [ ] "Limpiar historial" vacía la lista y restaura el cartel de vacío.
- [ ] 🔊/🔇 reflecta el estado con `aria-pressed`; "Reproducir última" habla solo si hay historial y NO hay modelo de prueba.

### Diagnóstico
- [ ] Oculto por defecto; se abre con "Diagnóstico" (Alt+D) y muestra Camera, MediaPipe, Model, Input shape, Output shape, Model version, Hands, FPS, RAW, SMOOTHED, Smoothing, Modelo.
- [ ] No aparece en la interfaz principal salvo que se abra.

### Accesibilidad y teclado
- [ ] Teclado: Tab recorre botones/toggles con `focus-visible` visible; Enter/Espacio activan; Shift+Tab retrocede.
- [ ] Enlace "Saltar al contenido" aparece al enfocar Tab la primera vez.
- [ ] `aria-pressed` en botón de voz; toggles con `<label>` clicable.
- [ ] Lector de pantalla: anuncia SOLO "Seña confirmada: HOLA" (y cambios de cámara/modelo), NUNCA fotogramas/RAW/FPS.
- [ ] Contrastes: texto #F8FAFC sobre #0F172A/#1A2742; badges de estado legibles.
- [ ] Preferencias → "reducir movimiento": el pulso y los hover dejan de animarse instantáneamente.

### Privacidad / honestidad
- [ ] Footer: "Tu cámara se procesa localmente. El video no se envía a servidores." y "Reconocimiento LSC real: PENDIENTE DE VALIDACIÓN."
- [ ] Sin pestañas de red hacia dominios externos (todo local: vendor/, sin CDN).
- [ ] Metadata: título, descripción, theme-color, favicon.svg y manifest.webmanifest presentes y servidos (200).

## Resultados

| Ítem | Estado | Notas |
|------|--------|-------|
| ...  | ✅/❌   | ...   |
