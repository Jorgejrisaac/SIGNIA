# PRUEBA MANUAL — Fase 9 (frontend base)

Requiere hardware (webcam o celular): NO se automatiza en CI.

## Configuración

```powershell
cd C:\Users\Usuario\Desktop\SIGNIA
node frontend/server.js          # http://localhost:8000
```

Para probar en el TELEFONO: getUserMedia exige contexto seguro (HTTPS).
Opciones: webcam del computador (recomendado), o servir con HTTPS
(`npx serve --ssl`) en la misma red local.

Nota: para la detección real NO está exportado un modelo LSC todavía;
sin `frontend/models/signia/model.json` la app debe mostrar honestamente
"Modelo de demostración no disponible / Modelo no cargado".

## Checklist

- [ ] La página abre correctamente (http://localhost:8000).
- [ ] Solicita permiso de cámara (chip "Esperando permiso de cámara").
- [ ] La cámara se muestra (video en espejo).
- [ ] Los landmarks aparecen (puntos + conexiones sobre la mano real).
- [ ] Left/Right funcionan (etiqueta según MediaPipe handedness).
- [ ] Se detectan 0/1/2 manos correctamente (chip ✋ y diagnóstico "Hands detected").
- [ ] El modelo carga correctamente cuando existe (chip "Modelo listo").
- [ ] Si no existe modelo, aparece error honesto ("Modelo no cargado" + diagnóstico ERROR).
- [ ] Sin mano: "Coloca tu mano frente a la cámara" (no se predice nada falso).
- [ ] No se envían frames a ningún servidor (ver "Verificación de privacidad").
- [ ] La página funciona en pantalla pequeña (responsive: cámara 3:4, botones grandes).
- [ ] No aparecen errores críticos en consola (F12 → Console).

## Errores de cámara

Probar de forma controlada:

- [ ] Permiso denegado → "Permiso de cámara denegado..." + botón Reintentar.
- [ ] Cámara ocupada (abrir otra app con la cámara) → mensaje claro.
- [ ] Sin cámara (desactivar o dispositivo sin webcam) → "No se encontró ninguna cámara".

## Modelo cargado (cuando exista)

- [ ] Con modelo real exportado (export_tfjs.py): chip verde "Modelo listo",
      shapes correctas en Diagnóstico, y con la mano en cámara la predicción
      muestra la seña real (label + %).
- [ ] Con modelo sintético de 2 clases (solo integración): debe aparecer el badge
      "MODELO DE PRUEBA — NO REPRESENTA RECONOCIMIENTO LSC".

## Verificación de privacidad (red)

Con la página abierta y la cámara activa:

1. DevTools → Network.
2. Activa "Preserve log" y observa durante ~30 s moviendo la mano.
3. Verifica que SOLO aparezcan peticiones a localhost (mismo origen)
   y ninguna a dominios externos (sin CDN, sin embeds, sin telemetría).

## Accesibilidad

- [ ] Tab llega a todos los botones; Enter/Espacio los activa.
- [ ] Los chips de estado tienen texto + icono (no solo color).
- [ ] Los cambios de estado se anuncian en la región aria-live (lector de pantalla).
- [ ] Contraste de texto suficiente (comprobable con inspección, p. ej. WCAG AA).