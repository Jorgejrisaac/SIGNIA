# Despliegue de SIGNIA

## Recomendacion

Para este MVP recomiendo **Cloudflare Pages**: SIGNIA es una aplicacion
estatica, no necesita backend, obtiene HTTPS automaticamente y sirve bien los
archivos grandes de MediaPipe WASM y el modelo TF.js.

## Cloudflare Pages

1. Crea un repositorio Git privado o publico y sube el proyecto.
2. En Cloudflare, abre **Workers & Pages** y crea una aplicacion Pages desde
   el repositorio.
3. Configura:
   - Build command: vacio
   - Output directory: `frontend`
   - Root directory: `/`
   - Variables de entorno: ninguna
4. Despliega y abre la URL `https://...pages.dev`.
5. Permite la camara cuando el navegador lo solicite.

No se debe publicar `ai/`, `.venv/` ni `tests/`: al usar `frontend` como
directorio de salida solo se publica la aplicacion necesaria.

## Prueba local

`http://localhost:8000` funciona para probar desde el mismo PC:

```powershell
node frontend/server.js
```

Para probar la camara desde un telefono hace falta HTTPS. `localhost` en el
telefono apunta al propio telefono, no al PC. Como alternativa temporal:

```powershell
npx serve frontend --ssl
```

El certificado de desarrollo puede requerir aceptar una advertencia en el
telefono. Para un uso real, usa el dominio HTTPS de Cloudflare Pages o un
dominio propio.

## Verificacion posterior

Comprueba que respondan con `200`:

- `/`
- `/models/signia/model.json`
- `/vendor/mediapipe/hand_landmarker.task`
- `/vendor/mediapipe/wasm/vision_wasm_internal.wasm`

En el panel de diagnostico deben aparecer `Camera: OK`, `MediaPipe: READY` y
`Model: READY`. La auditoria de vectores queda desactivada por defecto; para
depurar una sesion puntual se puede abrir la URL con `?debug=1`.

## Dominio propio

En Cloudflare Pages agrega el dominio en **Custom domains**. Cloudflare
configura el certificado TLS automaticamente. No habilites acceso de camara
por HTTP en produccion.

## Vercel

Vercel tambien es compatible. Configura el proyecto con:

- Framework preset: `Other`
- Root directory: `frontend`
- Build command: vacio
- Output directory: `.`
- Variables de entorno: ninguna

`frontend/vercel.json` aplica los headers de seguridad. Vercel entrega HTTPS
automaticamente, por lo que la camara funciona desde un telefono usando la
URL de produccion.

Si Vercel se configura con la raiz del repositorio en lugar de `frontend`,
el `vercel.json` de la raiz redirige la URL publica hacia `frontend/` y evita
el error `404: NOT_FOUND`.
