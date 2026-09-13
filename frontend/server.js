/**
 * Servidor estático mínimo para el frontend SIGNIA (sin dependencias).
 *
 * Uso (desde la raíz del repo):
 *   node frontend/server.js            -> http://localhost:8000
 *   node frontend/server.js --port 8080
 *
 * Sirve frontend/ como raíz. Los MIME correctos son importantes para
 * WASM, módulos JS y el modelo TF.js.
 *
 * Nota para teléfono: getUserMedia exige contexto seguro; para probar en
 * un celular sirve la carpeta con HTTPS (p. ej. `npx serve --ssl`) o usa
 * la webcam del computador (https://localhost con un túnel).
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream',
  '.task': 'application/octet-stream',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const portArg = process.argv.indexOf('--port');
const PORT = portArg >= 0 ? Number(process.argv[portArg + 1]) : 8000;

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  let filePath = path.normalize(path.join(ROOT, urlPath));

  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end('403');
    return;
  }

  fs.stat(filePath, (err, st) => {
    if (!err && st.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }
    fs.readFile(filePath, (err2, data) => {
      if (err2) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('404 — no encontrado');
        return;
      }
      const ext = path.extname(filePath).toLowerCase();
      res.writeHead(200, {
        'Content-Type': MIME[ext] ?? 'application/octet-stream',
        'Cache-Control': 'no-store', // desarrollo: siempre fresco
      });
      res.end(data);
    });
  });
});

server.listen(PORT, () => {
  console.log(`SIGNIA frontend -> http://localhost:${PORT}`);
});