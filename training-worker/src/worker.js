/**
 * BayerFlow Training Data API
 *
 * Endpoints:
 *   POST /v1/training/upload-request  → returns presigned PUT URL + batch_id
 *   PUT  <presigned_url>              → client uploads .bfpatch directly to R2
 *   POST /v1/training/upload-complete → confirms upload, logs metadata
 */

const ALLOWED_ORIGINS = [
  'https://bayerflow.com',
  'https://www.bayerflow.com',
];

const MAX_BATCH_BYTES = 50 * 1024 * 1024; // 50 MB per batch

function corsHeaders(origin) {
  const allowed = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    'Access-Control-Allow-Origin': allowed,
    'Access-Control-Allow-Methods': 'GET, POST, PUT, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function json(data, status = 200, origin = '') {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders(origin),
    },
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    // POST /v1/training/upload-request
    if (request.method === 'POST' && url.pathname === '/v1/training/upload-request') {
      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: 'invalid JSON' }, 400, origin);
      }

      const { batch_size, filename, device_id, app_version } = body;

      if (!filename || !device_id) {
        return json({ error: 'missing filename or device_id' }, 400, origin);
      }
      if (batch_size > MAX_BATCH_BYTES) {
        return json({ error: 'batch too large' }, 413, origin);
      }

      // Sanitize filename — only allow alphanumeric, dash, underscore, dot
      const safeName = String(filename).replace(/[^a-zA-Z0-9._-]/g, '_');
      const batch_id = crypto.randomUUID();
      const key = `batches/${new Date().toISOString().slice(0, 10)}/${device_id}/${batch_id}_${safeName}`;

      // Generate a presigned PUT URL valid for 1 hour
      const presigned_url = await env.TRAINING_BUCKET.createMultipartUpload
        ? null  // createMultipartUpload is for large files, skip
        : null;

      // R2 presigned URL via Workers API
      const signedUrl = await env.TRAINING_BUCKET.createPresignedUrl
        ? await env.TRAINING_BUCKET.createPresignedUrl('PUT', key, { expiresIn: 3600 })
        : null;

      // Fallback: use internal upload path if presigned URLs not available
      if (!signedUrl) {
        // Store the key mapping so upload-complete can verify
        await env.TRAINING_BUCKET.put(`pending/${batch_id}`, JSON.stringify({
          key, device_id, app_version, batch_size, filename: safeName,
          created_at: new Date().toISOString(),
        }), { httpMetadata: { contentType: 'application/json' } });

        return json({
          presigned_url: `${url.origin}/v1/training/upload/${batch_id}`,
          batch_id,
        }, 200, origin);
      }

      return json({ presigned_url: signedUrl, batch_id }, 200, origin);
    }

    // PUT /v1/training/upload/:batch_id  (internal presigned-URL fallback)
    if (request.method === 'PUT' && url.pathname.startsWith('/v1/training/upload/')) {
      const batch_id = url.pathname.split('/').pop();
      if (!batch_id) return json({ error: 'missing batch_id' }, 400, origin);

      // Look up pending metadata
      const metaObj = await env.TRAINING_BUCKET.get(`pending/${batch_id}`);
      if (!metaObj) return json({ error: 'unknown batch_id' }, 404, origin);

      const meta = JSON.parse(await metaObj.text());

      const body = await request.arrayBuffer();
      if (body.byteLength > MAX_BATCH_BYTES) {
        return json({ error: 'batch too large' }, 413, origin);
      }

      await env.TRAINING_BUCKET.put(meta.key, body, {
        httpMetadata: { contentType: 'application/octet-stream' },
        customMetadata: {
          device_id: meta.device_id,
          app_version: meta.app_version || 'unknown',
          batch_id,
        },
      });

      return json({ ok: true }, 200, origin);
    }

    // POST /v1/training/upload-complete
    if (request.method === 'POST' && url.pathname === '/v1/training/upload-complete') {
      let body;
      try {
        body = await request.json();
      } catch {
        return json({ error: 'invalid JSON' }, 400, origin);
      }

      const { batch_id, device_id } = body;
      if (!batch_id) return json({ error: 'missing batch_id' }, 400, origin);

      // Clean up the pending metadata entry
      await env.TRAINING_BUCKET.delete(`pending/${batch_id}`);

      // Log completion record
      await env.TRAINING_BUCKET.put(`completed/${batch_id}`, JSON.stringify({
        batch_id, device_id, completed_at: new Date().toISOString(),
      }), { httpMetadata: { contentType: 'application/json' } });

      return json({ ok: true }, 200, origin);
    }

    // Health check
    if (url.pathname === '/health') {
      return json({ status: 'ok', version: '1.0' }, 200, origin);
    }

    return json({ error: 'not found' }, 404, origin);
  },
};
