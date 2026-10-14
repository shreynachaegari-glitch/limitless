#!/usr/bin/env bash
# Start Postgres (podman), the API, the worker and the frontend dev server.
set -e
cd "$(dirname "$0")"
podman start rp-pg >/dev/null 2>&1 || podman run -d --name rp-pg -e POSTGRES_USER=rp -e POSTGRES_PASSWORD=rp \
  -e POSTGRES_DB=researchpulse -p 5433:5432 -v rp-pgdata:/var/lib/postgresql/data docker.io/pgvector/pgvector:pg17
mkdir -p logs
(cd backend && .venv/bin/uvicorn rp.api:app --host 127.0.0.1 --port 8000 > ../logs/api.log 2>&1 &)
(cd backend && .venv/bin/python -m rp.worker > ../logs/worker.log 2>&1 &)
(cd frontend && npm run dev -- --host 127.0.0.1 --port 5181 --strictPort > ../logs/web.log 2>&1 &)
echo "API http://127.0.0.1:8000/docs   UI http://127.0.0.1:5181   logs in ./logs"
