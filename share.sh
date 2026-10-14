#!/usr/bin/env bash
# Public demo link: serves the production build (vite preview, /api proxied to :8000) through a
# Cloudflare quick tunnel and prints the https URL. Needs ./run.sh running first (API, workers, Postgres).
# The link is temporary and works only while this laptop and the tunnel are up. Ctrl+C stops sharing.
set -e
cd "$(dirname "$0")"
mkdir -p logs
CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
command -v "$CF" >/dev/null || { echo "cloudflared not found (set CLOUDFLARED=/path/to/cloudflared)"; exit 1; }
curl -sf http://127.0.0.1:8000/api/health >/dev/null || { echo "API is not running: start ./run.sh first"; exit 1; }
(cd frontend && npm run build >/dev/null)
pkill -f "[v]ite preview --host 127.0.0.1 --port 5182" || true
(cd frontend && npx vite preview --host 127.0.0.1 --port 5182 --strictPort > ../logs/preview.log 2>&1 &)
sleep 3
: > logs/tunnel.log
"$CF" tunnel --no-autoupdate --url http://127.0.0.1:5182 > logs/tunnel.log 2>&1 &
TPID=$!
trap 'kill $TPID 2>/dev/null; echo; echo "sharing stopped"' EXIT
until URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/tunnel.log | head -1) && [ -n "$URL" ]; do sleep 1; done
echo "$URL" > logs/public-url.txt
echo "Public link: $URL   (Ctrl+C to stop sharing)"
wait $TPID
