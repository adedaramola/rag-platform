#!/bin/bash
# API node bootstrap — Amazon Linux 2023
# Clones the repo, writes secrets, installs deps, and starts the FastAPI service.
# All template variables are injected by Terraform at apply time.
set -euo pipefail
exec > >(tee /var/log/rag-platform-bootstrap.log | logger -t rag-platform-api) 2>&1

echo "=== RAG Platform: API bootstrap starting ==="

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
# curl-minimal is already on AL2023 — installing curl conflicts with it
# python3.11-pip is not a valid AL2023 package; use ensurepip instead
dnf install -y git python3.11
python3.11 -m ensurepip --upgrade
echo "Python $(python3.11 --version) installed"

# ---------------------------------------------------------------------------
# 2. App user (home at /home/rag-platform — separate from the app directory)
# ---------------------------------------------------------------------------
useradd -r -m -s /bin/bash rag-platform || true

# ---------------------------------------------------------------------------
# 3. Clone the private repo using GitHub PAT, then strip token from remote
#    /opt/rag-platform must not exist before clone — git creates it
# ---------------------------------------------------------------------------
REPO_DIR="/opt/rag-platform"
rm -rf "$REPO_DIR"
git clone --branch "${github_ref}" --single-branch "${github_clone_url}" "$REPO_DIR"

# Strip the PAT so it is never stored in .git/config after clone
git -C "$REPO_DIR" remote set-url origin "${github_repo}"
chown -R rag-platform:rag-platform "$REPO_DIR"
echo "Repo cloned to $REPO_DIR"

# ---------------------------------------------------------------------------
# 4. Write .env — connection config + API keys (injected by Terraform)
# ---------------------------------------------------------------------------
cat > "$REPO_DIR/.env" << 'ENVEOF'
RAG_PLATFORM_STORE_BACKEND=weaviate
RAG_PLATFORM_EMBED_BACKEND=${embed_backend}
RAG_PLATFORM_CACHE_BACKEND=${cache_backend}
RAG_PLATFORM_API_RATE_LIMIT=${api_rate_limit}
ENVEOF

# Append interpolated values separately to avoid Terraform/bash quoting issues
cat >> "$REPO_DIR/.env" << ENVEOF
RAG_PLATFORM_WEAVIATE_HOST=${weaviate_host}
RAG_PLATFORM_WEAVIATE_PORT=${weaviate_port}
RAG_PLATFORM_ANTHROPIC_API_KEY=${anthropic_api_key}
RAG_PLATFORM_OPENAI_API_KEY=${openai_api_key}
RAG_PLATFORM_API_KEY=${api_key}
RAG_PLATFORM_APPROVED_SOURCE_IDS='${approved_source_ids}'
ENVEOF

chown rag-platform:rag-platform "$REPO_DIR/.env"
chmod 600 "$REPO_DIR/.env"
echo ".env written"

# ---------------------------------------------------------------------------
# 5. Create venv and install dependencies (API + UI; OpenAI embeddings need no heavy extra)
# ---------------------------------------------------------------------------
sudo -u rag-platform python3.11 -m venv "$REPO_DIR/.venv"
sudo -u rag-platform "$REPO_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo mkdir -p /opt/pip-tmp && chmod 1777 /opt/pip-tmp
sudo -u rag-platform TMPDIR=/opt/pip-tmp "$REPO_DIR/.venv/bin/pip" install --no-cache-dir -e "$REPO_DIR[api,ui]"
echo "Dependencies installed"

# ---------------------------------------------------------------------------
# 6. Systemd services — FastAPI and Streamlit UI
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/rag-platform-api.service << 'UNIT'
[Unit]
Description=RAG Platform RAG FastAPI server
After=network-online.target
Wants=network-online.target

[Service]
User=rag-platform
Group=rag-platform
WorkingDirectory=/opt/rag-platform
EnvironmentFile=/opt/rag-platform/.env
ExecStart=/opt/rag-platform/.venv/bin/uvicorn rag.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers ${api_workers} \
    --no-access-log
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-platform-api

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/rag-platform-ui.service << 'UNIT'
[Unit]
Description=RAG Platform RAG Streamlit UI
After=rag-platform-api.service
Wants=rag-platform-api.service

[Service]
User=rag-platform
Group=rag-platform
WorkingDirectory=/opt/rag-platform
EnvironmentFile=/opt/rag-platform/.env
Environment=RAG_PLATFORM_API_URL=http://localhost:8000
ExecStart=/opt/rag-platform/.venv/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rag-platform-ui

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable rag-platform-api rag-platform-ui

# ---------------------------------------------------------------------------
# 7. Wait for Weaviate to be ready before starting the API
#    (cross-encoder model also downloads on first startup — allow time)
# ---------------------------------------------------------------------------
echo "Waiting for Weaviate at ${weaviate_host}:${weaviate_port}..."
for i in $(seq 1 36); do
  if curl -sf "http://${weaviate_host}:${weaviate_port}/v1/.well-known/ready" > /dev/null 2>&1; then
    echo "Weaviate is ready after $((i * 10))s"
    break
  fi
  if [ "$i" -eq 36 ]; then
    echo "WARNING: Weaviate not ready after 360s — starting API anyway (systemd will retry)"
  else
    echo "Attempt $i/36 — retrying in 10s..."
    sleep 10
  fi
done

systemctl start rag-platform-api
echo "rag-platform-api.service started"

systemctl start rag-platform-ui
echo "rag-platform-ui.service started"

echo "=== RAG Platform: API bootstrap complete ==="
echo "Monitor API: journalctl -u rag-platform-api -f"
echo "Monitor UI:  journalctl -u rag-platform-ui -f"
