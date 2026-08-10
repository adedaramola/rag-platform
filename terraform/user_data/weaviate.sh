#!/bin/bash
# Weaviate node bootstrap — Amazon Linux 2023
# Mounts the dedicated EBS data volume, installs Docker, and runs Weaviate.
set -euo pipefail
exec > >(tee /var/log/rag-platform-bootstrap.log | logger -t rag-platform-weaviate) 2>&1

echo "=== RAG Platform: Weaviate bootstrap starting ==="

# ---------------------------------------------------------------------------
# 1. Mount the dedicated data EBS volume (/dev/xvdf or NVMe equivalent)
#    On nitro-based instances, AWS maps /dev/xvdf → /dev/nvme1n1.
# ---------------------------------------------------------------------------
DATA_DEVICE=""
for dev in /dev/xvdf /dev/nvme1n1 /dev/nvme2n1; do
  if [ -b "$dev" ]; then
    DATA_DEVICE="$dev"
    break
  fi
done

if [ -z "$DATA_DEVICE" ]; then
  echo "ERROR: no data EBS volume found — volume may not be attached yet, retrying..."
  sleep 15
  for dev in /dev/xvdf /dev/nvme1n1 /dev/nvme2n1; do
    if [ -b "$dev" ]; then DATA_DEVICE="$dev"; break; fi
  done
fi

if [ -n "$DATA_DEVICE" ]; then
  # Format only if the volume has no filesystem
  if ! blkid "$DATA_DEVICE" > /dev/null 2>&1; then
    mkfs -t xfs "$DATA_DEVICE"
    echo "Formatted $DATA_DEVICE as xfs"
  fi
  mkdir -p /var/lib/weaviate
  mount "$DATA_DEVICE" /var/lib/weaviate
  # Persist mount across reboots
  DEVICE_UUID=$(blkid -s UUID -o value "$DATA_DEVICE")
  echo "UUID=$DEVICE_UUID /var/lib/weaviate xfs defaults,nofail 0 2" >> /etc/fstab
  echo "Mounted $DATA_DEVICE at /var/lib/weaviate"
else
  echo "WARNING: data volume not found — using root volume. Weaviate data will be lost on instance replacement."
  mkdir -p /var/lib/weaviate
fi

# ---------------------------------------------------------------------------
# 2. Install Docker
# ---------------------------------------------------------------------------
dnf install -y docker
systemctl enable docker
systemctl start docker
echo "Docker installed and started"

# ---------------------------------------------------------------------------
# 3. Run Weaviate as a Docker container with restart policy
# ---------------------------------------------------------------------------
docker run -d \
  --name weaviate \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 50051:50051 \
  -v /var/lib/weaviate:/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e LIMIT_RESOURCES=false \
  "cr.weaviate.io/semitechnologies/weaviate:${weaviate_version}"

echo "=== RAG Platform: Weaviate bootstrap complete ==="
