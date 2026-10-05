#!/usr/bin/env bash
set -euo pipefail

mkdir -p models reports

echo "========== 1. TRAIN AND VALIDATE MODEL =========="
python -m src.train

echo "========== 2. STATIC SECURITY SCAN =========="
bandit -r src -f json -o reports/bandit.json
bandit -r src

echo "========== 3. DEPENDENCY VULNERABILITY AUDIT =========="
pip-audit -r requirements.txt | tee reports/pip-audit.txt

echo "========== 4. AUTOMATED TESTS =========="
python -m pytest -q | tee reports/tests.txt

VERSION="$(
  python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("reports/training.json").read_text())
print(report["model_version"])
PY
)"

echo "========== 5. BUILD IMAGE: $VERSION =========="
docker build -t "secure-iris:$VERSION" .

if [ ! -f .env ]; then
  echo "ERROR: .env is missing. Create it locally; do not commit it."
  exit 1
fi

set -a
source .env
set +a

if [ "${#API_KEY}" -lt 32 ]; then
  echo "ERROR: API_KEY is too short or missing."
  exit 1
fi

echo "========== 6. REPLACE PREVIOUS CONTAINER =========="
docker rm -f secure-iris >/dev/null 2>&1 || true

echo "========== 7. DEPLOY RESTRICTED CONTAINER =========="
docker run -d \
  --name secure-iris \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 256m \
  --cpus 1 \
  -p 127.0.0.1:8000:8000 \
  -e API_KEY \
  "secure-iris:$VERSION" > reports/container-id.txt

echo "========== 8. WAIT FOR HEALTH CHECK =========="
READY=0

for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health > reports/health.json; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "ERROR: container did not become healthy."
  docker logs secure-iris || true
  exit 1
fi

echo "========== 9. AUTHENTICATED SMOKE TEST =========="
curl -fsS \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}' \
  http://127.0.0.1:8000/predict \
  > reports/smoke-prediction.json

python - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("reports/smoke-prediction.json").read_text())

if result["species"] != "setosa":
    raise SystemExit(
        f"Smoke test failed: expected setosa, got {result['species']}"
    )

print("Smoke test passed:", result)
PY

echo "========== 10. SAVE DEPLOYMENT EVIDENCE =========="
docker inspect secure-iris \
  --format 'User={{.Config.User}} ReadOnly={{.HostConfig.ReadonlyRootfs}} CapDrop={{.HostConfig.CapDrop}} SecurityOpt={{.HostConfig.SecurityOpt}} Memory={{.HostConfig.Memory}} NanoCPUs={{.HostConfig.NanoCpus}}' \
  > reports/container-security.txt

docker stats secure-iris --no-stream > reports/container-stats.txt

echo "PIPELINE PASSED"
echo "Deployed model version: $VERSION"
