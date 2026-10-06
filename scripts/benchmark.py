import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"

    if mode not in {"normal", "shifted"}:
        raise SystemExit(
            "Usage: python scripts/benchmark.py normal|shifted"
        )

    key = os.environ.get("API_KEY", "")
    if not key:
        raise SystemExit(
            "API_KEY is missing. Run: source .env; export API_KEY"
        )

    reference = json.loads(
        Path("models/reference.json").read_text()
    )["features"]

    latencies = []
    status_counts = {}

    with httpx.Client(
        base_url="http://127.0.0.1:8000",
        headers={"X-API-Key": key},
        timeout=10,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        model_version = health.json()["model_version"]

        for index in range(200):
            features = list(reference[index % len(reference)])

            if mode == "shifted":
                features[0] += 2.0

            start = time.perf_counter()

            response = client.post(
                "/predict",
                json={"features": features},
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            status = str(response.status_code)
            status_counts[status] = status_counts.get(status, 0) + 1

    ordered = sorted(latencies)
    count = len(ordered)
    successful = status_counts.get("200", 0)

    result = {
        "mode": mode,
        "model_version": model_version,
        "requests": count,
        "successful_requests": successful,
        "failed_requests": count - successful,
        "http_status_counts": status_counts,
        "mean_request_ms": round(statistics.mean(ordered), 3),
        "p95_request_ms": round(
            ordered[math.ceil(0.95 * count) - 1], 3
        ),
        "p99_request_ms": round(
            ordered[math.ceil(0.99 * count) - 1], 3
        ),
        "note": (
            "Sequential local HTTP benchmark, including the first "
            "prediction request. Not a concurrent load test."
        ),
    }

    Path("reports").mkdir(exist_ok=True)

    Path(f"reports/benchmark-{mode}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps(result, indent=2))

    if result["failed_requests"]:
        raise SystemExit(
            "Some requests failed. Resolve them before drift testing."
        )


if __name__ == "__main__":
    main()
