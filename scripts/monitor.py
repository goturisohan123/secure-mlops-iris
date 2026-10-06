import json
import os
from pathlib import Path

import httpx
from scipy.stats import ks_2samp

reference = json.loads(
    Path("models/reference.json").read_text()
)["features"]

model = json.loads(
    Path("models/model.json").read_text()
)

response = httpx.get(
    "http://127.0.0.1:8000/stats",
    headers={"X-API-Key": os.environ["API_KEY"]},
    timeout=10,
)

response.raise_for_status()

stats = response.json()
recent = stats["recent_inputs"]

if len(recent) < 20:
    raise SystemExit(
        "Send at least 20 valid predictions before checking drift."
    )

results = []
threshold = 0.05 / len(model["feature_names"])

for index, name in enumerate(model["feature_names"]):
    test = ks_2samp(
        [row[index] for row in reference],
        [row[index] for row in recent],
    )

    results.append(
        {
            "feature": name,
            "ks_statistic": round(float(test.statistic), 6),
            "p_value": round(float(test.pvalue), 6),
            "alert": bool(test.pvalue < threshold),
        }
    )

report = {
    "model_version": stats["model_version"],
    "reference_samples": len(reference),
    "recent_samples": len(recent),
    "per_feature_threshold": threshold,
    "any_alert": any(item["alert"] for item in results),
    "features": results,
    "limitation": (
        "This detects input-distribution change only. "
        "It does not measure accuracy or prove concept drift."
    ),
}

Path("reports").mkdir(exist_ok=True)

Path("reports/drift.json").write_text(
    json.dumps(report, indent=2) + "\n"
)

print(json.dumps(report, indent=2))
