import hashlib
import json
import os
from pathlib import Path

import sklearn
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

MODELS = Path("models")
REPORTS = Path("reports")


def evaluate(model, X, y):
    predicted = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    data = load_iris()

    X_train, X_other, y_train, y_other = train_test_split(
        data.data,
        data.target,
        test_size=0.4,
        random_state=42,
        stratify=data.target,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_other,
        y_other,
        test_size=0.5,
        random_state=42,
        stratify=y_other,
    )

    depth = int(os.getenv("TREE_DEPTH", "3"))

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )

    baseline = DummyClassifier(strategy="most_frequent")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )

    for candidate in (tree, baseline, logistic):
        candidate.fit(X_train, y_train)

    tree_validation = evaluate(tree, X_val, y_val)
    baseline_validation = evaluate(baseline, X_val, y_val)

    results = {
        "dataset": "Iris",
        "dataset_sha256": hashlib.sha256(
            data.data.tobytes() + data.target.tobytes()
        ).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "split_sizes": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "tree_depth": depth,
        "validation": {
            "decision_tree": tree_validation,
            "majority_baseline": baseline_validation,
            "logistic_regression": evaluate(logistic, X_val, y_val),
        },
        "quality_threshold_macro_f1": 0.85,
    }

    passed = (
        tree_validation["macro_f1"] >= 0.85
        and tree_validation["macro_f1"]
        > baseline_validation["macro_f1"]
    )

    results["quality_gate_passed"] = passed

    if not passed:
        (REPORTS / "training.json").write_text(
            json.dumps(results, indent=2)
        )
        raise SystemExit(
            "Quality gate failed. Model was not exported."
        )

    results["test"] = evaluate(tree, X_test, y_test)

    values = tree.tree_.value[:, 0, :]
    probabilities = values / values.sum(axis=1, keepdims=True)

    artifact = {
        "feature_names": data.feature_names,
        "class_names": data.target_names.tolist(),
        "children_left": tree.tree_.children_left.tolist(),
        "children_right": tree.tree_.children_right.tolist(),
        "feature": tree.tree_.feature.tolist(),
        "threshold": tree.tree_.threshold.tolist(),
        "probabilities": probabilities.tolist(),
    }

    model_text = json.dumps(artifact, sort_keys=True)
    version = hashlib.sha256(
        model_text.encode()
    ).hexdigest()[:12]

    (MODELS / "model.json").write_text(model_text)

    (MODELS / "reference.json").write_text(
        json.dumps(
            {"features": X_train.tolist()},
            indent=2,
        )
    )

    (MODELS / "parity_cases.json").write_text(
        json.dumps(
            {
                "features": X_test.tolist(),
                "predictions": tree.predict(X_test).tolist(),
                "probabilities": tree.predict_proba(X_test).tolist(),
            },
            indent=2,
        )
    )

    results["model_version"] = version

    (REPORTS / "training.json").write_text(
        json.dumps(results, indent=2)
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
