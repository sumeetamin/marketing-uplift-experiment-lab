"""Estimate treatment-effect heterogeneity on a randomized email experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TEST_FRACTION = 0.25
TOP_SHARE = 0.10
WILSON_Z_95 = 1.959963984540054


def pre_treatment_features(data: pd.DataFrame, treatment: str, outcome: str) -> pd.DataFrame:
    """Drop assignment and known post-campaign outcome columns from predictors."""
    blocked = [treatment, outcome]
    blocked.extend(column for column in ("segment", "visit", "spend") if column in data.columns)
    return data.drop(columns=blocked)


def binary_values(values: pd.Series, name: str) -> np.ndarray:
    """Validate a binary 0/1 column instead of silently truncating other values."""
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only binary 0/1 values") from exc
    if numeric.isna().any() or not numeric.isin([0, 1]).all() or set(numeric.unique()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes 0 and 1")
    return numeric.astype(int).to_numpy()


def build_outcome_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocessing),
            ("outcome", LogisticRegression(max_iter=1200)),
        ]
    )


def fit_arm(
    features: pd.DataFrame,
    outcomes: np.ndarray,
    indices: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    model = build_outcome_model(numeric, categorical)
    model.fit(features.iloc[indices], outcomes[indices])
    return model


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z_95) -> list[float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials or z <= 0:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and z > 0")
    if trials == 0:
        return None
    proportion = successes / trials
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / trials + z_squared / (4 * trials * trials)
    ) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def treatment_effect_summary(
    outcomes: np.ndarray, treatment: np.ndarray
) -> dict[str, object]:
    treated = outcomes[treatment == 1]
    control = outcomes[treatment == 0]
    if not len(treated) or not len(control):
        raise ValueError("Both treatment and control observations are required")
    treated_rate = float(treated.mean())
    control_rate = float(control.mean())
    effect = treated_rate - control_rate
    treated_interval = wilson_interval(int(treated.sum()), len(treated))
    control_interval = wilson_interval(int(control.sum()), len(control))
    lower = effect - np.sqrt(
        (treated_rate - treated_interval[0]) ** 2
        + (control_interval[1] - control_rate) ** 2
    )
    upper = effect + np.sqrt(
        (treated_interval[1] - treated_rate) ** 2
        + (control_rate - control_interval[0]) ** 2
    )
    return {
        "estimate": float(effect),
        "ci95": [float(max(-1.0, lower)), float(min(1.0, upper))],
        "treated_n": int(len(treated)),
        "treated_conversions": int(treated.sum()),
        "treated_conversion_rate": treated_rate,
        "control_n": int(len(control)),
        "control_conversions": int(control.sum()),
        "control_conversion_rate": control_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--source-sha256", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    data = pd.read_csv(input_path)
    for column in (args.treatment, args.outcome):
        if column not in data:
            parser.error(f"required column {column!r} is missing")
    data = data.dropna(subset=[args.treatment, args.outcome]).reset_index(drop=True)
    try:
        treatment = binary_values(data[args.treatment], "treatment")
        outcome = binary_values(data[args.outcome], "outcome")
    except ValueError as exc:
        parser.error(str(exc))

    features = pre_treatment_features(data, args.treatment, args.outcome)
    if features.empty or not len(features.columns):
        parser.error("no pre-treatment features remain")
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    stratification = pd.Series(treatment).astype(str) + "_" + pd.Series(outcome).astype(str)
    train_indices, test_indices = train_test_split(
        np.arange(len(data)),
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=stratification,
    )
    control_indices = train_indices[treatment[train_indices] == 0]
    treated_indices = train_indices[treatment[train_indices] == 1]
    control_model = fit_arm(features, outcome, control_indices, numeric, categorical)
    treatment_model = fit_arm(features, outcome, treated_indices, numeric, categorical)

    control_probability = control_model.predict_proba(features.iloc[test_indices])[:, 1]
    treatment_probability = treatment_model.predict_proba(features.iloc[test_indices])[:, 1]
    uplift = treatment_probability - control_probability
    ranked = np.argsort(-uplift, kind="stable")
    ranked_indices = test_indices[ranked]
    ranked_treatment = treatment[test_indices][ranked]
    ranked_outcome = outcome[test_indices][ranked]

    cumulative_treatment = np.cumsum(ranked_treatment * ranked_outcome)
    cumulative_control = np.cumsum((1 - ranked_treatment) * ranked_outcome)
    treatment_n = np.cumsum(ranked_treatment)
    control_n = np.cumsum(1 - ranked_treatment)
    count = np.arange(1, len(ranked) + 1)
    treatment_rate = np.divide(cumulative_treatment, treatment_n, out=np.zeros(len(ranked)), where=treatment_n > 0)
    control_rate = np.divide(cumulative_control, control_n, out=np.zeros(len(ranked)), where=control_n > 0)
    gain = np.where((treatment_n > 0) & (control_n > 0), (treatment_rate - control_rate) * count, 0.0)

    top_count = max(1, int(np.ceil(TOP_SHARE * len(test_indices))))
    holdout_effect = treatment_effect_summary(outcome[test_indices], treatment[test_indices])
    top_effect = treatment_effect_summary(
        ranked_outcome[:top_count], ranked_treatment[:top_count]
    )

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row": ranked_indices,
            "uplift": uplift[ranked],
            "treatment": ranked_treatment,
            "outcome": ranked_outcome,
            "cumulative_uplift_gain": gain,
        }
    ).to_csv(output_dir / "ranked_uplift.csv", index=False)

    report = {
        "rows": int(len(data)),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_sha256": args.source_sha256,
        "train_rows": int(len(train_indices)),
        "holdout_rows": int(len(test_indices)),
        "treatment_rate": float(treatment[test_indices].mean()),
        "holdout_observed_lift": holdout_effect["estimate"],
        "holdout_lift_ci95": holdout_effect["ci95"],
        "holdout_arm_counts": holdout_effect,
        "top_decile_rows": int(top_count),
        "top_decile_treatment_n": top_effect["treated_n"],
        "top_decile_control_n": top_effect["control_n"],
        "top_decile_observed_lift": top_effect["estimate"],
        "top_decile_lift_ci95": top_effect["ci95"],
        "top_decile_arm_counts": top_effect,
        "interval_method": "Newcombe 95% score interval based on independent Wilson intervals",
        "covariate_columns": list(features.columns),
        "note": (
            "Treatment assignment is randomized in the cited source experiment. Effects and uplift rankings "
            "are exploratory single-holdout estimates; a campaign decision requires cost-aware utility and replication."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump(
        {"control_model": control_model, "treatment_model": treatment_model},
        output_dir / "models.joblib",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

