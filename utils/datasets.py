from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PowerTransformer
)
from sklearn.impute import SimpleImputer
from scipy.stats import normaltest
import numpy as np


def use_dataset(name, version=1, p_threshold=0.05):

    dataset = fetch_openml(
        name=name,
        version=version,
        as_frame=True
    )

    X = dataset.data
    y = dataset.target

    # -------------------------
    # Feature detection
    # -------------------------
    nominal_features = list(
        X.select_dtypes(include=["object", "category"]).columns
    )

    numeric_features = list(
        X.select_dtypes(include=["number"]).columns
    )

    # -------------------------
    # (Optional) normality split - NOT used in final pipeline logic
    # kept for future experimentation
    # -------------------------
    normal_numeric = []
    non_normal_numeric = []

    for col in numeric_features:
        series = X[col].dropna()

        if len(series) < 8:
            non_normal_numeric.append(col)
            continue

        try:
            _, p = normaltest(series)
            if p > p_threshold:
                normal_numeric.append(col)
            else:
                non_normal_numeric.append(col)
        except Exception:
            non_normal_numeric.append(col)

    # -------------------------
    # FACTORY WITH FLAGS
    # -------------------------
    def make_preprocessor(
        use_onehot=True,
        use_scaling=True,
        use_power=True,
        model_type="tree"   # "tree" or "linear_nn"
    ):

        # -------------------------
        # NUMERIC PIPELINE
        # -------------------------
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        # scaling + power only for non-tree models
        if model_type != "tree":

            if use_power:
                numeric_steps.append(
                    ("power", PowerTransformer(method="yeo-johnson"))
                )

            if use_scaling:
                numeric_steps.append(
                    ("scaler", StandardScaler())
                )

        numeric_pipeline = Pipeline(numeric_steps)

        # -------------------------
        # CATEGORICAL PIPELINE
        # -------------------------
        if use_onehot:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ])
        else:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])

        # -------------------------
        # COMBINE
        # -------------------------
        transformers = []

        if numeric_features:
            transformers.append((
                "num",
                numeric_pipeline,
                numeric_features
            ))

        if nominal_features:
            transformers.append((
                "cat",
                categorical_pipeline,
                nominal_features
            ))

        return ColumnTransformer(transformers)

    return X, y, make_preprocessor