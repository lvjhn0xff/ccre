from sklearn.datasets import fetch_openml

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PowerTransformer
)

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

    # Detect feature types
    nominal_features = list(
        X.select_dtypes(include=["object", "category"]).columns
    )

    numeric_features = list(
        X.select_dtypes(include=["number"]).columns
    )

    # Normality detection
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

    # Pipelines
    normal_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    non_normal_pipeline = Pipeline([
        ("power", PowerTransformer(method="yeo-johnson"))
    ])

    nominal_pipeline = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Preprocessor
    preprocessor = ColumnTransformer([

        (
            "normal_num",
            normal_pipeline,
            normal_numeric
        ),

        (
            "non_normal_num",
            non_normal_pipeline,
            non_normal_numeric
        ),

        (
            "nominal",
            nominal_pipeline,
            nominal_features
        )

    ])

    return X, y, preprocessor