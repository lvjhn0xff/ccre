from __future__ import annotations

from typing import Any

# =========================
# SCIKIT-LEARN CLASSIFIERS
# =========================

# Linear Models
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)

# Support Vector Machines
from sklearn.svm import (
    SVC,
    NuSVC,
    LinearSVC,
)

# Neighbors
from sklearn.neighbors import (
    KNeighborsClassifier,
    RadiusNeighborsClassifier,
    NearestCentroid,
)

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessClassifier

# Decision Trees
from sklearn.tree import (
    DecisionTreeClassifier,
    ExtraTreeClassifier,
)

# Ensemble Methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

# Naive Bayes
from sklearn.naive_bayes import (
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
)

# Neural Networks
from sklearn.neural_network import MLPClassifier

# Semi-Supervised
from sklearn.semi_supervised import (
    LabelPropagation,
    LabelSpreading,
    SelfTrainingClassifier,
)

# Discriminant Analysis
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

# Dummy / Baseline
from sklearn.dummy import DummyClassifier

# XGBoost
from xgboost import (
    XGBClassifier,
    XGBRFClassifier,
)

# =========================================================
# GLOBAL VERBOSE DEFAULTS
# =========================================================

DEFAULT_VERBOSE = 1
DEFAULT_VERBOSITY = 1


# =========================================================
# TEMPLATE HELPERS
# =========================================================

# Pattern:
# def use_ModelName(**overrides: Any) -> ModelName:
#     defaults = {}
#     defaults.update(overrides)
#     return ModelName(**defaults)


# =========================================================
# LINEAR MODELS
# =========================================================


def use_LogisticRegression(**overrides: Any) -> LogisticRegression:
    defaults = {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return LogisticRegression(**defaults)


def use_LogisticRegressionCV(**overrides: Any) -> LogisticRegressionCV:
    defaults = {
        "cv": 5,
        "max_iter": 1000,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return LogisticRegressionCV(**defaults)


def use_PassiveAggressiveClassifier(**overrides: Any) -> PassiveAggressiveClassifier:
    defaults = {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return PassiveAggressiveClassifier(**defaults)


def use_Perceptron(**overrides: Any) -> Perceptron:
    defaults = {
        "max_iter": 1000,
        "eta0": 1.0,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return Perceptron(**defaults)


def use_RidgeClassifier(**overrides: Any) -> RidgeClassifier:
    defaults = {
        "alpha": 1.0,
        "random_state": 42,
    }
    defaults.update(overrides)
    return RidgeClassifier(**defaults)


def use_RidgeClassifierCV(**overrides: Any) -> RidgeClassifierCV:
    defaults = {
        "alphas": (0.1, 1.0, 10.0),
    }
    defaults.update(overrides)
    return RidgeClassifierCV(**defaults)


def use_SGDClassifier(**overrides: Any) -> SGDClassifier:
    defaults = {
        "loss": "log_loss",
        "max_iter": 1000,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return SGDClassifier(**defaults)


# =========================================================
# SVM
# =========================================================


def use_SVC(**overrides: Any) -> SVC:
    defaults = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
        "verbose": True,
    }
    defaults.update(overrides)
    return SVC(**defaults)


def use_NuSVC(**overrides: Any) -> NuSVC:
    defaults = {
        "nu": 0.5,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
        "verbose": True,
    }
    defaults.update(overrides)
    return NuSVC(**defaults)


def use_LinearSVC(**overrides: Any) -> LinearSVC:
    defaults = {
        "C": 1.0,
        "max_iter": 5000,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return LinearSVC(**defaults)


# =========================================================
# NEIGHBORS
# =========================================================


def use_KNeighborsClassifier(**overrides: Any) -> KNeighborsClassifier:
    defaults = {
        "n_neighbors": 5,
        "weights": "uniform",
    }
    defaults.update(overrides)
    return KNeighborsClassifier(**defaults)


def use_RadiusNeighborsClassifier(**overrides: Any) -> RadiusNeighborsClassifier:
    defaults = {
        "radius": 1.0,
        "weights": "uniform",
    }
    defaults.update(overrides)
    return RadiusNeighborsClassifier(**defaults)


def use_NearestCentroid(**overrides: Any) -> NearestCentroid:
    defaults = {}
    defaults.update(overrides)
    return NearestCentroid(**defaults)


# =========================================================
# GAUSSIAN PROCESS
# =========================================================


def use_GaussianProcessClassifier(**overrides: Any) -> GaussianProcessClassifier:
    defaults = {
        "random_state": 42,
    }
    defaults.update(overrides)
    return GaussianProcessClassifier(**defaults)


# =========================================================
# TREES
# =========================================================


def use_DecisionTreeClassifier(**overrides: Any) -> DecisionTreeClassifier:
    defaults = {
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
    }
    defaults.update(overrides)
    return DecisionTreeClassifier(**defaults)


def use_ExtraTreeClassifier(**overrides: Any) -> ExtraTreeClassifier:
    defaults = {
        "random_state": 42,
    }
    defaults.update(overrides)
    return ExtraTreeClassifier(**defaults)


# =========================================================
# ENSEMBLES
# =========================================================


def use_AdaBoostClassifier(**overrides: Any) -> AdaBoostClassifier:
    defaults = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    defaults.update(overrides)
    return AdaBoostClassifier(**defaults)


def use_BaggingClassifier(**overrides: Any) -> BaggingClassifier:
    defaults = {
        "n_estimators": 100,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return BaggingClassifier(**defaults)


def use_ExtraTreesClassifier(**overrides: Any) -> ExtraTreesClassifier:
    defaults = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return ExtraTreesClassifier(**defaults)


def use_GradientBoostingClassifier(**overrides: Any) -> GradientBoostingClassifier:
    defaults = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return GradientBoostingClassifier(**defaults)


def use_HistGradientBoostingClassifier(**overrides: Any) -> HistGradientBoostingClassifier:
    defaults = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_depth": None,
        "random_state": 42,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return HistGradientBoostingClassifier(**defaults)


def use_RandomForestClassifier(**overrides: Any) -> RandomForestClassifier:
    defaults = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": DEFAULT_VERBOSE,
    }
    defaults.update(overrides)
    return RandomForestClassifier(**defaults)


# =========================================================
# NAIVE BAYES
# =========================================================


def use_BernoulliNB(**overrides: Any) -> BernoulliNB:
    defaults = {
        "alpha": 1.0,
    }
    defaults.update(overrides)
    return BernoulliNB(**defaults)


def use_CategoricalNB(**overrides: Any) -> CategoricalNB:
    defaults = {
        "alpha": 1.0,
    }
    defaults.update(overrides)
    return CategoricalNB(**defaults)


def use_ComplementNB(**overrides: Any) -> ComplementNB:
    defaults = {
        "alpha": 1.0,
    }
    defaults.update(overrides)
    return ComplementNB(**defaults)


def use_GaussianNB(**overrides: Any) -> GaussianNB:
    defaults = {}
    defaults.update(overrides)
    return GaussianNB(**defaults)


def use_MultinomialNB(**overrides: Any) -> MultinomialNB:
    defaults = {
        "alpha": 1.0,
    }
    defaults.update(overrides)
    return MultinomialNB(**defaults)


# =========================================================
# NEURAL NETWORKS
# =========================================================


def use_MLPClassifier(**overrides: Any) -> MLPClassifier:
    defaults = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 1000,
        "random_state": 42,
        "verbose": True,
    }
    defaults.update(overrides)
    return MLPClassifier(**defaults)


# =========================================================
# SEMI-SUPERVISED
# =========================================================


def use_LabelPropagation(**overrides: Any) -> LabelPropagation:
    defaults = {
        "kernel": "rbf",
        "gamma": 20,
        "max_iter": 1000,
    }
    defaults.update(overrides)
    return LabelPropagation(**defaults)


def use_LabelSpreading(**overrides: Any) -> LabelSpreading:
    defaults = {
        "kernel": "rbf",
        "gamma": 20,
        "max_iter": 1000,
    }
    defaults.update(overrides)
    return LabelSpreading(**defaults)


def use_SelfTrainingClassifier(
    base_estimator=None,
    **overrides: Any,
) -> SelfTrainingClassifier:
    defaults = {
        "estimator": base_estimator or LogisticRegression(),
        "verbose": True,
    }
    defaults.update(overrides)
    return SelfTrainingClassifier(**defaults)


# =========================================================
# DISCRIMINANT ANALYSIS
# =========================================================


def use_LinearDiscriminantAnalysis(**overrides: Any) -> LinearDiscriminantAnalysis:
    defaults = {}
    defaults.update(overrides)
    return LinearDiscriminantAnalysis(**defaults)


def use_QuadraticDiscriminantAnalysis(**overrides: Any) -> QuadraticDiscriminantAnalysis:
    defaults = {
        "reg_param": 0.0,
    }
    defaults.update(overrides)
    return QuadraticDiscriminantAnalysis(**defaults)


# =========================================================
# DUMMY
# =========================================================


def use_DummyClassifier(**overrides: Any) -> DummyClassifier:
    defaults = {
        "strategy": "most_frequent",
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


def use_DummyClassifierMostFrequent(**overrides: Any) -> DummyClassifier:
    defaults = {
        "strategy": "most_frequent",
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


def use_DummyClassifierPrior(**overrides: Any) -> DummyClassifier:
    defaults = {
        "strategy": "prior",
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


def use_DummyClassifierStratified(**overrides: Any) -> DummyClassifier:
    defaults = {
        "strategy": "stratified",
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


def use_DummyClassifierUniform(**overrides: Any) -> DummyClassifier:
    defaults = {
        "strategy": "uniform",
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


def use_DummyClassifierConstant(
    constant: Any = 1,
    **overrides: Any,
) -> DummyClassifier:
    defaults = {
        "strategy": "constant",
        "constant": constant,
        "random_state": 42,
    }
    defaults.update(overrides)
    return DummyClassifier(**defaults)


# =========================================================
# XGBOOST
# =========================================================


def use_XGBClassifier(**overrides: Any) -> XGBClassifier:
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": DEFAULT_VERBOSITY,
    }
    defaults.update(overrides)
    return XGBClassifier(**defaults)


def use_XGBRFClassifier(**overrides: Any) -> XGBRFClassifier:
    defaults = {
        "n_estimators": 300,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bynode": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": DEFAULT_VERBOSITY,
    }
    defaults.update(overrides)
    return XGBRFClassifier(**defaults)

# =========================================================
# MODEL FACTORY MAP
# =========================================================

MODEL_FACTORIES: dict[str, Any] = {
    # Linear Models
    "LogisticRegression": lambda **kw: use_LogisticRegression(**kw),
    "LogisticRegressionCV": lambda **kw: use_LogisticRegressionCV(**kw),
    "PassiveAggressiveClassifier": lambda **kw: use_PassiveAggressiveClassifier(**kw),
    "Perceptron": lambda **kw: use_Perceptron(**kw),
    "RidgeClassifier": lambda **kw: use_RidgeClassifier(**kw),
    "RidgeClassifierCV": lambda **kw: use_RidgeClassifierCV(**kw),
    "SGDClassifier": lambda **kw: use_SGDClassifier(**kw),

    # SVM
    "SVC": lambda **kw: use_SVC(**kw),
    "NuSVC": lambda **kw: use_NuSVC(**kw),
    "LinearSVC": lambda **kw: use_LinearSVC(**kw),

    # Neighbors
    "KNeighborsClassifier": lambda **kw: use_KNeighborsClassifier(**kw),
    "RadiusNeighborsClassifier": lambda **kw: use_RadiusNeighborsClassifier(**kw),
    "NearestCentroid": lambda **kw: use_NearestCentroid(**kw),

    # Gaussian Process
    "GaussianProcessClassifier": lambda **kw: use_GaussianProcessClassifier(**kw),

    # Trees
    "DecisionTreeClassifier": lambda **kw: use_DecisionTreeClassifier(**kw),
    "ExtraTreeClassifier": lambda **kw: use_ExtraTreeClassifier(**kw),

    # Ensembles
    "AdaBoostClassifier": lambda **kw: use_AdaBoostClassifier(**kw),
    "BaggingClassifier": lambda **kw: use_BaggingClassifier(**kw),
    "ExtraTreesClassifier": lambda **kw: use_ExtraTreesClassifier(**kw),
    "GradientBoostingClassifier": lambda **kw: use_GradientBoostingClassifier(**kw),
    "HistGradientBoostingClassifier": lambda **kw: use_HistGradientBoostingClassifier(**kw),
    "RandomForestClassifier": lambda **kw: use_RandomForestClassifier(**kw),

    # Naive Bayes
    "BernoulliNB": lambda **kw: use_BernoulliNB(**kw),
    "CategoricalNB": lambda **kw: use_CategoricalNB(**kw),
    "ComplementNB": lambda **kw: use_ComplementNB(**kw),
    "GaussianNB": lambda **kw: use_GaussianNB(**kw),
    "MultinomialNB": lambda **kw: use_MultinomialNB(**kw),

    # Neural Networks
    "MLPClassifier": lambda **kw: use_MLPClassifier(**kw),

    # Semi-Supervised
    "LabelPropagation": lambda **kw: use_LabelPropagation(**kw),
    "LabelSpreading": lambda **kw: use_LabelSpreading(**kw),
    "SelfTrainingClassifier": lambda **kw: use_SelfTrainingClassifier(**kw),

    # Discriminant Analysis
    "LinearDiscriminantAnalysis": lambda **kw: use_LinearDiscriminantAnalysis(**kw),
    "QuadraticDiscriminantAnalysis": lambda **kw: use_QuadraticDiscriminantAnalysis(**kw),

    # Dummy
    "DummyClassifier": lambda **kw: use_DummyClassifier(**kw),
    "DummyClassifierMostFrequent": lambda **kw: use_DummyClassifierMostFrequent(**kw),
    "DummyClassifierPrior": lambda **kw: use_DummyClassifierPrior(**kw),
    "DummyClassifierStratified": lambda **kw: use_DummyClassifierStratified(**kw),
    "DummyClassifierUniform": lambda **kw: use_DummyClassifierUniform(**kw),
    "DummyClassifierConstant": lambda **kw: use_DummyClassifierConstant(**kw),

    # XGBoost
    "XGBClassifier": lambda **kw: use_XGBClassifier(**kw),
    "XGBRFClassifier": lambda **kw: use_XGBRFClassifier(**kw),
}


# =========================================================
# PREPROCESSOR ARGUMENT MAP
# =========================================================

COMMON_PREPROCESSOR_SETUP: dict[str, dict[str, Any]] = {

    # =====================================================
    # LINEAR MODELS
    # =====================================================

    "LogisticRegression": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "LogisticRegressionCV": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "PassiveAggressiveClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "Perceptron": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "RidgeClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "RidgeClassifierCV": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "SGDClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # SVM
    # =====================================================

    "SVC": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "NuSVC": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "LinearSVC": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # NEIGHBORS
    # =====================================================

    "KNeighborsClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "RadiusNeighborsClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "NearestCentroid": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # GAUSSIAN PROCESS
    # =====================================================

    "GaussianProcessClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # TREES
    # =====================================================

    "DecisionTreeClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "ExtraTreeClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    # =====================================================
    # ENSEMBLES
    # =====================================================

    "AdaBoostClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "BaggingClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "ExtraTreesClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "GradientBoostingClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "HistGradientBoostingClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "RandomForestClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    # =====================================================
    # NAIVE BAYES
    # =====================================================

    "BernoulliNB": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "linear_nn",
    },

    "CategoricalNB": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "linear_nn",
    },

    "ComplementNB": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "linear_nn",
    },

    "GaussianNB": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "MultinomialNB": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "linear_nn",
    },

    # =====================================================
    # NEURAL NETWORKS
    # =====================================================

    "MLPClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # SEMI-SUPERVISED
    # =====================================================

    "LabelPropagation": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "LabelSpreading": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "SelfTrainingClassifier": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # DISCRIMINANT ANALYSIS
    # =====================================================

    "LinearDiscriminantAnalysis": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    "QuadraticDiscriminantAnalysis": {
        "use_onehot": True,
        "use_scaling": True,
        "use_power": True,
        "model_type": "linear_nn",
    },

    # =====================================================
    # DUMMY
    # =====================================================

    "DummyClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "DummyClassifierMostFrequent": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "DummyClassifierPrior": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "DummyClassifierStratified": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "DummyClassifierUniform": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "DummyClassifierConstant": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    # =====================================================
    # XGBOOST
    # =====================================================

    "XGBClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    },

    "XGBRFClassifier": {
        "use_onehot": True,
        "use_scaling": False,
        "use_power": False,
        "model_type": "tree",
    }
}


# =========================================================
# EXAMPLE USAGE
# ================================

if __name__ == "__main__":
    model1 = use_LogisticRegression(C=0.5)
    model2 = use_RandomForestClassifier(n_estimators=500)
    model3 = use_XGBClassifier(max_depth=8)

    print(model1)
    print(model2)
    print(model3)
