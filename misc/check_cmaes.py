import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from joblib import Parallel, delayed
from utils.datasets import use_dataset
import cma


# =========================================================
# CONFIG
# =========================================================
DATASET_NAME = "wdbc"
FOLDS = 10
MAX_EVALS = 3000   # reduced for speed (adjust if needed)
POP_SIZE = 20      # faster than 40
SEED = 42


# =========================================================
# LOAD DATASET
# =========================================================
print("Loading dataset...")
X, y, make_preprocessor = use_dataset(DATASET_NAME)

preprocessor = make_preprocessor(
    use_onehot=True,
    use_scaling=True,
    use_power=True,
    model_type="cma"
)

X = preprocessor.fit_transform(X).astype(np.float32)

# binary conversion
y = (y == list(set(y))[0]).astype(np.int8)


# =========================================================
# PRECOMPUTE FUNCTIONS (VERY IMPORTANT SPEEDUP)
# =========================================================
print("Precomputing nonlinear transforms...")

n_samples, n_features = X.shape

F_all = np.zeros((4, n_samples, n_features), dtype=np.float32)

F_all[0] = X
F_all[1] = np.sin(X)
F_all[2] = np.cos(X)
F_all[3] = np.tanh(X)


# =========================================================
# MODEL
# =========================================================
def model_predict_fast(X, F, params):
    n_samples, n_features = X.shape
    out = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_features):
        base = i * 3

        op_id = int(np.clip(np.round(params[base]), 0, 2))
        weight = params[base + 1]
        func_id = int(np.clip(np.round(params[base + 2]), 0, 3))

        fx = F[func_id, :, i]

        if op_id == 0:
            out += weight + fx
        elif op_id == 1:
            out += weight - fx
        else:
            out += weight * fx

    return (out > 0).astype(np.int8)


# =========================================================
# FITNESS
# =========================================================
def fitness_fast(params, X, F, y, n_features):
    y_pred = model_predict_fast(X, F, params)
    return 1.0 - balanced_accuracy_score(y, y_pred)


# =========================================================
# CROSS VALIDATION
# =========================================================
print("Starting CMA-ES cross-validation...")

cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

dim = n_features * 3
scores = []


for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"\n--- FOLD {fold + 1} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    F_train = F_all[:, train_idx, :]
    F_test = F_all[:, test_idx, :]

    es = cma.CMAEvolutionStrategy(
        np.zeros(dim, dtype=np.float32),
        0.5,
        {
            "popsize": POP_SIZE,
            "seed": SEED
        }
    )

    best_params = None
    best_fit = np.inf
    evals = 0

    while evals < MAX_EVALS:
        solutions = es.ask()

        # =====================================================
        # PARALLEL FITNESS (MAJOR SPEEDUP)
        # =====================================================
        fits = Parallel(n_jobs=-1)(
            delayed(fitness_fast)(s, X_train, F_train, y_train, n_features)
            for s in solutions
        )

        es.tell(solutions, fits)
        evals += len(solutions)

        idx = np.argmin(fits)
        if fits[idx] < best_fit:
            best_fit = fits[idx]
            best_params = solutions[idx].copy()

        if es.stop():
            break

        if evals % 200 == 0:
            print(f"Eval {evals:05d} | Train BA: {1 - best_fit:.4f}")

    # =====================================================
    # TEST EVALUATION
    # =====================================================
    y_pred = model_predict_fast(X_test, F_test, best_params)
    score = f1_score(y_test, y_pred)

    scores.append(score)

    print(f"Fold {fold + 1} Test Balanced Accuracy: {score:.4f}")


# =========================================================
# FINAL RESULT
# =========================================================
print("\n==============================")
print("CMA-ES FINAL RESULTS")
print(f"Mean Balanced Accuracy: {np.mean(scores):.4f}")
print(f"Std Dev              : {np.std(scores):.4f}")
print("==============================")