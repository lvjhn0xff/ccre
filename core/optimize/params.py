# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

NORMALIZERS = {
    "sampling": {
        "random_10k": {"random": 10_000},
        "random_100k": {"random": 100_000},
        "random_1m": {"random": 1_000_000},
    },
    "tree_para": {
        "rf_100": {"n_estimators": 100},
        "rf_150": {"n_estimators": 150},
        "rf_200": {"n_estimators": 200},
        "rf_100_d10": {"n_estimators": 100, "max_depth": 10},
        "rf_150_d15": {"n_estimators": 150, "max_depth": 15},
    },
}


def normalize(name: str, value):
    """
    Convert Optuna-safe string values back into the dictionaries
    expected by the optimizer implementations.
    """
    return NORMALIZERS.get(name, {}).get(value, value)


# =============================================================================
# OPTIMIZER SEARCH SPACES
# =============================================================================

OPTIMIZER_SPACES = {

    # ==================== LOCAL SEARCH ALGORITHMS ====================

    "HillClimbingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "StochasticHillClimbingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "p_accept": trial.suggest_float("p_accept", 0.0, 0.5),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "RepulsingHillClimbingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "repulsion_factor": trial.suggest_float(
            "repulsion_factor",
            2.0,
            10.0
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "SimulatedAnnealingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "start_temp": trial.suggest_float("start_temp", 0.5, 10.0),
        "annealing_rate": trial.suggest_float(
            "annealing_rate",
            0.90,
            0.999
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "DownhillSimplexOptimizer": lambda trial: {
        "alpha": trial.suggest_float("alpha", 0.8, 1.2),
        "gamma": trial.suggest_float("gamma", 1.5, 2.5),
        "beta": trial.suggest_float("beta", 0.3, 0.7),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    # ==================== GLOBAL SEARCH ALGORITHMS ====================

    "RandomSearchOptimizer": lambda trial: {},

    "GridSearchOptimizer": lambda trial: {
        "step_size": trial.suggest_int("step_size", 1, 5),
        "direction": trial.suggest_categorical(
            "direction",
            ["diagonal", "orthogonal"]
        ),
    },

    "RandomRestartHillClimbingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "n_iter_restart": trial.suggest_int("n_iter_restart", 5, 50),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "RandomAnnealingOptimizer": lambda trial: {
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.5, log=True),
        "distribution": trial.suggest_categorical(
            "distribution",
            ["normal", "laplace", "logistic"]
        ),
        "n_neighbours": trial.suggest_int("n_neighbours", 1, 16),
        "start_temp": trial.suggest_float("start_temp", 1.0, 20.0),
        "annealing_rate": trial.suggest_float(
            "annealing_rate",
            0.95,
            0.999
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "PatternSearch": lambda trial: {
        "n_positions": trial.suggest_int("n_positions", 4, 16),
        "pattern_size": trial.suggest_float("pattern_size", 0.1, 0.5),
        "reduction": trial.suggest_float("reduction", 0.5, 0.95),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "PowellsMethod": lambda trial: {
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    # Slow
    "LipschitzOptimizer": lambda trial: {
        "sampling": normalize(
            "sampling",
            trial.suggest_categorical(
                "sampling",
                list(NORMALIZERS["sampling"].keys())
            )
        ),
        "max_sample_size": trial.suggest_int(
            "max_sample_size",
            10000,
            10000000,
            log=True
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "DirectAlgorithm": lambda trial: {
        "sampling": normalize(
            "sampling",
            trial.suggest_categorical(
                "sampling",
                list(NORMALIZERS["sampling"].keys())
            )
        ),
        "max_sample_size": trial.suggest_int(
            "max_sample_size",
            10000,
            10000000,
            log=True
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    # ==================== POPULATION-BASED ALGORITHMS ====================

    "ParticleSwarmOptimizer": lambda trial: {
        "population": trial.suggest_int("population", 5, 50),
        "inertia": trial.suggest_float("inertia", 0.3, 0.9),
        "cognitive_weight": trial.suggest_float(
            "cognitive_weight",
            0.1,
            2.0
        ),
        "social_weight": trial.suggest_float(
            "social_weight",
            0.1,
            2.0
        ),
        "temp_weight": trial.suggest_float("temp_weight", 0.1, 1.0),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "SpiralOptimization": lambda trial: {
        "population": trial.suggest_int("population", 5, 50),
        "decay_rate": trial.suggest_float("decay_rate", 0.95, 0.999),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "ParallelTemperingOptimizer": lambda trial: {
        "population": trial.suggest_int("population", 4, 20),
        "n_iter_swap": trial.suggest_int("n_iter_swap", 5, 50),
    },

    "GeneticAlgorithmOptimizer": lambda trial: {
        "population": trial.suggest_int("population", 10, 100),
        "offspring": trial.suggest_int("offspring", 10, 100),
        "mutation_rate": trial.suggest_float(
            "mutation_rate",
            0.1,
            0.9
        ),
        "crossover_rate": trial.suggest_float(
            "crossover_rate",
            0.5,
            1.0
        ),
        "crossover": trial.suggest_categorical(
            "crossover",
            ["discrete-recombination"]
        ),
        "n_parents": trial.suggest_int("n_parents", 2, 4),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "EvolutionStrategyOptimizer": lambda trial: {
        "population": trial.suggest_int("population", 5, 50),
        "offspring": trial.suggest_int("offspring", 10, 100),
        "mutation_rate": trial.suggest_float(
            "mutation_rate",
            0.5,
            0.9
        ),
        "crossover_rate": trial.suggest_float(
            "crossover_rate",
            0.1,
            0.5
        ),
        "replace_parents": trial.suggest_categorical(
            "replace_parents",
            [True, False]
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "DifferentialEvolutionOptimizer": lambda trial: {
        "population": trial.suggest_int("population", 10, 50),
        "mutation_rate": trial.suggest_float(
            "mutation_rate",
            0.5,
            1.0
        ),
        "crossover_rate": trial.suggest_float(
            "crossover_rate",
            0.7,
            1.0
        ),
        "rand_rest_p": trial.suggest_float("rand_rest_p", 0.0, 0.1),
    },

    "CMAESOptimizer": lambda trial: (
        lambda population: {
            "population": population,
            "mu": trial.suggest_int("mu", 2, population),
            "sigma": trial.suggest_float(
                "sigma",
                0.1,
                0.5,
                log=True,
            ),
            "ipop_restart": trial.suggest_categorical(
                "ipop_restart",
                [True, False],
            ),
            "rand_rest_p": trial.suggest_float(
                "rand_rest_p",
                0.0,
                0.1,
            ),
        }
    )(trial.suggest_int("population", 4, 50)),

    # ==================== SMBO ====================

    # Slow
    "BayesianOptimizer": lambda trial: {
        "xi": trial.suggest_float("xi", 0.01, 0.1, log=True),
    },

    # Slow
    "TreeStructuredParzenEstimators": lambda trial: {
        "gamma_tpe": trial.suggest_float("gamma_tpe", 0.1, 0.3)
    },

    # Slow
    "ForestOptimizer": lambda trial: {
        "xi": trial.suggest_float("xi", 0.01, 0.1, log=True),
        "tree_regressor": trial.suggest_categorical(
            "tree_regressor",
            ["extra_tree", "random_forest", "gradient_boost"]
        ),

        "tree_para": normalize(
            "tree_para",
            trial.suggest_categorical(
                "tree_para",
                list(NORMALIZERS["tree_para"].keys())
            )
        ),
    },
}