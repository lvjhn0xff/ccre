from utils.datasets import use_dataset 
from core.classifiers.common.templates import *

from .base import CrossFoldExperiment

# --- PARAMETERS ------------------------------------------------------------ # 

DATASET             = "wdbc"
DATASET_VERSION     = 1
SEED                = 42
SPLITS              = 10 
CLASSIFIER          = "MLPClassifier"
PREPROCESSOR_ARGS   = COMMON_PREPROCESSOR_SETUP[CLASSIFIER]

# --------------------------------------------------------------------------- #

# Create experiment.
experiment = CrossFoldExperiment(
    dataset = DATASET, 
    dataset_version = DATASET_VERSION, 
    seed = SEED,
    model = MODEL_FACTORIES[CLASSIFIER],
    preprocessor_args = PREPROCESSOR_ARGS
)

# Display information about ex
experiment.display_info()

# Execute experiment. 
experiment.execute()