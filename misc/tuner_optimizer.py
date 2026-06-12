from core.optimize.adapter import OptimizerAdapter
from core.optimize.objectives.registry import *
from core.optimize.params import *
from gradient_free_optimizers.ask_tell import *
from utils.objective import Objective
import numpy as np 


objective = Objective(
    handler = OBJECTIVES["Michalewicz"],  
    track_k = 10, 
    use_log = True
)

search_space = {
    0 : (0, 3.141592654), 
    1 : (0, 3.141592654)
}

initial_candidate = { 0 : 0, 1 : 0 }
values = list(initial_candidate.values())

def maximize(trial):
    opt_space = OPTIMIZER_SPACES["CMAESOptimizer"](trial)


    adapter = OptimizerAdapter(
        optimizer = CMAESOptimizer(
            search_space,
            initial_evaluations = [
                (initial_candidate, objective(values)) 
            ] * 100,
            **opt_space
        ),
        objective = objective
    )
    
    score = None
    for i in range(1000): 
        # print(f"# --- Running iteration #{i}")
        score = adapter.step()

    return score

from utils.tuner import TunerAdapter
tuner = TunerAdapter(maximize, direction="minimize")
tuner.optimize(n_trials=100)