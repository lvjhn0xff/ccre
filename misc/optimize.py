from core.optimize.adapter import OptimizerAdapter 
from gradient_free_optimizers.ask_tell import HillClimbingOptimizer
from core.optimize.objectives.registry import *
from utils.objective import Objective
import numpy as np


search_space = {
    0 : np.linspace(-10, 10, 100), 
    1 : np.linspace(-10, 10, 100)
}

objective = Objective(
    handler = OBJECTIVES["Rastrigin"],  
    track_k = 10, 
    use_log = True
)

initial_candidate = { 0 : 0, 1 : 0 }
values = list(initial_candidate.values())

optimizer = HillClimbingOptimizer(
    search_space = search_space, 
    initial_evaluations = [
        (initial_candidate, objective(values))
    ]
)

adapter = OptimizerAdapter(
    optimizer=optimizer,
    objective=objective
)


for i in range(100): 
    print(f"# --- Running iteration #{i}")
    adapter.step()


print(objective.log)