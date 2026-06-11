from core.optimize.objectives.registry import * 
from utils.objective import Objective 
import numpy as np

objective_fn = OBJECTIVES["Rastrigin"]
objective = Objective(
    handler=objective_fn, 
    track_k=10,
    use_log=True
)

for i in range(30): 
    objective.call(np.random.randn(10)) 

print(objective.log)