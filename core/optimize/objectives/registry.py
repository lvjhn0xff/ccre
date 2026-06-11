import os 

from utils.helpers import autoload_folder
from core.optimize.templates import *

def register_objective(name, module): 
    Objective = getattr(module, name)
    OBJECTIVES[name] = Objective()

autoload_folder("unimodal", register_objective)
autoload_folder("multimodal", register_objective) 



