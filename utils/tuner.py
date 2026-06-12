import optuna

class TunerAdapter: 
    def __init__(self, objective, direction="mazimize"): 
        self.objective = objective 
        self.study = optuna.create_study(direction=direction)
        
    def optimize(self, n_trials=100): 
        self.study.optimize(self.objective, n_trials)
    
