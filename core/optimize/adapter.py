

class OptimizerAdapter: 
    def __init__(
        self, 
        optimizer = None,
        objective = None,
    ):
        self.optimizer = optimizer 
        self.objective = objective

    def step(self): 
        candidate = self.optimizer.ask(n = 1)[0]
        score = self.objective(list(candidate.values()))
        self.optimizer.tell([score])
        return score