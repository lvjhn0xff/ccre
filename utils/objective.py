from .heap import IndexHeap


class Objective: 
    def __init__(
        self, 
        handler = None, 
        context = None, 
        track_k = None,
        use_log = False, 
        log_interval = 1
    ): 

        # Handler class for the objective.
        self.handler = handler

        # Context data. 
        self.context = context 

        # Register context. 
        self.handler.scope = self

        # Heap data. 
        self.solutions = IndexHeap(track_k, order="min")

        # History of calls. 
        self.log = []

        # Determine whether to track history of scores. 
        self.use_log = use_log

        # Function evaluation counts. 
        self.evals = 0

        # Log interval. 
        self.log_interval = log_interval

    def call(self, solution): 
        # Get score.
        score = self.handler(solution)

        # Log score to history.
        if self.use_log and self.evals % self.log_interval == 0:
            self.log.append(score)

        # Register solution to heap.
        self.solutions.push(self.evals, tuple(solution), score)

        # Register function evaluation.
        self.evals += 1

        return score

