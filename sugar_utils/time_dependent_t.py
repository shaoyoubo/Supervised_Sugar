import math

class LambdaScheduler:
    def __init__(self, strategy='const', **kwargs):
        self.strategy = strategy
        self.params = kwargs

    def get_lambda(self, t):
        if self.strategy == 'linear':
            start_t = self.params.get('start_t', 9000)
            end_t = self.params.get('end_t', 15000)
            duration = end_t - start_t
            return max(0.0, min(1.0, (end_t - t) / duration))
        elif self.strategy == 'exp':
            k = self.params.get('k', 5e-4)
            start_t = self.params.get('start_t', 9000)
            return math.exp(-k * (t - start_t))
        elif self.strategy == 'sigmoid':
            midpoint = self.params.get('midpoint', 12000)  # Midpoint of transition
            steepness = self.params.get('steepness', 0.001)  # Controls how sharp the transition is
            return 1 / (1 + math.exp(steepness * (t - midpoint)))
        elif self.strategy == 'const_1':
            return 1.0
        elif self.strategy == 'const_0':
            return 0.0
        else:
            raise ValueError("Unknown lambda schedule")