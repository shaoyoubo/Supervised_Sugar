import math

class LambdaScheduler:
    def __init__(self, strategy='const', **kwargs):
        self.strategy = strategy
        self.params = kwargs

    def get_lambda(self, t):
        if self.strategy == 'linear':
            T = self.params.get('T', 6000)
            return max(1 - t / T, 0)
        elif self.strategy == 'exp':
            k = self.params.get('k', 5e-4)
            return math.exp(-k * t)
        elif self.strategy == 'sigmoid':
            a = self.params.get('a', 0.002)
            b = self.params.get('b', 3000)
            return 1 / (1 + math.exp(a * (t - b)))
        elif self.strategy == 'const':
            return 0.5
        else:
            raise ValueError("Unknown lambda schedule")
