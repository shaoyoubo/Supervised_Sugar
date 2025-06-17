import math

class LambdaScheduler:
    def __init__(self, strategy='const', **kwargs):
        self.strategy = strategy
        self.params = kwargs

    def get(self, t):
        if self.strategy == 'linear':
            start_t = self.params.get('start_t', 9000)
            end_t = self.params.get('end_t', 15000)
            duration = end_t - start_t
            lambda_t = max(0.0, min(1.0, (t - start_t) / duration))
            return {
                "sdf_estimation": 1 - lambda_t,
                "sdf_better_normal": 1 - lambda_t,
                "external_sdf_better_normal": lambda_t,
                "external_depth": lambda_t,
            }
        if self.strategy == 'linear2':
            start_t = self.params.get('start_t', 9000)
            end_t = self.params.get('end_t', 15000)
            duration = end_t - start_t
            lambda_t = max(0.0, min(1.0, (t - start_t) / duration))
            return {
                "sdf_estimation": lambda_t,
                "sdf_better_normal": lambda_t,
                "external_sdf_better_normal": 1 - lambda_t,
                "external_depth": 1 - lambda_t,
            }
        elif self.strategy == 'exp':
            k = self.params.get('k', 5e-4)
            start_t = self.params.get('start_t', 9000)
            lambda_t = math.exp(-k * (t - start_t))
            return {
                "sdf_estimation": lambda_t,
                "sdf_better_normal": lambda_t,
                "external_sdf_better_normal": 1 - lambda_t,
                "external_depth": 1 - lambda_t,
            }
        elif self.strategy == 'const_1':
            lambda_t = 1.0
            return {
                "sdf_estimation": 1 - lambda_t,
                "sdf_better_normal": 1 - lambda_t,
                "external_sdf_better_normal": lambda_t,
                "external_depth": lambda_t,
            }
        elif self.strategy == 'const_0':
            lambda_t = 0.0
            return {
                "sdf_estimation": 1 - lambda_t,
                "sdf_better_normal": 1 - lambda_t,
                "external_sdf_better_normal": lambda_t,
                "external_depth": lambda_t,
            }
        elif self.strategy == 'custom_0':
            if 9000 <= t < 14000:
                start_t = 9000
                end_t = 11000
                duration = end_t - start_t
                lambda_t = max(0.0, min(1.0, (t - start_t) / duration))
                return {
                    "sdf_estimation": 1 - lambda_t,
                    "sdf_better_normal": 1 - lambda_t,
                    "external_sdf_better_normal": lambda_t,
                    "external_depth": lambda_t,
                }
            else:
                return {
                    "sdf_estimation": 0,
                    "sdf_better_normal": 1,
                    "external_sdf_better_normal": 0,
                    "external_depth": 0,
                }
        elif self.strategy == 'custom_1':
            if 9000 <= t < 13000:
                start_t = 9000
                end_t = 11000
                duration = end_t - start_t
                lambda_t = max(0.0, min(1.0, (t - start_t) / duration))
                return {
                    "sdf_estimation": 1 - lambda_t,
                    "sdf_better_normal": 1 - lambda_t,
                    "external_sdf_better_normal": lambda_t,
                    "external_depth": lambda_t,
                }
            else:
                return {
                    "sdf_estimation": 1,
                    "sdf_better_normal": 1,
                    "external_sdf_better_normal": 1,
                    "external_depth": 1,
                }
        elif self.strategy == 'custom_2':
            if 9000 <= t < 13000:
                start_t = 9000
                end_t = 11000
                duration = end_t - start_t
                lambda_t = max(0.0, min(1.0, (t - start_t) / duration))
                return {
                    "sdf_estimation": 1 - lambda_t,
                    "sdf_better_normal": 1 - lambda_t,
                    "external_sdf_better_normal": lambda_t,
                    "external_depth": lambda_t,
                }
            else:
                return {
                    "sdf_estimation": 0,
                    "sdf_better_normal": 1,
                    "external_sdf_better_normal": 1.5,
                    "external_depth": 0,
                }
        else:
            raise ValueError("Unknown lambda schedule")