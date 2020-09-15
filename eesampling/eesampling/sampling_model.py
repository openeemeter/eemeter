import pandas as pd 

class SamplingModel:
    """Abstract Sampling Model class.  Child classes must implement __init__, _fit, _predict."""

    def __init__(self):
        pass

    def _check_inputs(self):
        if self.X_treatment is not None:
            if type(self.X_treatment) != pd.DataFrame:
                raise ValueError("X_treatment must be a pandas DataFrame with one row per meter.")

        if type(self.X_pool) != pd.DataFrame:
            raise ValueError("X_pool must be a pandas DataFrame with one row per meter.")

        if len(self.X_pool) < self.n_outputs:
            raise ValueError(str(self.n_outputs) + " outputs requested, but only " + str(len(self.X_pool)) + " available in pool.")


    def sample(self, X_treatment, X_pool, n_outputs):
        self.fitted = False
        self.X_treatment = X_treatment
        self.X_pool = X_pool
        self.n_outputs = n_outputs
        self._check_inputs()

        self._fit(self.X_treatment)
        self.X_sample = self._predict(self.X_pool, n_outputs)
        return self.X_sample

    def _fit(self, X_treament):
        pass

    def _predict(self, X_pool, n_outputs):
        pass

    def diagnostics(self):
        pass


class RandomSamplingModel(SamplingModel):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def sample(self, X_pool, n_outputs):
        return super().sample(X_treatment=None, X_pool=X_pool, n_outputs=n_outputs)

    def _predict(self, X_pool, n_outputs):
        return X_pool.sample(n_outputs, random_state=self.random_seed)
