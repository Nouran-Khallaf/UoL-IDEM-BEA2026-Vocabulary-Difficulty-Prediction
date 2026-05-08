from __future__ import annotations


class BaseModelRunner:
    model_name = "base"

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
