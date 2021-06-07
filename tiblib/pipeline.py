from typing import List

from .blueprint import Pipe
from .blueprint import Faucet


class Pipeline:
    def __init__(self, step_list):
        self.steps = step_list

    def fit(self, x, y):
        temp_x = x
        for step in self.steps:
            if isinstance(step, Pipe):
                temp_x = step.fit_transform(x, y)
            if isinstance(step, Faucet):
                step.fit(temp_x, y)

    def predict(self, x, return_prob=False):
        temp_x = x
        for step in self.steps:
            if isinstance(step, Pipe):
                temp_x = step.transform(temp_x)
            if isinstance(step, Faucet):
                return step.predict(temp_x, return_prob)
        return temp_x

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x, y)