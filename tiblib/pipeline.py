from typing import List

from .blueprint import Pipe
from .blueprint import Faucet


class Pipeline:
    def __init__(self, step_list: List):
        self.steps = step_list

    def add_step(self, step):
        self.steps.append(step)

    def rem_step(self):
        self.steps.pop(-1)

    def fit(self, x, y):
        temp_x = x
        for step in self.steps:
            if isinstance(step, Pipe):
                temp_x = step.fit_transform(temp_x, y)
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

    def __str__(self):
        return ''.join([i.__str__() + "->" for i in self.steps])[:-2]
