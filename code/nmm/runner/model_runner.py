import numpy as np

from models.builder import ExtendedTemplate
from models.components import PYRATES_NEW_VERSION


class InputFunction:

    @staticmethod
    def normal(loc=220.0, scale=22.0):
        return lambda size: np.random.normal(loc=loc, scale=scale, size=size)

    @staticmethod
    def uniform(low=120.0, high=320.0):
        return lambda size: np.random.uniform(low=low, high=high, size=size)

    @staticmethod
    def constant(value=220.0):
        return lambda size: np.tile([[value]], size[0])


class ModelRunner:

    def __init__(self, template: ExtendedTemplate):
        self.template = template
        if PYRATES_NEW_VERSION:
            self.circuit = self.template.template
        else:
            self.circuit = self.template.template.apply().compile()
        self.state_vars = None

    if PYRATES_NEW_VERSION:
        def run(self, seconds: float, step_size: float = 1e-3, sampling_rate: float = 1000.0,
                inputs=None):
            if inputs is None:
                inputs = {}
            input_size = (int(np.round(seconds / step_size, decimals=0)), 1)
            res = self.circuit.run(simulation_time=seconds, step_size=step_size,
                                   inputs={self.template.input_str: InputFunction.normal()(input_size),
                                           **inputs},
                                   sampling_step_size=1.0 / sampling_rate, solver='scipy', method='RK45',
                                   outputs=self.template.outputs, in_place=False)
            self.state_vars = self.circuit
            return self.template.evaluate_result(res, {})
    else:
        def run(self, seconds: float, step_size: float = 1e-3, sampling_rate: float = 1000.0):
            input_size = (int(np.round(seconds / step_size, decimals=0)), 1)
            res = self.circuit.run(simulation_time=seconds, step_size=step_size,
                                   inputs={self.template.input_str: InputFunction.normal()(input_size)},
                                   sampling_step_size=1.0 / sampling_rate, solver='scipy', backend='numpy',
                                   outputs=self.template.outputs, )
            return self.template.evaluate_result(res, {})

    def sep(self):
        ...

    # def run_old(self, seconds: float, step_size: float = 1e-3):
    #     input_size = (int(np.round(seconds / step_size, decimals=0)), 1)
    #     args = self.args
    #     # args2 = (*args[:38], InputFunction.normal()(input_size), *args[-2:])
    #     # return self.func(*args2)
    #     return self.func(*args)
