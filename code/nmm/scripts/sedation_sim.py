import pickle
import time
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from root_path_util import ROOT_DIR
from models.builder import ModelBuilder, ExtendedTemplate
from runner.model_runner import ModelRunner, InputFunction


class SedationSim:
    def __init__(self, run_name: str, template: ExtendedTemplate = None):
        self.results = None
        self.model = template if template is not None else ModelBuilder.build_david_friston_default()

        self.run_name = run_name
        self.runner = ModelRunner(self.model)
        self.sim_class = 'sedation_sim'
        self.base_path = Path(ROOT_DIR, 'thesis', 'data', self.sim_class)

    def save(self):
        Path(self.base_path, 'raw_data').mkdir(parents=True, exist_ok=True)
        with open(Path(self.base_path, 'raw_data', f'{self.run_name}.pkl'), 'wb') as file:
            pickle.dump(
                {
                    'results': self.results,
                    'factors': self.factor_series
                },
                file)

    def run(self, seconds, factor_series, cut=1.0, input_noise=InputFunction.normal):
        self.factor_series = factor_series
        t = time.time()
        inputs = {
            'PC/RPO_i/tau_fac_0': factor_series,
            'PC/RPO_i/tau_fac_1': factor_series,
        }
        self.results = self.runner.run(seconds + cut, cut=cut, inputs=inputs, input_noise=input_noise)
        print(f'simulating {seconds + cut}s took {time.time() - t} seconds')
        self.save()

    def gen_factors(self, sim_time_secs, initial_cutoff, start=1.0, middle=1.7, step=0.01,
                    init_weight=35, middle_weight=40):
        factor_series = [*[start] * init_weight,
                         *np.arange(start, middle + step, step),
                         *[middle] * middle_weight,
                         *np.flip(np.arange(start, middle, step)),
                         *[start] * init_weight]

        lf = len(factor_series)

        f_ser = np.interp(np.linspace(1, lf, sim_time_secs * 1000), range(lf), factor_series)

        f_ser = np.pad(f_ser, (int(initial_cutoff * 1000), 0), mode='constant', constant_values=(start,))
        return f_ser


def run_sedation(seconds, template, name):
    sedation = SedationSim(f'{name}_SEDATION_{seconds}', template=template)
    factor_series = sedation.gen_factors(seconds, 1.0, start=1.0, middle=3.0, init_weight=20)
    sedation.run(seconds, factor_series)
    plt.plot(sedation.results)
    plt.show()


def jr_sedation(seconds):
    run_sedation(seconds, template=ModelBuilder.build_jansen_rit_default(), name='JR')


def df_sedation(seconds):
    run_sedation(seconds, template=ModelBuilder.build_david_friston_default(), name='DF')


if __name__ == '__main__':
    jr_sedation(150)
    df_sedation(150)