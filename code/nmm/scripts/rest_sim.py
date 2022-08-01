import pickle
import time
from matplotlib import pyplot as plt
from pathlib import Path

from root_path_util import ROOT_DIR
from models.builder import ModelBuilder
from runner.model_runner import ModelRunner, InputFunction


class RestSim:
    def __init__(self, run_name:str, model_builder=ModelBuilder.build_david_friston_default):
        self.results = None
        self.model = model_builder()
        self.run_name = run_name
        self.runner = ModelRunner(self.model)
        self.sim_class = 'rest_sim'
        self.base_path = Path(ROOT_DIR, 'thesis', 'data', self.sim_class)

    def save(self):
        Path(self.base_path, 'raw_data').mkdir(parents=True, exist_ok=True)
        with open(Path(self.base_path, 'raw_data', f'{self.run_name}.pkl'), 'wb') as file:
            pickle.dump(self.results, file)

    def run(self, seconds, cut=1.0, input_noise=InputFunction.normal):
        t = time.time()
        self.results = self.runner.run(seconds + cut, cut=cut, input_noise=input_noise)
        print(f'simulating {seconds + cut}s took {time.time() - t} seconds')
        self.save()


def run_rest(seconds, model_builder, name):
    rest = RestSim(f'{name}_REST_{seconds}', model_builder=model_builder)

    rest.run(seconds)
    plt.plot(rest.results)
    plt.show()


def jr_rest(seconds):
    run_rest(seconds, model_builder=ModelBuilder.build_jansen_rit_default, name='JR')


def df_rest(seconds):
    run_rest(seconds, model_builder=ModelBuilder.build_david_friston_default, name='DF')


if __name__ == '__main__':
    jr_rest(60)
    df_rest(60)



