from pathlib import Path

from matplotlib import pyplot as plt

from utils import ROOT_DIR, write_results_to_file
from models.builder import ModelBuilder
from runner.model_runner import ModelRunner


class Methodology:
    def __init__(self):
        self.model = ModelBuilder.build_jansen_rit_default()
        self.runner = ModelRunner(self.model)


if __name__ == '__main__':
    methodology = Methodology()
    results = methodology.runner.run(3.0)
    write_results_to_file(results, Path(ROOT_DIR, 'thesis', 'data', 'methodology', 'uncut.csv'))
    plt.plot(results)
    plt.show()