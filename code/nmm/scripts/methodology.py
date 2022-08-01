from pathlib import Path

from matplotlib import pyplot as plt

from root_path_util import ROOT_DIR
from utils.analysis import PostProcessing
from utils.files import write_results_to_file, write_signal_to_file
from models.builder import ModelBuilder
from runner.model_runner import ModelRunner


class Methodology:
    def __init__(self):
        # Jansen-Rit (1995) Model
        self.jr_model = ModelBuilder.build_jansen_rit_default()
        self.jr_runner = ModelRunner(self.jr_model)
        # David-Friston (2003) Model
        self.df_model = ModelBuilder.build_david_friston_default()
        self.df_runner = ModelRunner(self.df_model)

    def initial_oscillations(self, down_sample_factor=10, plot=True):
        results = self.jr_runner.run(7.0)
        idx, sig = write_results_to_file(results, Path(ROOT_DIR, 'thesis', 'data', 'methodology', 'uncut.csv'),
                                         down_sample_factor=down_sample_factor)
        if plot:
            plt.plot(results)
            plt.figure()
            if down_sample_factor != 1:
                plt.plot(idx, sig)
                plt.show()

    def psd_plot_(self, model, seconds):
        if model == 'jr':
            runner = self.jr_runner
        elif model == 'df':
            runner = self.df_runner
        else:
            raise Exception(f'{model} is not allowed. must be either `df` or `jr`')
        results = runner.run(seconds, cut=5.0)
        idx, sig = write_results_to_file(results=results, path=Path(ROOT_DIR, 'thesis', 'data', 'methodology',
                                                                    f'psd_data_{model}_{seconds}.csv'),
                                         down_sample_factor=10)
        plt.figure('signal')
        plt.plot(idx, sig)
        plt.show()
        for method in ['welch', 'fft', 'multitaper']:
            func = getattr(PostProcessing, method)
            frequencies, filtered = func(results, None, fs=1000, freq_range=(0, 30))
            write_signal_to_file(index=frequencies, signal=filtered,
                                 path=Path(ROOT_DIR, 'thesis', 'data',  'methodology',
                                           f'psd_{method}_{model}_{seconds}.csv'))
            plt.figure(method)
            plt.plot(frequencies, filtered)
            plt.show()


if __name__ == '__main__':
    methodology = Methodology()
    methodology.initial_oscillations()
    methodology.psd_plot_('jr', 35)
    methodology.psd_plot_('df', 35)
