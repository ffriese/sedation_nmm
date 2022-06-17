import time
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from root_path_util import ROOT_DIR
from utils.files import write_results_to_file, write_signal_to_file
from models.builder import ModelBuilder
from runner.model_runner import ModelRunner, InputFunction


class SedationSim:
    def __init__(self, model_builder=ModelBuilder.build_david_friston_default):
        self.model = model_builder()
        self.runner = ModelRunner(self.model)

    def gen_factors(self, sim_time_secs, initial_cutoff, start=1.0, middle=1.7, step=0.01):
        factor_series = [*[start] * 15,
                         *np.arange(start, middle + step, step),
                         *[middle] * 60,
                         *np.flip(np.arange(start, middle, step)),
                         *[start] * 15]

        lf = len(factor_series)

        f_ser = np.interp(np.linspace(1, lf, sim_time_secs * 1000), range(lf), factor_series)

        f_ser = np.pad(f_ser, (int(initial_cutoff * 1000), 0), mode='constant', constant_values=(start,))
        return f_ser

    def generate_mirco_molar_timecourse(self):
        ...

    def convert_to_lambda(self, micro_molars: np.array):
        """
        derived from exponential regression on data-points from McDougal et al. (2008).
            DOI: 10.1016/j.neuropharm.2007.11.001
         (visually extracted from plots:)

           0.0 uM -> 1.0
           0.3 uM -> 1.1
           1.0 uM -> 1.5
           3.0 uM -> 1.8
           10  uM -> 2.8
           30  uM -> 5.5

          =>  factor ~= exp(0.063*uM)
        """
        return np.exp(0.063 * micro_molars)

    def plot_histogram(self, data, run_name, freq_Hz = 1000, NFFT=2048, backend='Qt5Agg',
                       vmin=None, vmax=None):
        matplotlib.use(backend)
        if backend == 'pgf':
            matplotlib.rcParams.update(
                {
                    # Adjust to your LaTex-Engine
                    "pgf.texsystem": "pdflatex",
                    "font.family": "serif",
                    "text.usetex": True,
                    "pgf.rcfonts": False,
                    "axes.unicode_minus": False,
                }
            )

        import pylab
        plt.specgram(data, NFFT=NFFT, noverlap=NFFT - int(0.25 * freq_Hz), Fs=freq_Hz, scale='dB',
                     # vmin=-111, vmax=-81,  # highlights stable states
                     # vmin=-100, vmax=-42,  # highlights unstable states
                     cmap=pylab.cm.nipy_spectral, interpolation='bicubic')
        plt.axis(ymin=0, ymax=40)

        if backend == 'pgf':
            plt.savefig(Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'{run_name}.pgf'))


if __name__ == '__main__':
    sedation = SedationSim(model_builder=ModelBuilder.build_jansen_rit_default)

    seconds = 65
    cut = 1.0
    f_ser = sedation.gen_factors(seconds, cut, start=1.0,
                                 middle=3.0)

    run_name = 'jr_simple'

    inputs = {
        'PC/RPO_i/tau_fac_0': f_ser,
        'PC/RPO_i/tau_fac_1': f_ser,
    }
    t = time.time()
    plt.plot(f_ser)
    plt.show()
    results = sedation.runner.run(seconds + cut, inputs=inputs, cut=cut, input_noise=InputFunction.normal)
    print(f'simulating {seconds + cut}s took {time.time() - t} seconds')
    idx, sig = write_results_to_file(results, Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'{run_name}.csv'),
                                     down_sample_factor=10)
    plt.figure()
    plt.plot(results)
    plt.show()

    factor_timeline = np.array([i for idx, i in enumerate(f_ser[int(cut * 1000):]) if idx % 10 == 0])

    write_signal_to_file(idx, factor_timeline,
                         Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'{run_name}_factors.csv')
                         )
    half = int(len(sig) / 2)

    plt.figure()
    plt.plot(factor_timeline[:half], sig[:half])
    plt.show()
    plt.figure()
    plt.plot(factor_timeline[half:], sig[half:])
    plt.show()



    import tikzplotlib

    # plt.figure(figsize=(14, 8), dpi=300)
    # plt.specgram(results, NFFT=NFFT, noverlap=NFFT - int(0.25 * fs_Hz), Fs=fs_Hz, scale='dB',
    #              #vmin=-111, vmax=-81,
    #              cmap=pylab.cm.nipy_spectral, interpolation='bicubic')
    # plt.axis(ymin=0, ymax=40)
    sedation.plot_histogram(results, run_name, backend='pgf')
    sedation.plot_histogram(results, run_name)

    #
    # plt.xlabel("$x_1$")  # Latex commands can be used
    # plt.ylabel("$x_2$")
    # # Insert width and height from \printsizes
    # # A factor can be used to create some whitespace
    # factor = 0.9
    # plt.gcf().set_size_inches(3.43745 * factor, 3.07343 * factor)
    # # Fixes cropped labels
    # plt.tight_layout()
    # # Save as pgf
    # plt.savefig(Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'{run_name}.pgf'))

    # tikzplotlib.save(Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'{run_name}.tex'))
    #
    # plt.savefig(Path(ROOT_DIR, 'thesis', 'data', 'full_sedation_sim', f'____{run_name}.png'), format='png',
    #             bbox_inches='tight', pad_inches=0)

    #matplotlib.use("Qt5Agg")
    plt.show()
