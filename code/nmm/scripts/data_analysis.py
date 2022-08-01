import pickle
from collections import OrderedDict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from root_path_util import ROOT_DIR
from utils.files import write_results_to_file, write_signal_to_file, write_multiple_lines_to_file


class DataAnalysis:

    def __init__(self, run_name, simulation_class='full_sedation_sim'):
        self.factors = None
        self.raw_results = None
        self.results = None
        self.run_name = run_name
        self.sim_class = simulation_class
        self.base_path = Path(ROOT_DIR, 'thesis', 'data', self.sim_class)
        self.load()

    def load(self):
        with open(Path(self.base_path, 'raw_data', f'{self.run_name}_wfac.pkl'), 'rb') as file:
            self.raw_results = pickle.load(file)
            self.results = self.raw_results['results']
            self.factors = self.raw_results['factors']

    def colorbar(self, mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar

    def save_signal_and_factors(self):
        idx, sig = write_results_to_file(self.results, Path(self.base_path, f'{self.run_name}.csv'),
                                         down_sample_factor=10)

        cut = abs(len(self.factors) - len(self.results))

        factor_timeline = np.array([i for idx, i in enumerate(self.factors[int(cut):]) if idx % 10 == 0])

        write_signal_to_file(idx, factor_timeline,
                             Path(self.base_path, f'{self.run_name}_factors.csv')
                             )

    def save_histogram(self, vmin=None, vmax=None):
        self.plot_histogram(vmin=vmin, vmax=vmax, backend='pgf')

    def save(self, vmin=None, vmax=None):
        self.save_histogram(vmin=vmin, vmax=vmax)
        self.save_signal_and_factors()

    def plot_histogram(self, freq_Hz=1000, NFFT=2048, backend='Qt5Agg',
                       vmin=None, vmax=None, suffix=None):
        plt.figure()
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

        spec, freqs, t, ax = plt.specgram(self.results, NFFT=NFFT, noverlap=NFFT - int(0.25 * freq_Hz),
                                          Fs=freq_Hz,
                                          scale='dB',
                                          vmin=vmin, vmax=vmax,
                                          cmap='nipy_spectral', interpolation='bicubic')
        plt.axis(ymin=0, ymax=40)
        self.colorbar(ax).set_label('Intensity [dB]')

        if backend == 'pgf':
            plt.savefig(Path(ROOT_DIR, 'thesis', 'data', self.sim_class,
                             f'{self.run_name}{f"_{suffix}" if suffix else ""}.pgf'))
            # self.specgram3d(spec, freqs, t)
        elif backend == 'Qt5Agg':
            # plt.show()

            # print(spec)
            self.do_band_power(t, spec, freqs)
            plt.show()

    def band_power(self, spec, freqs):
        bands = OrderedDict([
            ('delta', [1, 4]),
            ('theta', [4, 8]),
            ('alpha', [8, 13]),
            ('beta', [13, 30]),
            ('beta1', [13, 20]),
            ('beta2', [20, 30]),
            ('gamma', [30, 80])]
        )
        return {
            **{k: np.sum(spec[(freqs > v[0]) & (freqs < v[1])], axis=0) for k, v in bands.items()},
            'total': np.sum(spec[2:], axis=0)
        }

    def do_band_power(self, t, spec, freqs):
        band_powers = self.band_power(spec, freqs)
        rel_band_powers = {k: band_powers[k]/band_powers['total'] for k in band_powers.keys() if k not in ['total']}

        write_multiple_lines_to_file(t, band_powers, Path(self.base_path, f'{self.run_name}_raw_bands.csv'))
        write_multiple_lines_to_file(t, rel_band_powers, Path(self.base_path, f'{self.run_name}_rel_bands.csv'))

        bands = ['alpha', 'beta', 'theta', 'delta', 'gamma']
        plt.figure('Absolute Band Power')
        for p in bands:
            plt.plot(t, band_powers[p], label=p)
        plt.yscale('log')
        plt.legend()

        plt.figure(f'Relative Band Power')
        for p in bands:
            plt.plot(t, band_powers[p] / band_powers['total'], label=p)

        plt.legend()


def model_default_output_spectrogram(seconds):
    a = DataAnalysis(f'DF_REST_{seconds}', 'rest_sim')

    a.plot_histogram(vmin=-110, vmax=-82, backend='pgf')
    a.plot_histogram(vmin=-110, vmax=-82)
    # plt.show()
    a = DataAnalysis(f'JR_REST_{seconds}', 'rest_sim')
    a.plot_histogram(vmin=-110, vmax=-55, backend='pgf')
    a = DataAnalysis(f'JR_REST_{seconds}', 'rest_sim')
    a.plot_histogram(vmin=-110, vmax=-82, backend='pgf', suffix='dfscale')
    a.plot_histogram(vmin=-110, vmax=-55)
    plt.show()


def sedation_session(seconds):
    a = DataAnalysis(f'DF_SEDATION_{seconds}', "sedation_sim")
    a.save(vmin=-110, vmax=-40)
    a.plot_histogram(vmin=-110, vmax=-40)
    a = DataAnalysis(f'JR_SEDATION_{seconds}', "sedation_sim")
    a.save(vmin=-110, vmax=-40)
    a.plot_histogram(vmin=-110, vmax=-40)

    plt.show()


if __name__ == '__main__':
    model_default_output_spectrogram(60)
    sedation_session(150)
