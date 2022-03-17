import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from utils import ROOT_DIR, write_results_to_file
from models.builder import ModelBuilder
from runner.model_runner import ModelRunner


class SedationSim:
    def __init__(self):
        self.model = ModelBuilder.build_david_friston_default()
        self.runner = ModelRunner(self.model)

    def gen_factors(self, seconds, cut):
        start = 1.0
        middle = 1.7
        step = 0.01
        factor_series = [*[start] * 15,
                         *np.arange(start, middle + step, step),
                         *[middle] * 60,
                         *np.flip(np.arange(start, middle, step)),
                         *[start] * 15]

        lf = len(factor_series)

        f_ser = np.interp(np.linspace(1, lf, seconds * 1000), range(lf), factor_series)

        f_ser = np.pad(f_ser, (int(cut * 1000), 0), mode='constant', constant_values=(start,))
        return f_ser


if __name__ == '__main__':
    sedation = SedationSim()

    seconds = 40
    cut = 1.0
    f_ser = sedation.gen_factors(seconds, cut)

    inputs = {
        'PC/RPO_i/tau_fac_0': f_ser,
        'PC/RPO_i/tau_fac_1': f_ser,
    }
    t = time.time()
    plt.plot(f_ser)
    plt.show()
    results = sedation.runner.run(seconds+cut, inputs=inputs, cut=cut)
    print(f'simulating {seconds+cut}s took {time.time()-t} seconds')
    # write_results_to_file(results, Path(ROOT_DIR, 'thesis', 'data', 'methodology', 'uncut.csv'))
    plt.figure()
    plt.plot(results)
    plt.show()

    import pylab

    fs_Hz = 1000
    NFFT = 2048

    plt.specgram(results, NFFT=NFFT, noverlap=NFFT - int(0.25 * fs_Hz), Fs=fs_Hz, scale='dB',
                 vmin=-111, vmax=-81,
                 cmap=pylab.cm.nipy_spectral, interpolation='bicubic')
    plt.axis(ymin=0, ymax=40)

    plt.show()
