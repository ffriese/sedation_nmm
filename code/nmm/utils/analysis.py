import numpy as np
import scipy.signal
from mne.time_frequency import psd_array_multitaper


class PostProcessing:
    def __init__(self, func, f_kwargs=None, plot_kwargs=None, combine_rows=False):
        self.function = func
        self.f_kwargs = {} if f_kwargs is None else f_kwargs
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        self.combine_rows = combine_rows

    @staticmethod
    def periodogram(data, index, fs, freq_range):
        _f, power = scipy.signal.periodogram(data, fs)
        10 * np.log10(np.power(power, 2))
        selection = np.where((freq_range[0] <= _f) & (_f <= freq_range[1]))
        f = _f[selection]
        power = power[selection]
        return f, power

    @staticmethod
    def welch(data, index, fs, freq_range):
        w_len = min(4096, int(len(data) / 2))
        win = scipy.signal.windows.hamming(w_len)
        _f, power = scipy.signal.welch(data, fs, nperseg=w_len,
                                       noverlap=int(w_len / 2), window=win)
        power /= np.sum(power)
        selection = np.where((freq_range[0] <= _f) & (_f <= freq_range[1]))

        return _f[selection], power[selection]

    @staticmethod
    def multitaper(data, index, fs, freq_range, bandwidth=None):
        power, _f = psd_array_multitaper(data, fs, bandwidth=bandwidth, fmin=freq_range[0], fmax=freq_range[1],
                                         adaptive=True, normalization='full', verbose=0)
        return _f, power

    @staticmethod
    def fft(data, index, fs, freq_range):
        n = len(data)

        # Get closest power of 2 that includes n for zero padding
        n_two = 1 if n == 0 else 2 ** (n - 1).bit_length()

        data_tmp = data
        data_tmp = data_tmp - np.mean(data_tmp)

        freqs = np.linspace(0, fs, n_two)
        spec = np.fft.fft(data_tmp, n=n_two, axis=0)

        # Cut of PSD and frequency arrays since its mirrored at N/2
        power = np.abs(spec[:int(len(spec) / 2)])
        10 * np.log10(np.power(power, 2))
        _f = freqs[:int(len(freqs) / 2)]
        selection = np.where((freq_range[0] <= _f) & (_f <= freq_range[1]))
        return _f[selection], power[selection]
