import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from pyrates.frontend import OperatorTemplate, NodeTemplate, CircuitTemplate
from pyrates.utility.data_analysis import fft
from pyrates.utility.grid_search import grid_search


def create_pro(name):
    return OperatorTemplate(
        name=name, path='',
        equations=["rate_out = m_max / (1 + exp(r*(v_0 - PSP)))"],
        variables={'rate_out': {'default': 'output'},
                   'PSP': {'default': 'input'},
                   'v_0': {'default': 6e-3},
                   'm_max': {'default': 5.},
                   'r': {'default': 560.0}},
        description="sigmoidal potential-to-rate operator")


def create_rpo(name, tau_0, tau_1, h_0, h_1):
    return OperatorTemplate(
        name=name, path='',
        equations=['d/dt * y_0 = z_0',
                   'd/dt * z_0 = H_0/tau_0 * rate_in - (1./tau_0)^2. * y_0 - 2. * 1./tau_0 * z_0',
                   'd/dt * y_1 = z_1',
                   'd/dt * z_1 = H_1/tau_1 * rate_in - (1./tau_1)^2. * y_1 - 2. * 1./tau_1 * z_1',
                   'PSP = w_0*y_0 + w_1*y_1'],
        variables={'PSP': {'default': 'output'},
                   'y_0': {'default': 'variable'},
                   'y_1': {'default': 'variable'},
                   'z_0': {'default': 'variable'},
                   'z_1': {'default': 'variable'},
                   'rate_in': {'default': 'input'},
                   'w_0': {'default': 1.0},
                   'w_1': {'default': 0.0},
                   'tau_0': {'default': tau_0},
                   'tau_1': {'default': tau_1},
                   'H_0': {'default': h_0},
                   'H_1': {'default': h_1}},
        description="rate-to-potential operator")


def subpopulation_model():
    """
        Creates a Jansen-Rit Circuit with two subpopulations in each rate-to-potential operator
         as proposed by David and Friston
    """
    pro = create_pro(name='PRO')
    pro_pc = create_pro(name='PRO_pc')

    # subpopulation parameters taken from the paper
    tau_e = [10.8e-3, 4.6e-3]
    tau_i = [22e-3, 2.9e-3]

    h_e = [3.25e-3 * 10e-3 / tau for tau in tau_e]
    h_i = [-22e-3 * 20e-3 / tau for tau in tau_i]

    rpo_e = create_rpo('RPO_e', tau_e[0], tau_e[1], h_e[0], h_e[1])
    rpo_e_pc = rpo_e.update_template('RPO_e_pc', '', equations={'replace': {'rate_in': '(rate_in+p)'}},
                                     variables={'p': {'default': 220.0}})
    rpo_i = create_rpo('RPO_i', tau_i[0], tau_i[1], h_i[0], h_i[1])

    ein = NodeTemplate(name="EIN", path='', operators=[pro, rpo_e])
    iin = NodeTemplate(name="IIN", path='', operators=[pro, rpo_e])
    pc = NodeTemplate(name="PC", path='', operators=[pro_pc, rpo_e_pc, rpo_i])

    for rp in [rpo_i, rpo_e, rpo_e_pc]:
        print(rp.equations)
        print(rp.variables)

    jrc = CircuitTemplate(
        name="JRC", nodes={'PC': pc, 'EIN': ein, 'IIN': iin},
        edges=[
               ("PC/PRO_pc/rate_out", "EIN/RPO_e/rate_in", None, {'weight': 135.}),
               ("EIN/PRO/rate_out", "PC/RPO_e_pc/rate_in", None, {'weight': 108.}),
               ("PC/PRO_pc/rate_out", "IIN/RPO_e/rate_in", None, {'weight': 33.75}),
               ("IIN/PRO/rate_out", "PC/RPO_i/rate_in", None, {'weight': 33.75})],
        path='')
    return jrc


def simulate(template, step_size, sampling_step_size, seconds, cutoff):
    """
        Performs a grid search for w (mix the subpopulations)
    """
    ws = np.arange(0, 1.1, 0.2)
    param_grid = {
        'w_0': [round(w, 1) for w in ws],
        'w_1': [round(1-w, 1) for w in ws],
    }
    param_map = {
        'w_0': {
            'vars': ['RPO_e/w_0',
                     'RPO_e_pc/w_0',
                     'RPO_i/w_0'], 'nodes': ['PC', 'EIN', 'IIN']},
        'w_1': {
            'vars': ['RPO_e/w_1',
                     'RPO_e_pc/w_1',
                     'RPO_i/w_1'], 'nodes': ['PC', 'EIN', 'IIN']},
    }
    size = (int(np.round(seconds / step_size, decimals=0)), 1)

    results, results_map = grid_search(circuit_template=template,
                                       param_grid=param_grid,
                                       param_map=param_map,
                                       simulation_time=seconds,
                                       step_size=step_size,
                                       sampling_step_size=sampling_step_size,
                                       inputs={'PC/RPO_e_pc/p': np.random.normal(loc=220.0, scale=22.0, size=size)},
                                       outputs={'I0_pce': 'PC/RPO_e_pc/y_0',
                                                'I1_pce': 'PC/RPO_e_pc/y_1',
                                                'I0_pci': 'PC/RPO_i/y_0',
                                                'I1_pci': 'PC/RPO_i/y_1'},
                                       init_kwargs={'backend': 'numpy', 'solver': 'scipy'},
                                       verbose=False,
                                       permute_grid=False)
    results = results.loc[cutoff:]

    results = (results['I0_pce'] + results['I0_pci'])*results_map.loc[:, 'w_0'].values + \
              (results['I1_pce'] + results['I1_pci'])*results_map.loc[:, 'w_1'].values

    return results, results_map, sampling_step_size


def plot(results, results_map, sampling_step_size):
    """
        Plot the resulting signals and their spectrum
    """
    fig = plt.figure(figsize=(8, 12))
    fig.suptitle('David and Friston (2003) Fig.5')
    gs = fig.add_gridspec(len(results_map), 2)

    def periodogram(data, fs, freq_range):
        _f, power = scipy.signal.periodogram(data, fs)
        selection = np.where((freq_range[0] <= _f) & (_f <= freq_range[1]))
        f = _f[selection]
        power = power[selection]
        return f, power

    def welch(data, fs, freq_range):
        w_len = min(4096, int(len(data) / 2))
        win = scipy.signal.windows.hamming(w_len)
        _f, power = scipy.signal.welch(data, fs, nperseg=w_len,
                                       noverlap=int(w_len / 2), window=win)
        power /= np.sum(power)
        selection = np.where((freq_range[0] <= _f) & (_f <= freq_range[1]))

        return _f[selection], power[selection]

    def simple_fft(data, fs, freq_range):
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
        f = _f[selection]
        power = power[selection]
        return f, power

    for row, key in enumerate(results_map.index):

        # get infos from results map
        ws = [results_map.at[key, c] for c in results_map.columns]
        v_lb = ", ".join([f'${key} = {w}$' for key, w in zip(results_map.columns, ws)])

        # calculate combined LFP
        #psp_0 = results.loc[:, ('I0_pce', key)] + results.loc[:, ('I0_pci', key)]
        #psp_1 = results.loc[:, ('I1_pce', key)] + results.loc[:, ('I1_pci', key)]
        #v = psp_0*ws[0] + psp_1*ws[1]
        #v = psp_0 + psp_1
        #v = results.loc[:, (key,)]
        v = results.iloc[:, row]
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(results.index, v, label=v_lb, color=f'C{row}')
        ax.legend(loc='upper right')
        ax.margins(0.0)

        ax2 = fig.add_subplot(gs[row, 1])
        ax2.margins(0.0)
        ax2.plot(*simple_fft(data=np.squeeze(v.values), fs=1/sampling_step_size, freq_range=(0.1, 100)),
                 label=v_lb, color=f'C{row}')
        ax2.legend(loc='upper right')
        row += 1
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    res = []
    for i in range(1):
        results, results_map, sampling_step_size = simulate(template=subpopulation_model(),
                   step_size=1e-4,  # choosing 1e-3 here does not really make a difference
                   sampling_step_size=1e-3,
                   seconds=2.0, cutoff=1.0)
        res.append(results)
    results = sum(res) / len(res)

    plot(results, results_map, sampling_step_size)
    # plot(*simulate(template=subpopulation_model(),
    #                step_size=1e-4,  # choosing 1e-3 here does not really make a difference
    #                sampling_step_size=1e-3,
    #                seconds=2.0, cutoff=1.0))