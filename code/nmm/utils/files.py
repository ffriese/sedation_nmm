import os
from pathlib import Path
from typing import Dict

import numpy as np
import scipy.signal
from pandas import DataFrame


def write_results_to_file(results: DataFrame, path: Path, down_sample_factor: int = 1):

    if down_sample_factor != 1:
        down_sample_factor = int(down_sample_factor)
        signal = scipy.signal.resample(results, int(len(results)/down_sample_factor))
        index = np.array([i for idx, i in enumerate(results.index) if idx % down_sample_factor == 0])
    else:
        signal = np.array(results)
        index = np.array(results.index)
    write_signal_to_file(index, signal, path)
    return index, signal


def write_signal_to_file(index, signal, path: Path):
    with open(path, 'w') as file:
        file.write('x,y\n')
        for idx, x in enumerate(index):
            file.write(f'{x},{float(signal[idx])}\n')


def write_multiple_lines_to_file(index, signals: Dict, path: Path):
    with open(path, 'w') as file:
        keys = list(signals.keys())
        file.write(f'x,{",".join(keys)}\n')
        for idx, x in enumerate(index):
            file.write(f'{x},{",".join([str(float(signals[key][idx])) for key in keys])}\n')
