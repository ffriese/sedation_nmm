import inspect
import os
import sys
from inspect import FrameInfo
from pathlib import Path

import numpy as np
import scipy.signal
from pandas import DataFrame


def get_project_root_dir() -> Path:
    """
    Returns the name of the project root directory.

    :return: Project root directory name
    """

    # stack trace history related to the call of this function
    frame_stack: [FrameInfo] = inspect.stack()

    # get info about the module that has invoked this function
    # (index=0 is always this very module, index=1 is fine as long this function is not called by some other
    # function in this module)
    frame_info: FrameInfo = frame_stack[1]

    # if there are multiple calls in the stacktrace of this very module, we have to skip those and take the first
    # one which comes from another module
    if frame_info.filename == __file__:
        for frame in frame_stack:
            if frame.filename != __file__:
                frame_info = frame
                break

    # path of the module that has invoked this function
    caller_path: str = frame_info.filename

    # absolute path of the of the module that has invoked this function
    caller_absolute_path: str = os.path.abspath(caller_path)

    # get the top most directory path which contains the invoker module
    paths: [str] = [p for p in sys.path if p in caller_absolute_path]
    paths.sort(key=lambda p: len(p))
    caller_root_path: str = paths[0]

    if not os.path.isabs(caller_path):
        # file name of the invoker module (eg: "mymodule.py")
        caller_module_name: str = Path(caller_path).name

        # this piece represents a subpath in the project directory
        # (eg. if the root folder is "myproject" and this function has ben called from myproject/foo/bar/mymodule.py
        # this will be "foo/bar")
        project_related_folders: str = caller_path.replace(os.sep + caller_module_name, '')

        # fix root path by removing the undesired subpath
        caller_root_path = caller_root_path.replace(project_related_folders, '')

    return Path(caller_root_path).absolute()


ROOT_DIR = get_project_root_dir()


def write_results_to_file(results: DataFrame, path: Path, downsample: int = 1):

    if downsample != 1:
        downsample = int(downsample)
        signal = scipy.signal.resample(results, int(len(results)/downsample))
        index = np.array([i for idx, i in enumerate(results.index) if idx % downsample == 0])
    else:
        signal = np.array(results)
        index = np.array(results.index)
    with open(path, 'w') as file:
        file.write('x,y\n')
        for idx, x in enumerate(index):
            file.write(f'{x},{float(signal[idx])}\n')
    return index, signal
