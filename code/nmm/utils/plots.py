import time
from typing import List

import matplotlib
import matplotlib.pyplot as plt

from utils.analysis import PostProcessing


class Subplot:
    def __init__(self, data=None, row_idx=0, col_idx=0, plt_kwargs=None, **kwargs):
        self.data = ([1, 2, 3], [0, 25, 0]) if data is None else data
        self.kwargs = kwargs
        self.plt_kwargs = {} if plt_kwargs is None else plt_kwargs
        self.col_idx = col_idx
        self.row_idx = row_idx


class PlotDefinition:
    def __init__(self):
        self.subplots = []

    def rows(self) -> int:
        return max([sp.row_idx if sp.row_idx is not None else 0 for sp in self.subplots]) + 1

    def cols(self) -> int:
        return max([sp.col_idx if sp.col_idx is not None else 0 for sp in self.subplots]) + 1

    def add_subplot(self, subplot: Subplot, ):
        self.subplots.append(subplot)


class MultiPlotter:

    @staticmethod
    def plot(plot_def: PlotDefinition, plot_title: str = 'PlotTitle', block=True, save_to_file=False, no_window=False):
        all_nums = plt.get_fignums()
        next_num = max(all_nums) + 1 if all_nums else 1
        fig = plt.figure(figsize=(12, 10), num=f'Fig. {next_num}: {plot_title}')
        fig.suptitle(plot_title)
        gs = fig.add_gridspec(plot_def.rows(), plot_def.cols())

        axes = {}
        for sp in plot_def.subplots:
            idx = (sp.row_idx, sp.col_idx)
            if axes.get(idx, None) is None:
                if plot_def.rows() == plot_def.cols() == 1:
                    axes[idx] = fig.add_subplot(gs[:, :], **sp.kwargs)
                elif sp.row_idx is None:
                    axes[idx] = fig.add_subplot(gs[:, sp.col_idx], **sp.kwargs)
                elif sp.col_idx is None:
                    axes[idx] = fig.add_subplot(gs[sp.row_idx, :], **sp.kwargs)
                else:
                    axes[idx] = fig.add_subplot(gs[sp.row_idx, sp.col_idx], **sp.kwargs)
            axes[idx].margins(0.0)
            axes[idx].plot(*sp.data, **sp.plt_kwargs)
            axes[idx].legend(loc='upper right')
        plt.tight_layout()
        MultiPlotter.move_figure(fig, 0, 0, 1200, 1000)
        plt.tight_layout()
        # plt.draw()
        # plt.
        if save_to_file:
            plt.savefig(save_to_file)
        if not no_window:
            time.sleep(0.01)
            plt.show(block=block)
            time.sleep(0.1)

    @staticmethod
    def move_figure(f, x, y, w=None, h=None):
        """
            Move figure's upper left corner to pixel (x, y)
        """
        backend = matplotlib.get_backend()
        if backend == 'TkAgg':
            f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
        elif backend == 'WXAgg':
            f.canvas.manager.window.SetPosition((x, y))
        elif backend != 'module://backend_interagg':  # pyCharm SciView cannot move anything
            # This works for QT and GTK
            # You can also use window.setGeometry
            f.canvas.manager.window.move(x, y)
            if w is not None or h is not None:
                w = w if w is not None else f.canvas.manager.window.width()
                h = h if h is not None else f.canvas.manager.window.height()
                f.canvas.manager.window.resize(w, h)

    @staticmethod
    def format_values(val_dict: dict, ignore_keys=None, replace_keys=None) -> str:
        value_dict = dict(val_dict)

        def repl(var):
            return var.replace('tau', r'\tau')

        if ignore_keys:
            for k in ignore_keys:
                value_dict.pop(k, None)
        if replace_keys:
            for k, replacement in replace_keys.items():
                v = value_dict.pop(k, None)
                value_dict[replacement] = v
        ret = {r'$%s$' % repl(k): v for k, v in value_dict.items()}
        for k in list(ret.keys()):
            if len(str(ret[k])) > 5:
                if(round(ret[k], 8) - ret[k]) < 1e-8:
                    ret[k] = round(ret[k], 8)
                else:
                    ret[k] = f'{ret[k]:.2e}'

        vs = [f'{k}: {v}' for k, v in ret.items()]
        return ', '.join(vs)

    @staticmethod
    def create_plot_def(results, results_map, post_proc: List[PostProcessing], ignore_keys, replace_keys=None):
        if results_map:
            r_map = [
                {key: results_map.at[results_map.index[i], key] for key in list(results_map.columns)}
                for i in range(len(results_map.index))
            ]
        else:
            r_map = [{}]
        plotdef = PlotDefinition()
        for i in range(len(results.columns) if hasattr(results, 'columns') else 1):
            print('COLUMN', i)
            x = results.index
            y = results.iloc[:, i] if hasattr(results, 'columns') else results
            # plotdef.add_subplot(Subplot((x, y), plt_kwargs={'color': f'C{i}', 'label': r_map[i]}, row_idx=i,
            #                             xlabel='time (s)',
            #                             ylabel='PSP (mv)'))
            for col, proc in enumerate(post_proc):
                plotdef.add_subplot(Subplot((*proc.function(**{**{'data': y, 'index': x}, **proc.f_kwargs}),),
                                            plt_kwargs={**{'label':
                                                               MultiPlotter.format_values(r_map[i],  ignore_keys,
                                                                                          replace_keys),
                                                           'color': f'C{i}'}, **proc.plot_kwargs},
                                            col_idx=col,
                                            row_idx=None if proc.combine_rows else i,
                                            ),
                                    )
        return plotdef

    @staticmethod
    def create_grid(results, results_map, proc: PostProcessing, ignore_keys, replace_keys=None):
        r_map = [
            {key: results_map.at[results_map.index[i], key] for key in list(results_map.columns)}
            for i in range(len(results_map.index))
        ]
        #cols = {results_map.at[results_map.index[i], key] for key in list(results_map.columns)}
        cols = sorted(list({r['tau_e']: 0 for r in r_map}.keys()))
        rows = sorted(list({r['tau_i']: 0 for r in r_map}.keys()))
        plotdef = PlotDefinition()
        for i in range(len(results.columns)):
            x = results.index
            y = results.iloc[:, i]
            plotdef.add_subplot(Subplot((*proc.function(**{**{'data': y, 'index': x}, **proc.f_kwargs}),),
                                        plt_kwargs={**{'label': MultiPlotter.format_values(r_map[i], ignore_keys,
                                                                                           replace_keys),
                                                       'color': f'C{i}'}, **proc.plot_kwargs},
                                        col_idx=cols.index(r_map[i]['tau_e']),
                                        row_idx=rows.index(r_map[i]['tau_i']),
                                        ),
                                    )
        return plotdef
