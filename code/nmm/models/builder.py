from typing import List

from models.components import RateToPotentialOperator, Subpopulation, PotentialToRateOperator, RATE_OUT, RATE_IN, PC, \
    EIN, IIN, PYRATES_NEW_VERSION, VarMan

if PYRATES_NEW_VERSION:
    from pyrates import CircuitTemplate, NodeTemplate
else:
    from pyrates.frontend import CircuitTemplate, NodeTemplate

from pyrates.ir import CircuitIR


class ExtendedTemplate:

    def __init__(self, template: CircuitTemplate):
        self.template = template
        self.input_str = None
        self.outputs = None
        self.res_map_reps = []
        self.output_eval = 'V_pci-V_pce'
        self.param_map = None
        self.default_values = {}

    def evaluate_result(self, res, results_map):
        eval_str = self.output_eval
        for key in self.outputs.keys():
            eval_str = eval_str.replace(key, f"res['{key}']")
        for param in self.res_map_reps:
            if param in results_map:
                eval_str = eval_str.replace(param, f"results_map.loc[:, '{param}'].values")
            else:
                eval_str = eval_str.replace(param, f"{self.default_values[param]}")
        print(eval_str)
        res = eval(eval_str)
        return res


class ModelBuilder:

    @staticmethod
    def build_david_friston(exc_sub_pops: List[Subpopulation],
                            inh_sub_pops: List[Subpopulation]) -> ExtendedTemplate:

        pro = PotentialToRateOperator('PRO').template
        pro_pc = pro.update_template(name='PRO_pc', path='')
        rpo_e = RateToPotentialOperator('RPO_e', exc_sub_pops).template
        rpo_i = RateToPotentialOperator('RPO_i', inh_sub_pops).template
        rpo_e_pc = rpo_e.update_template(
            'RPO_e_pc', path='',
            equations={'replace': {RATE_IN: f'({RATE_IN}+p)'}},
            variables={'p': VarMan.create_var(220.0)})
        ein = NodeTemplate(name=EIN, path='', operators=[pro, rpo_e])
        iin = NodeTemplate(name=IIN, path='', operators=[pro, rpo_e])
        pc = NodeTemplate(name=PC, path='', operators=[pro_pc, rpo_e_pc, rpo_i])
        circuit = CircuitTemplate(
            name="DavidFristonCircuit", nodes={PC: pc, EIN: ein, IIN: iin},
            edges=[
                (f"{PC}/PRO_pc/{RATE_OUT}", f"{EIN}/RPO_e/{RATE_IN}", None, {'weight': 135.}),
                (f"{EIN}/PRO/{RATE_OUT}", f"{PC}/RPO_e_pc/{RATE_IN}", None, {'weight': 108.}),
                (f"{PC}/PRO_pc/{RATE_OUT}", f"{IIN}/RPO_e/{RATE_IN}", None, {'weight': 33.75}),
                (f"{IIN}/PRO/{RATE_OUT}", f"{PC}/RPO_i/{RATE_IN}", None, {'weight': 33.75})],
            path='')
        temp = ExtendedTemplate(circuit)
        temp.inhibitory_sign = -1
        temp.input_str = f'{PC}/RPO_e_pc/p'
        pc_e_out = {f'V{i}_pce': f'{PC}/RPO_e_pc/y_{i}' for i in range(len(exc_sub_pops))}
        pc_i_out = {f'V{i}_pci': f'{PC}/RPO_i/y_{i}' for i in range(len(inh_sub_pops))}
        temp.outputs = {**pc_e_out, **pc_i_out}
        temp.res_map_reps.extend([f'w_e_{i}' for i in range(len(exc_sub_pops))])
        temp.res_map_reps.extend([f'w_i_{i}' for i in range(len(inh_sub_pops))])
        for i in range(len(inh_sub_pops)):
            temp.default_values[f'w_i_{i}'] = VarMan.get_var(rpo_i.variables[f'w_{i}'])
        for i in range(len(exc_sub_pops)):
            temp.default_values[f'w_e_{i}'] = VarMan.get_var(rpo_e.variables[f'w_{i}'])

        # temp.output_eval = ' + '.join([f'(V{i}_pce + V{i}_pci)*w_{i}' for i in range(len(sub_pops))])

        temp.output_eval = ' + '.join([f'V{i}_pce*w_e_{i}' for i in range(len(exc_sub_pops))])
        temp.output_eval += ' +'
        temp.output_eval += ' + '.join([f'V{i}_pci*w_i_{i}' for i in range(len(inh_sub_pops))])

        temp.param_map = {
            'v0': {'vars': ['PRO_pc/V_thr'], 'nodes': [PC]},
            'c1': {'vars': ['weight'], 'edges': [(PC, EIN)]},
            'c2': {'vars': ['weight'], 'edges': [(EIN, PC)]},
            'c3': {'vars': ['weight'], 'edges': [(PC, IIN)]},
            'c4': {'vars': ['weight'], 'edges': [(IIN, PC)]}
        }
        return temp

    @staticmethod
    def build_david_friston_default() -> ExtendedTemplate:

        # subpopulation parameters taken from the paper
        tau_e = [10.8e-3, 4.6e-3]
        tau_i = [22e-3, 2.9e-3]

        ws = [0.8, 0.2]

        exc_sub_pops = [
            Subpopulation(h=ModelBuilder.h_e_from_tau(tau), tau=tau, w=ws[idx]) for idx, tau in enumerate(tau_e)
        ]
        inh_sub_pops = [
            Subpopulation(h=ModelBuilder.h_i_from_tau(tau), tau=tau, w=ws[idx]) for idx, tau in enumerate(tau_i)
        ]
        return ModelBuilder.build_david_friston(exc_sub_pops, inh_sub_pops)

    @staticmethod
    def build_jansen_rit_default() -> ExtendedTemplate:
        return ModelBuilder.build_jansen_rit()

    @staticmethod
    def build_jansen_rit(tau_e=None, tau_i=None):
        tau_e = 10e-3 if tau_e is None else tau_e
        tau_i = 20e-3 if tau_i is None else tau_i
        return ModelBuilder.build_david_friston(
            [Subpopulation(h=ModelBuilder.h_e_from_tau(tau_e), tau=tau_e, w=1.0)],
            [Subpopulation(h=ModelBuilder.h_i_from_tau(tau_i), tau=tau_i, w=1.0)]
        )

    @staticmethod
    def h_e_from_tau(tau_e: float):
        return 3.25e-3 * 10e-3 / tau_e

    @staticmethod
    def h_i_from_tau(tau_i: float):
        return -22e-3 * 20e-3 / tau_i
