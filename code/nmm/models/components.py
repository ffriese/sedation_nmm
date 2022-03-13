from typing import List

from packaging import version
from pyrates import __version__ as pyrates_version

PYRATES_NEW_VERSION = version.parse(pyrates_version) > version.parse('0.9.5')

from pyrates.frontend import OperatorTemplate

RATE_IN = 'rate_in'
RATE_OUT = 'rate_out'

PC = 'PC'
EIN = 'EIN'
IIN = 'IIN'
PRO_e = 'PRO_e'
PRO_i = 'PRO_i'
PRO_e_pc = 'PRO_e_pc'
RPO = 'RPO'


class VarMan:
    """
    PyRates Template Variable Manager.
        handles different PyRates Versions
    """
    @staticmethod
    def create_var(value=None):
        if PYRATES_NEW_VERSION:
            return value
        else:
            return {'default': value}

    @staticmethod
    def get_var(var):
        if PYRATES_NEW_VERSION:
            return var
        else:
            return var['default']


class Subpopulation:

    def __init__(self, h, tau, w):
        self.h = h
        self.tau = tau
        self.w = w
        self._equations = [
            'd/dt * {y} = {z}',
            f'd/dt * {{z}} = ({{h}}/{{tau}})*{RATE_IN} - (2/{{tau}})*{{z}} - (1/{{tau}})^2 * {{y}}'
        ]
        self._variables = {
            'y': VarMan.create_var('variable'),
            'z': VarMan.create_var('variable'),
            'tau': VarMan.create_var(self.tau),
            'h': VarMan.create_var(self.h),
            'w': VarMan.create_var(self.w),
        }

    def get_equations(self, replacement_index=None):

        repl = {var: f'{var}_{replacement_index}' for var in self._variables.keys()} \
            if replacement_index is not None else {var: var for var in self._variables.keys()}

        return [e.format(**repl) for e in self._equations]

    def get_variables(self, replacement_index=None):

        repl = {var: f'{var}_{replacement_index}' for var in self._variables.keys()} \
            if replacement_index is not None else {var: var for var in self._variables.keys()}

        return {repl[k]: v for k, v in self._variables.items()}

    @staticmethod
    def get_weighted_out(idx):
        return f'(y_{idx}*w_{idx})'


class PotentialToRateOperator:
    def __init__(self, name: str):
        self.template = OperatorTemplate(
            name=name, path='',
            equations=[f"{RATE_OUT} = m_max / (1 + exp(r*(v_0 - PSP)))"],
            variables={RATE_OUT: VarMan.create_var('output'),
                       'PSP': VarMan.create_var('input'),
                       'v_0': VarMan.create_var(6e-3),
                       'm_max': VarMan.create_var(5.),
                       'r': VarMan.create_var(560.0)},
            description="sigmoidal potential-to-rate operator")


class RateToPotentialOperator:
    def __init__(self, name: str, subpopulations: List[Subpopulation]):
        equations = []
        variables = {'PSP': VarMan.create_var('output'),
                     RATE_IN: VarMan.create_var('input')}
        weighted_outputs = []
        for idx, population in enumerate(subpopulations):
            equations.extend(population.get_equations(idx))
            variables = {**variables, **population.get_variables(idx)}
            weighted_outputs.append(population.get_weighted_out(idx))

        equations.append(f"PSP = {'+'.join(weighted_outputs)}")
        self.template = OperatorTemplate(
            name=name, path='',
            equations=equations,
            variables=variables,
            description=f'rate-to-potential operator {name} with subpopulations'
        )




