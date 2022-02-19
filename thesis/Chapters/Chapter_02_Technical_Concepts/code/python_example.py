from pyrates.frontend import OperatorTemplate
from copy import deepcopy

pro = OperatorTemplate(
    name='PRO', path=None,
    equations=[
        # $R_{out} = \frac{2e_0}{1+e^{r(v_0-v)}}$
        "rate_out = 2.*e_0 / (1 + exp(r*(v_0 - v)))"],
    variables={
        'rate_out': {'default': 'output'}, # output pulse density $m_{out}$
        'v': {'default': 'input'},  # incoming avg. membrane potential $v$
        'v_0': {'default': 6e-3},  # avg. firing thresh. $v_0=6mV$
        'e_0': {'default': 2.5},  # half of max. firing rate $e_0=2.5Hz$
        'r': {'default': 560.0}},  # sigmoidal steepness $r=560V^{-1}$ 
    description="sigmoidal potential-to-rate operator")

rpo_e = OperatorTemplate(
    name='RPO_e', path=None,
    equations=[
        # $\dot{y}(t) = z(t)$
        'd/dt * y = z',
        # $\dot{z}(t) = \frac{H}{\tau}x(t) - \frac{2}{\tau}z(t) - {\frac{1}{\tau}}^2y(t)$
        'd/dt * z = H/tau * x - 2 * z/tau - y/tau^2'],
    variables={
        'y': {'default': 'output'},  # output membrane potential $y(t)$
        'z': {'default': 'variable'},  # helper variable $z(t) = \dot{y}(t)$
        'x': {'default': 'input'},  # incoming pulse density $x(t)$
        'tau': {'default': 0.01},  # exc. delays $\tau_e=0.01s$ 
        'H': {'default': 0.00325}},  # exc. synaptical gain $H_e=3.25mV$
    description="excitatory rate-to-potential operator")
    
rpo_i = deepcopy(rpo_e).update_template(
    name='RPO_i', path='', 
    variables={
        'tau': {'default': 0.02},  # inh. delays $\tau_i=0.02s$ 
        'H': {'default':  0.022}},  # inh. synaptical gain $H_i=22mV$ 
    description="inhibitory rate-to-potential operator")
    