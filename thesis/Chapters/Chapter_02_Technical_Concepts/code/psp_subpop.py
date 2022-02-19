rpo_e = OperatorTemplate(
    name='RPO_e', path=None,
    equations=[
        #--------------------------
        # Subpopulation 0: $h_0(t)$
        # $\dot{y}_0 = z_0$
        'd/dt * y_0 = z_0',
        # $\dot{z}_0 = \frac{H_0}{\tau_0}x-\frac{2}{\tau_0}z_0-\frac{1}{\tau_0}^2y_0$
        'd/dt * z_0 = H_0/tau_0 * x - 2./tau_0 * z_0 - (1./tau_0)^2. * y_0',
        #--------------------------
        # Subpopulation 1: $h_1(t)$
        # $\dot{y}_1 = z_1$
        'd/dt * y_1 = z_1',
        # $\dot{z}_1 = \frac{H_1}{\tau_1}x-\frac{2}{\tau_1}z_1-\frac{1}{\tau_1}^2y_1$
        'd/dt * z_1 = H_1/tau_1 * x - 2./tau_1 * z_1 - (1./tau_1)^2. * y_1',
        #--------------------------
        # Population output:
        # $y = \sum_{n=0}^{N}(w_ny_n)$
        'PSP = w_0*y_0 + w_1*y_1'
        #--------------------------
        ],
    variables={
        'PSP': {'default': 'output'},
        **{var: {'default': 'variable'} for var in ['y_0', 'y_1', 'z_0', 'z_1']},
        'x': {'default': 'input'},
        'w_0': {'default': 1.0},
        'w_1': {'default': 0.0},
        'tau_0': {'default': tau_0},
        'tau_1': {'default': tau_1},
        'H_0': {'default': h_0},
        'H_1': {'default': h_1}},
    description="rate-to-potential operator")