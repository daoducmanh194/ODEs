import torch
from torch.autograd.functional import vjp
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fehlberg2 import Fehlberg2
from .fixed_grid import Euler, Midpoint, Heun3, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .scipy_wrapper import ScipyWrapperODESolver
from .misc import _check_inputs, _flat_to_shape
from .interp import _interp_evaluate


SOLVERS = {
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'fehlberg2': Fehlberg2,
    'adaptive_heun': AdaptiveHeunSolver,
    'euler': Euler,
    'midpoint': Midpoint,
    'heun3': Heun3,
    'rk4': RK4,
    'explicit_adams': AdamsBashforth,
    'implicit_adams': AdamsBashforthMoulton,
    # Backward compatibility: use the same name as before
    'fixed_adams': AdamsBashforthMoulton,
    # ~Backwards compatibility
    'scipy_solver': ScipyWrapperODESolver,
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`, in either increasing or decreasing order. The first element of
            this sequence is taken to be the initial time point.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.
        event_fn: Function that maps the state `y` to a Tensor. The solve terminates when
            event_fn evaluates to zero. If this is not None, all but the first elements of
            `t` are ignored.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t)
        if t_is_reversed:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def odeint_dense(func, y0, t0, t1, *, rtol=1e-7, atol=1e-9, method=None, options=None):
    """
    A function to integrate a system of ordinary differential equations.
    
    :param func: The system of ordinary differential equations to integrate.
    :param y0: The initial state of the system.
    :param t0: The initial time.
    :param t1: The final time.
    :param rtol: The relative tolerance for the solver.
    :param atol: The absolute tolerance for the solver.
    :param method: The integration method to use (default is None).
    :param options: Additional options for the solver (default is None).
    :return: The function for evaluating the solution at any time within the integration interval.
    :rtype: function
    """

    assert torch.is_tensor(y0)  # TODO: handle tuple of tensors

    t = torch.tensor([t0, t1]).to(t0)

    shapes, func, y0, t, rtol, atol, method, options, _, _ = _check_inputs(func, y0, t, rtol, atol, method, options, None, SOLVERS)

    assert method == "dopri5"

    solver = Dopri5Solver(func=func, y0=y0, rtol=rtol, atol=atol, **options)    
    
    # The integration loop
    solution = torch.empty(len(t), *solver.y0.shape, dtype=solver.y0.dtype, device=solver.y0.device)
    solution[0] = solver.y0
    t = t.to(solver.dtype)
    solver._before_integrate(t)
    t0 = solver.rk_state.t0

    times = [t0]
    interp_coeffs = []

    for i in range(1, len(t)):
        next_t = t[i]
        while next_t > solver.rk_state.t1:
            solver.rk_state = solver._adaptive_step(solver.rk_state)
            t1 = solver.rk_state.t1

            if t1 != t0:
                # Step accepted.
                t0 = t1
                times.append(t1)
                interp_coeffs.append(torch.stack(solver.rk_state.interp_coeff))

        solution[i] = _interp_evaluate(solver.rk_state.interp_coeff, solver.rk_state.t0, solver.rk_state.t1, next_t)

    times = torch.stack(times).reshape(-1).cpu()
    interp_coeffs = torch.stack(interp_coeffs)

    def dense_output_fn(t_eval):
        """
        Function to perform dense output interpolation at the given evaluation time.

        Parameters:
            t_eval (torch.Tensor): The time at which interpolation is to be performed.

        Returns:
            torch.Tensor: The interpolated value at the evaluation time.
        """
        idx = torch.searchsorted(times, t_eval, side="right")
        t0 = times[idx - 1]
        t1 = times[idx]
        coef = [interp_coeffs[idx - 1][i] for i in range(interp_coeffs.shape[1])]
        return _interp_evaluate(coef, t0, t1, t_eval)

    return dense_output_fn