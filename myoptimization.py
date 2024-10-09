import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
from collections import defaultdict
from scipy.linalg import cho_factor, cho_solve

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):

        if previous_alpha is not None:
            alpha = previous_alpha
        else:
            alpha = self.alpha_0
        phi = lambda alpha: oracle.func(x_k + alpha * d_k)
        phi_prime = lambda alpha: oracle.grad_directional(x_k, d_k)
        alpha, _, _ = scalar_search_wolfe2(phi, phi_prime, a=0, c1=1e-4, c2=0.9)

        if alpha is None:
            c1 = 1e-4
            alpha = self.alpha_0

            while oracle.func(x_k + alpha * d_k) > oracle.func(x_k) + c1 * alpha * phi_prime(0):
                alpha *= 0.5

            if alpha < 1e-10:
                return None

        return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)

        if np.linalg.norm(grad_k) < tolerance:
            message = 'success'
            break

        alpha = line_search_tool.line_search(x_k, grad_k)

        if alpha is None:
            alpha = 1.0
            c1 = line_search_options.get('c1', 1e-4)

            while oracle.func(x_k - alpha * grad_k) > oracle.func(x_k) - c1 * alpha * np.dot(grad_k, grad_k):
                alpha *= 0.5

        x_k = x_k - alpha * grad_k

        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_k))
            history['x'].append(np.copy(x_k))

        if display:
            print(f"Iteration {iteration}: x = {x_k}, f(x) = {oracle.func(x_k)}, grad_norm = {np.linalg.norm(grad_k)}")

    else:
        message = 'iterations_exceeded'

    return x_k, message, history




def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    for iteration in range(max_iter):
        grad_k = oracle.grad(x_k)
        hess_k = oracle.hess(x_k)

        if np.linalg.norm(grad_k) < tolerance:
            return x_k, 'success', history

        try:
            L, lower = cho_factor(hess_k)
        except np.linalg.LinAlgError:
            return x_k, 'newton_direction_error', history
        y_k = cho_solve((L, lower), -grad_k)

        alpha = line_search_tool.line_search(oracle, x_k, y_k)

        if alpha is None:
            return x_k, 'computational_error', history

        x_k += alpha * y_k

        if trace:
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_k))
            history['x'].append(np.copy(x_k))

        if display:
            print(f"Iteration {iteration + 1}: x_k = {x_k}, f(x_k) = {oracle.func(x_k)}")

    return x_k, 'iterations_exceeded', history

