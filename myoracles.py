import numpy as np
import scipy
from scipy.special import expit

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(w, A, b):
    z = A @ w
    return -np.mean(b * np.log(sigmoid(z)) + (1 - b) * np.log(1 - sigmoid(z)))

def grad(w, A, b):
    z = A @ w
    predictions = sigmoid(z)
    return -A.T @ (b - predictions) / len(b)

def hess(w, A):
    z = A @ w
    predictions = sigmoid(z)
    diag = predictions * (1 - predictions)
    return -A.T @ np.diag(diag) @ A / len(diag)

class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


import numpy as np

class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        Ax = self.matvec_Ax(x)
        logits = -self.b * Ax
        log_loss = np.mean(np.log(1 + np.exp(logits)))
        reg_term = (self.regcoef / 2) * np.linalg.norm(x) ** 2
        return log_loss + reg_term

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        logits = -self.b * Ax
        probs = 1 / (1 + np.exp(logits))

        grad_loss = -self.matvec_ATx(probs - self.b) / self.m
        reg_grad = self.regcoef * x
        return grad_loss + reg_grad

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        logits = -self.b * Ax
        probs = 1 / (1 + np.exp(logits))

        diag_weights = probs * (1 - probs)
        hessian_loss = self.matmat_ATsA(diag_weights) / self.m
        reg_hess = self.regcoef * np.eye(len(x))
        return hessian_loss + reg_hess

class LogRegL2OptimizedOracle(LogRegL2Oracle):

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):

    m, n = A.shape

    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x

    def matmat_ATsA(s):
        return A.T @ (A @ s)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(w, A, b, epsilon=1e-5):
    grad_approx = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[i] += epsilon
        w_minus[i] -= epsilon
        grad_approx[i] = (logistic_loss(w_plus, A, b) - logistic_loss(w_minus, A, b)) / (2 * epsilon)
    return grad_approx


def hess_finite_diff(w, A, epsilon=1e-5):
    hess_approx = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        for j in range(len(w)):
            w_plus_i = w.copy()
            w_minus_i = w.copy()
            w_plus_j = w.copy()
            w_minus_j = w.copy()
            w_plus_i[i] += epsilon
            w_minus_i[i] -= epsilon
            w_plus_j[j] += epsilon
            w_minus_j[j] -= epsilon

            f_plus_plus = logistic_loss(w_plus_i + w_plus_j, A, b)
            f_plus_minus = logistic_loss(w_plus_i + w_minus_j, A, b)
            f_minus_plus = logistic_loss(w_minus_i + w_plus_j, A, b)
            f_minus_minus = logistic_loss(w_minus_i + w_minus_j, A, b)

            hess_approx[i, j] = (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / (4 * epsilon ** 2)
    return hess_approx
