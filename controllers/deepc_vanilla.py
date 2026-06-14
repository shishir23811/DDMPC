import numpy as np
import cvxpy as cp
from sim.check_pe import hankel

class VanillaDeePC:
    """Baseline DeePC with a fixed Hankel matrix built from offline
    data."""

    def __init__(self, U_data, Y_data, L=24, n=6, lam=1e-3, Q=None, R=None, u_min=None, u_max=None):
        self.m, self.p = U_data.shape[1], Y_data.shape[1]
        self.L, self.n, self.lam = L, n, lam
        self.Q = np.eye(self.p) if Q is None else Q
        self.R = 1e-2 * np.eye(self.m) if R is None else R

        self.u_min = u_min if u_min is not None else np.array([-np.deg2rad(15), -2.0])
        self.u_max = u_max if u_max is not None else np.array([np.deg2rad(15), 2.0])

        # Build Hankel matrices of order (L + n)
        self.Hu = hankel(U_data, L + n)
        self.Hy = hankel(Y_data, L + n)

        # Split rows into "past" (first n blocks) and "future" (next L blocks)
        self.Hu_p = self.Hu[:n * self.m, :]
        self.Hu_f = self.Hu[n * self.m:, :]
        self.Hy_p = self.Hy[:n * self.p, :]
        self.Hy_f = self.Hy[n * self.p:, :]

        self._build_problem()

    def _build_problem(self):
        cols = self.Hu.shape[1]
        self.alpha = cp.Variable(cols)

        self.u_bar = cp.Variable((self.L, self.m))
        self.y_bar = cp.Variable((self.L, self.p))

        self.u_past = cp.Parameter(self.n * self.m)
        self.y_past = cp.Parameter(self.n * self.p)

        self.y_ref = cp.Parameter((self.L, self.p))

        cost = self.lam * cp.sum_squares(self.alpha)
        for k in range(self.L):
            cost += cp.quad_form(self.y_bar[k] - self.y_ref[k], self.Q)
            cost += cp.quad_form(self.u_bar[k], self.R)

        cons = [
            self.Hu_p @ self.alpha == self.u_past,
            self.Hy_p @ self.alpha == self.y_past,
            self.Hu_f @ self.alpha == cp.reshape(
                self.u_bar.T,
                (self.L * self.m,),
                order='C'
            ),
            self.Hy_f @ self.alpha == cp.reshape(
                self.y_bar.T,
                (self.L * self.p,),
                order='C'
            ),
            self.u_bar >= self.u_min,
            self.u_bar <= self.u_max,
        ]

        self.prob = cp.Problem(cp.Minimize(cost), cons)

    def __call__(self, u_past_flat, y_past_flat, y_ref_block):
        self.u_past.value = u_past_flat
        self.y_past.value = y_past_flat
        self.y_ref.value = y_ref_block

        self.prob.solve(solver=cp.OSQP,
                        warm_start=True,
                        verbose=False)

        if self.prob.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(
                f"QP did not solve: {self.prob.status}"
            )

        return self.u_bar.value[0]