import numpy as np
import cvxpy as cp

from controllers.deepc_vanilla import VanillaDeePC


class ReducedOrderDeePC(VanillaDeePC):
    """SVD-reduced DeePC."""

    def __init__(
        self,
        U_data,
        Y_data,
        energy_ratio=None,
        rank=None,
        **kwargs
    ):
        self.energy_ratio = energy_ratio
        self.rank = rank

        super().__init__(U_data, Y_data, **kwargs)

    def _build_problem(self):

        H_full = np.vstack([self.Hu, self.Hy])

        U_svd, s, Vt = np.linalg.svd(
            H_full,
            full_matrices=False
        )

        if self.rank is not None:

            k = min(self.rank, len(s))

            print(
                f"Reduced-order: k={k}/{len(s)} "
                f"(rank mode)"
            )

        else:

            energy = np.cumsum(s**2) / np.sum(s**2)

            k = (
                np.searchsorted(
                    energy,
                    self.energy_ratio,
                    side="left"
                ) + 1
            )

            k_min = self.n * (self.m + self.p)

            k = max(k, k_min)
            k = min(k, len(s))

            print(
                f"Reduced-order: k={k}/{len(s)} "
                f"({100 * energy[k - 1]:.1f}% energy)"
            )

        self.k_kept = k

        V_r = Vt[:k, :].T

        Hu_p_r = self.Hu_p @ V_r
        Hy_p_r = self.Hy_p @ V_r
        Hu_f_r = self.Hu_f @ V_r
        Hy_f_r = self.Hy_f @ V_r

        self.alpha_r = cp.Variable(k)

        self.u_past = cp.Parameter(self.n * self.m)
        self.y_past = cp.Parameter(self.n * self.p)

        self.y_ref = cp.Parameter(
            (self.L, self.p)
        )

        self.u_bar = cp.Variable(
            (self.L, self.m)
        )

        self.y_bar = cp.Variable(
            (self.L, self.p)
        )

        constraints = [

            Hu_p_r @ self.alpha_r == self.u_past,
            Hy_p_r @ self.alpha_r == self.y_past,

            Hu_f_r @ self.alpha_r ==
            cp.reshape(
                self.u_bar,
                (self.L * self.m,),
                order="C"
            ),

            Hy_f_r @ self.alpha_r ==
            cp.reshape(
                self.y_bar,
                (self.L * self.p,),
                order="C"
            )
        ]

        cost = self.lam * cp.sum_squares(self.alpha_r)

        for k_step in range(self.L):
            cost += cp.sum_squares(
                self.Q_sqrt @ (
                    self.y_bar[k_step] - self.y_ref[k_step]
                )
            )
            cost += cp.sum_squares(self.R_sqrt @ self.u_bar[k_step])

        constraints += [
            self.u_bar >= self.u_min,
            self.u_bar <= self.u_max,
        ]

        self.prob = cp.Problem(
            cp.Minimize(cost),
            constraints
        )
