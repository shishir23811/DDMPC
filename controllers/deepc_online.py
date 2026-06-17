import numpy as np
from controllers.deepc_vanilla import VanillaDeePC
from sim.check_pe import hankel


class OnlineDeePC(VanillaDeePC):
    """Sliding-window online DeePC. Rebuilds the QP every
    `update_every` steps."""

    def __init__(self, U_data, Y_data,
                 window_size=None, update_every=10, **kwargs):
        super().__init__(U_data, Y_data, **kwargs)

        self.U_buf = list(U_data)
        self.Y_buf = list(Y_data)

        self.window_size = window_size or len(U_data)
        self.update_every = update_every
        self.step_count = 0

    def push_measurement(self, u_meas, y_meas):
        self.U_buf.append(u_meas)
        self.Y_buf.append(y_meas)

        if len(self.U_buf) > self.window_size:
            self.U_buf.pop(0)
            self.Y_buf.pop(0)

        self.step_count += 1

        if self.step_count % self.update_every == 0:
            self._rebuild_hankel()

    def _rebuild_hankel(self):

        U = np.array(self.U_buf)
        Y = np.array(self.Y_buf)

        self.Hu = hankel(U, self.L + self.n)
        self.Hy = hankel(Y, self.L + self.n)

        self.Hu_p = self.Hu[:self.n * self.m, :]
        self.Hu_f = self.Hu[self.n * self.m:, :]

        self.Hy_p = self.Hy[:self.n * self.p, :]
        self.Hy_f = self.Hy[self.n * self.p:, :]

        # print(np.linalg.matrix_rank(self.Hu_p), np.linalg.matrix_rank(self.Hy_p))
        self._build_problem()