import numpy as np
from pyDOE import lhs
from tqdm import tqdm

import plotly.graph_objs as go

np.seterr(all='raise')
np.random.seed(123)

class Brot:
    def __init__(self, n_points, max_iterations=100, tol=1e10):
        self.n_points = n_points
        self.max_iterations = max_iterations
        self. tol=tol

        self.full_set = np.vstack(self._generate_points()).T
        self.brot_set = self._generate_brot_set()
        return

    def _in_set(self, complex_num) -> bool:
        if complex_num == 0:
            return True
        z_arr = np.zeros(self.max_iterations, dtype=np.complex128)
        z_arr[0] = complex_num
        for itt in range(self.max_iterations-1):
            z_arr[itt+1] = z_arr[itt] ** 2 + complex_num
            # add following to seperate util.py?
            check_ratio = np.abs(z_arr[itt+1] / z_arr[itt])
            if check_ratio > self.tol:
                return False
        return True

    def _generate_points(self):
        reals, imags = lhs(2, samples=self.n_points).T
        reals = reals * (2 + 1.25) - 2  # add doe bounds as an attribute?
        imags = imags * 2 - 1
        return reals, imags

    def _generate_brot_set(self):
        brot_bool = []
        # includes progress bar cli
        for real, imag in tqdm(self.full_set):
            brot_bool.append(self._in_set(real + imag*1j))
        self.brot_set = self.full_set[brot_bool]
        return self.brot_set


if __name__ == '__main__':
    brot = Brot(n_points=100_000)
    np.save("data/full_set.npy", brot.full_set)
    np.save("data/brot_set.npy", brot.brot_set)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=brot.brot_set[:, 0], y=brot.brot_set[:, 1], mode='markers'))
    fig.write_html("figures/brot_set.html")
