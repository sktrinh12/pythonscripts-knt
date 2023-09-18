import numpy as np


def grid(
    d1min,
    d1max,
    d2min,
    d2max,
    n_points1,
    n_points2,
    replicates=1,
    logscale=True,
    include_zero=False,
):
    replicates = int(replicates)

    if logscale:
        d1 = np.logspace(np.log10(d1min), np.log10(d1max), num=n_points1)
        d2 = np.logspace(np.log10(d2min), np.log10(d2max), num=n_points2)
    else:
        d1 = np.linspace(d1min, d1max, num=n_points1)
        d2 = np.linspace(d2min, d2max, num=n_points2)

    if include_zero and logscale:
        if d1min > 0:
            d1 = np.append(0, d1)
        if d2min > 0:
            d2 = np.append(0, d2)

    D1, D2 = np.meshgrid(d1, d2)
    D1 = D1.flatten()
    D2 = D2.flatten()

    D1 = np.hstack(
        [
            D1,
        ]
        * replicates
    )
    D2 = np.hstack(
        [
            D2,
        ]
        * replicates
    )

    return D1, D2


npoints = 8
npoints2 = 12

D1, D2 = grid(1e-3 / 3, 1 / 3, 1e-2, 10, npoints, npoints2, include_zero=True)

print(D1)
print(D2)
