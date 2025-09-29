# -*- coding: utf-8 -*-
"""
Frechet
======================

This example illustrates the use of Frechet distance between time
series and plots the matches obtained by the method [1]_ compared to DTW.

The Frechet distance is plotted in red:

.. math::

   Frechet(X, Y) = \max_{(i, j) \in \pi} \|X_{i} - Y_{j}\|

.. [1] FRÉCHET, M. "Sur quelques points du calcul fonctionnel.
   Rendiconti del Circolo Mathematico di Palermo", 22, 1–74, 1906.
"""

# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from tslearn.metrics import frechet_path, dtw_path

np.random.seed(42)

nb_points = 100
angle1 = 0.25*np.linspace(0, 4*np.pi, nb_points)
s1 = np.sin(angle1) + 0.1 * np.random.rand(nb_points) + 1
angle2 = np.linspace(0, 2 * np.pi, nb_points)
s2 = 0.5 * np.sin(angle2) + 0.1 * np.random.rand(nb_points)

path_dtw, _ = dtw_path(s1, s2)
path_frechet, distance_frechet = frechet_path(s1, s2)

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
ax.plot(s1)
ax.plot(s2)
for (i, j) in path_frechet:
    is_max = np.linalg.norm(s1[i] - s2[j]) == distance_frechet
    ax.plot(
        [i, j],
        [s1[i], s2[j]],
        'rd:' if is_max else 'k--',
        alpha=1 if is_max else 0.1
    )
ax.set_title("Frechet")

ax = plt.subplot(1, 2, 2)
ax.plot(s1)
ax.plot(s2)
for (i, j) in path_dtw:
    ax.plot([i, j],[s1[i], s2[j]], 'k--', alpha=0.1)
ax.set_title("DTW")

plt.tight_layout()
plt.show()
