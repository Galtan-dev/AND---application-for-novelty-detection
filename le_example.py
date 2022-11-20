import numpy as np
import matplotlib.pylab as plt
import padasip as pa

# data creation
n = 5
N = 2000
x = np.random.normal(0, 1, (N, n))
d = np.sum(x, axis=1) + np.random.normal(0, 0.1, N)

# perturbation insertion
d[1000] += 2.

print(d)
print(x)

# creation of learning model (adaptive filter)
f = pa.filters.FilterNLMS(n, mu=1., w=np.ones(n))
y, e, w = f.run(d, x)
print(w)

# estimation of LE with weights from learning model
le = pa.detection.learning_entropy(w, m=30, order=2, alpha=[8., 9., 10., 11., 12., 13.])
print(le)
# LE plot
plt.plot(le)
plt.show()