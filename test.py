from multiprocessing import Pool


def create_trajectory(inputs):
    time_array, parameters = inputs
    return [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]


with Pool(4) as p:
    # input_data, output_data, v_data = \
    #     p.map(create_trajectory, [[1.0, 1.0] for _ in range(10)])
    results = list(zip(*p.map(create_trajectory, [[1.0, 1.0] for _ in range(10)])))

import numpy as np
from sklearn.linear_model import Ridge

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y)
