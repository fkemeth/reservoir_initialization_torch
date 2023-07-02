import rctorch
import torch

fp_data = rctorch.data.load("forced_pendulum", train_proportion=0.2)

force_train, force_test = fp_data["force"]
target_train, input_test = fp_data["target"]

hps = {
    "connectivity": 0.4,
    "spectral_radius": 1.13,
    "n_nodes": 202,
    "regularization": 1.69,
    "leaking_rate": 0.0098085,
    "bias": 0.49,
}

my_rc = RcNetwork(**hps, random_state=210, feedback=True)

# fitting the data:
my_rc.fit(X=force_train, y=target_train)

# making our prediction
score, prediction = my_rc.test(X=force_test, y=target_test)
my_rc.combined_plot()
