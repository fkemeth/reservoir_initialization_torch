import torch

config = {}
config["DATA"] = {}
# config["DATA"]["n_train"] = 400
# config["DATA"]["n_val"] = 50
# config["DATA"]["n_test"] = 50
config["DATA"]["n_train"] = 600
config["DATA"]["n_val"] = 4
config["DATA"]["n_test"] = 4
config["DATA"]["l_trajectories"] = 150
config["DATA"]["l_trajectories_test"] = 200
config["DATA"]["parameters"] = {}
config["DATA"]["parameters"]["a"] = 1.0
config["DATA"]["parameters"]["b"] = 2.1
config["DATA"]["max_warmup"] = 50
config["PATH"] = "examples/brusselator/data/"

config["TRAINING"] = {}
config["TRAINING"]["epochs"] = 4000
config["TRAINING"]["batch_size"] = 256
config["TRAINING"]["learning_rate"] = 5e-3
config["TRAINING"]["ridge"] = True
config["TRAINING"]["dtype"] = torch.float64
config["TRAINING"]["gh_num_eigenpairs"] = 100
config["TRAINING"]["offset"] = 1
config["TRAINING"]["device"] = "cpu"


config["MODEL"] = {}
# Number of variables to use when using the lorenz system
config["MODEL"]["input_size"] = 1
config["MODEL"]["hidden_size"] = []

# config["MODEL"]["scale_rec"] = 0.9
# config["MODEL"]["scale_in"] = 0.02
# config["MODEL"]["leaking_rate"] = 0.5
# config["MODEL"]["reservoir_size"] = 2000
# config["TRAINING"]["ridge_factor"] = 1e-7
config["MODEL"]["reservoir_size"] = 2**10
config["MODEL"]["scale_rec"] = 0.9805780023782984
config["MODEL"]["scale_in"] = 0.12843039315361987
config["MODEL"]["leaking_rate"] = 0.5068542695918522
config["TRAINING"]["ridge_factor"] = 1e-2

# {'leaking_rate': 0.49944111058657376, 'reservoir_size': 10, 'ridge_factor': 3, 'scale_rec': , 'scale_in': }

# {'leaking_rate': 0.44489563467681187, 'ridge_factor': 1, 'scale_rec': 0.9851103234465945, 'scale_in': 0.15194592415313382}
# {'leaking_rate': 0.5068542695918522, 'ridge_factor': -2, 'scale_rec': , 'scale_in': 0.12843039315361987}

# Train loss: 0.000068
# Val loss: 0.000051
