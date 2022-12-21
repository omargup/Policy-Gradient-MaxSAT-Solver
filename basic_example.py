from src.solvers import pg_solver
import os


config = {
    # Architecture
    "cell": 'GRU',  # 'GRU', 'LSTM'  # check
    "hidden_size": 128,  
    "num_layers": 1,  
    "clip_logits_c": 1,
    "output_size": 1,  #Decoder output size: 1, 2

    # Training
    "num_episodes": 10000,
    "accumulation_episodes": 1,
    "baseline": None,  # None, -1, 1, 2, 3, 4, 5
    "batch_size": 10,
    "permute_vars": False,
    "permute_seed": None,  # 2147483647
    "clip_grad": 1,
    "entropy_weight": 1,
    "lr": 0.0005,  # 0.00015

    # Regularization
    "dropout": 0,

    # Initializers
    "dec_var_initializer": "BasicVar",
    "dec_context_initializer": "EmptyContext",
    "dec_state_initializer": "ZerosState",  # "ZerosState", "TrainableState"
    "initial_state_a": -0.8, #initialize trainable state uniform in (a,b)
    "initial_state_b": 0.8,

    # Embeddings
    "assignment_emb_size": 64,
    "variable_emb_size": 64,  # Useful for encoder and encoder with BasicVar
    "context_emb_size": 64,  # Useful for encoder

    "log_episodes": 100,
    "eval_episodes": 100,
    "eval_strategies": [0, 5],
    "tensorboard_on": True,
    "raytune": False,
    "data_dir": os.path.abspath('data/uf20-91/uf20-01.cnf'),

    "log_dir": 'logs1',
    "output_dir": 'outputs',
    "exp_name": 'exp1',
    "run_name": 'run2',
    "gpu": True,
    "checkpoint_dir": None} 

# Run policy gradient solver
#pg_solver(config, checkpoint_dir=config['checkpoint_dir'], data_dir=config['data_dir'])
pg_solver(config)
