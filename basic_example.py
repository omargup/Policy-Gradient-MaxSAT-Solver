from src.solvers import pg_solver
import os

config = {
    # Encoder
    "node2vec": True,  # {False, True}
    "n2v_dir": "n2v_emb",
    "n2v_dim": 64,
    "n2v_pretrained": True,  # {False, True}
    "n2v_walk_len": 10,
    "n2v_context_size": 5,
    "n2v_walks_per_node": 5,
    "n2v_p": 1,
    "n2v_q": 1,
    "n2v_batch_size": 32,
    "n2v_lr": 0.01,
    "n2v_num_epochs": 100,
    "n2v_workers": 0,  # {0, 1, 2, 3, 4}
    "n2v_verbose": 1,  # {0, 1, 2}

    # Initializers
    "dec_var_initializer": "Node2VecVar",  # {"BasicVar", "Node2VecVar"}
    "dec_context_initializer": "Node2VecContext",  # {"EmptyContext", "Node2VecContext"}

    # Embeddings
    "var_emb_size": 128,
    "assignment_emb_size": 64,
    "context_emb_size": 128,
    "model_dim": 128, 

    # Architecture
    "decoder": 'Transformer',  # {'GRU', 'LSTM', "Transformer"}
    "num_layers": 2,  
    "output_size": 2,  #Decoder output size: 1, 2
    "dropout": 0,

    #"hidden_size": 128,  #hidden_size if RNN
    #"trainable_state": True,  # {False, True}

    "num_heads": 2,
    "dense_size":128,

    # Training
    "num_episodes": 2000,  #4000
    "accumulation_episodes": 1,
    "batch_size": 10,  #10
    "permute_vars": True,
    "permute_seed": None,  # 2147483647
    "clip_grad": 1,
    "lr": 0.00001,  # 0.00015   0.00001

    # Baseline
    "baseline": 'sample',  # {None, 'greedy', 'sample'. 'ema'}
    "alpha_ema": 0.99,  # 0 <= alpha <= 1. EMA decay, useful if baseline == 'ema'
    "k_samples": 10,  # int, k >= 1. Number of samples used to obtain the baseline value, useful if baseline == 'sample'

    # Exploration
    "logit_clipping": None,  # {None, int >= 1}
    "logit_temp": None,  # {None, int >= 1}. Useful for improve exploration in evaluation.
    "entropy_estimator": 'crude',  # {'crude', 'smooth'}
    "beta_entropy": 0.03,  # float, beta >= 0.

    # Regularization
    "early_stopping": False,
    "patience": 6,
    "entropy_value": 0,

    "log_interval": 20,
    "eval_interval": 100,
    "eval_strategies": [0, 32],
    "tensorboard_on": True,
    "extra_logging": False,  # log TrainableState's weights
    "raytune": False,
    "progress_bar": True,
    "data_dir": os.path.abspath('data/sat_rand/sat_rand_n=0020_k=03_m=0040_i=1.cnf'),

    "log_dir": 'logs',
    "output_dir": 'outputs',
    "exp_name": 'exp',
    "run_name": 'run',
    "gpu": True,
    "checkpoint_dir": None} 


pg_solver(config)