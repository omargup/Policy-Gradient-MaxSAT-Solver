from src.solvers import pg_solver
import os

config = {
    # Encoder
    "node2vec": True,  # (bool).
    "n2v_dir": os.path.abspath("n2v_emb"),  # (str).
    "n2v_dim": 64,  # (int).
    "n2v_pretrained": True,  # {False, True}
    "n2v_walk_len": 10,  # (int).
    "n2v_context_size": 5,  # (int).
    "n2v_walks_per_node": 5,  # (int).
    "n2v_p": 1,  # (int).
    "n2v_q": 1,  # (int).
    "n2v_batch_size": 32,  # (int).
    "n2v_lr": 0.01,  # (float).
    "n2v_num_epochs": 100,  # (int).
    "n2v_workers": 4,  # (int). {0, 1, 2, 3, 4}
    "n2v_verbose": 1,  # (int). {0, 1, 2}

    # Initializers
    "dec_var_initializer": "Node2VecVar",  # (str). {"BasicVar", "Node2VecVar"}
    "dec_context_initializer": "EmptyContext",  # (str). {"EmptyContext", "Node2VecContext"}

    # Embeddings
    "var_emb_size": 128,  # (int).
    "assignment_emb_size": 64,  # (int).
    "context_emb_size": 128,  # (int).
    "model_dim": 128,  # (int).

    # Architecture
    "decoder": 'Transformer',  # (str). {'GRU', 'LSTM', "Transformer"}
    "num_layers": 2,  # (int).
    "output_size": 1,  # (int). Decoder output size: {1, 2}
    "dropout": 0,  # (float).

    "hidden_size": 128,  # (int). Hidden size of the RNN.
    "trainable_state": True,  # (bool). Trainable initial state of the RNN if True, else zeros initial state.

    "num_heads": 2,  # (int). Number of heads of the Transformer decoder.
    "dense_size":128,  # (int). Number of units of the position-wise FFN in the Transformer decoder.

    # Training
    "num_samples": 15000,  # (int).
    "accumulation_episodes": 1,  # (int).
    "batch_size": 20,  #10 # (int).
    "permute_vars": True,  # (bool).
    "permute_seed": None,  # (int). e.g.: 2147483647
    "clip_grad": 1.0,  # {None, float} e.g.:0.00015.
    "lr": 0.00015 ,  # (float). e.g.: 0.00015. or 0.00001

    # Baseline
    "baseline": 'sample',  # {None, 'greedy', 'sample'. 'ema'}
    "alpha_ema": 0.99,  # (float). 0 <= alpha <= 1. EMA decay.
    "k_samples": 10,  # (int). k >= 1. Number of samples used to obtain the sample baseline value.

    # Exploration
    "logit_clipping": 5,  # {None, int >= 1}
    "logit_temp": 2,  # {None, float >= 1}. Useful for improve exploration in evaluation.
    "entropy_estimator": 'crude',  # (str). {'crude', 'smooth'}
    "beta_entropy": 0.03,  # (float). beta >= 0.

    # Misc
    "sat_stopping": True,  # (bool). Stop when num_sat is equal with the num of clauses.
    "log_interval": 20,  # (int).
    "eval_interval": 100,  # (int).
    "eval_strategies": [0, 32],  # (list of ints). 0 for greedy search, k >= 1 for k samples.
    "tensorboard_on": True,  # (bool).
    "extra_logging": False,  # (bool). Log Trainable state's weights.
    "raytune": False,  # (bool).
    "data_dir": os.path.abspath('data/rand/0020/0040/rand_n=0020_k=03_m=0040_i=1.cnf'),  # (str).
    "verbose": 1,  # {0, 1, 2}. If raytune is True, then verbose is set to 0.

    "log_dir": 'logs',  # (str).
    "output_dir": 'outputs',  # (str).
    "exp_name": 'ex_basic',  # (str).
    "run_name": 'run',  # (str).
    "gpu": True,  # (bool).
    "checkpoint_dir": None} 


pg_solver(config)