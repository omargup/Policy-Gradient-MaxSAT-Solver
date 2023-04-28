import os
import time
import json
import argparse
import torch

def get_config(new_config=None):
    optuna_by_run = new_config.get("optuna_by_run", False)
    if optuna_by_run:
        config = new_config
    else:
        config = {
            # Encoder
            "node2vec": False,  # (bool).
            "n2v_dir": "n2v_emb",  # (str).
            "n2v_dim": 64,  # (int).
            "n2v_pretrained": True,  # (bool).
            "n2v_walk_len": 10,  # (int).
            "n2v_context_size": 5,  # (int).
            "n2v_walks_per_node": 5,  # (int).
            "n2v_p": 1,  # (float).
            "n2v_q": 1,  # (float).
            "n2v_batch_size": 32,  # (int).
            "n2v_lr": 0.01,  # (float).
            "n2v_num_epochs": 150,  # (int).
            "n2v_workers": 0,  # (int). {0, 1, 2, 3, 4}
            "n2v_verbose": 1,  # (int). {0, 1, 2}

            # Initializers
            "dec_var_initializer": "BasicVar",  # (str). {"BasicVar", "Node2VecVar"}
            "dec_context_initializer": "EmptyContext",  # (str). {"EmptyContext", "Node2VecContext"}

            # Embeddings
            "var_emb_size": 64,  # (int).
            "assignment_emb_size": 64,  # (int).
            "context_emb_size": 64,  # (int).
            "model_dim": 256,  # (int).

            # Architecture
            "decoder": 'GRU',  # (str). {'GRU', 'LSTM', "Transformer"}
            "num_layers": 1,  # (int).
            "output_size": 2,  # (int). Decoder output size: {1, 2}
            "dropout": 0,  # (float).

            "hidden_size": 128,  # (int). Hidden size of the RNN.
            "trainable_state": False,  # (bool). Trainable initial state of the RNN if True, else zeros initial state.

            "num_heads": 2,  # (int). Number of heads of the Transformer decoder.
            "dense_size": 256,  # (int). Number of units of the position-wise FFN in the Transformer decoder.

            # Training
            "num_samples": 15000, # (int).
            "accumulation_episodes": 1,  # (int).
            "batch_size": 10,  # (int).
            "permute_vars": True,  # (bool).
            "permute_seed": None,  # (int). e.g.: 2147483647
            "clip_grad": 1,  # {None, float} e.g.:0.00015.
            "lr": 0.00015,  # (float). e.g.: 0.00015.

            # Baseline
            "baseline": 'greedy',  # {'zero', 'greedy', 'sample'. 'ema'}
            "alpha_ema": 0.99,  # (float). 0 <= alpha <= 1. EMA decay.
            "k_samples": 10,  # (int). k >= 1. Number of samples used to obtain the sample baseline value.
            
            # Exploration
            "logit_clipping": None,  # {None, int >= 1}
            "logit_temp": None,  # {None, float >= 1}. Useful for improve exploration in evaluation.
            "entropy_estimator": 'crude',  # (str). {'crude', 'smooth'}
            "beta_entropy": 0,  # (float). beta >= 0.

            # Misc
            "sat_stopping": True,  # (bool). Stop when num_sat is equal with the num of clauses.
            "log_interval": 100,  # (int).
            "eval_interval": 200,  # (int).
            "eval_strategies": [0, 32],  # (list of ints). 0 for greedy search, k >= 1 for k samples.
            "tensorboard_on": True,  # (bool).
            "extra_logging": False,  # (bool). Log Trainable state's weights.
            "raytune": False,  # (bool).
            "data_dir": None,  # (str).
            "verbose": 1,  # (int). {0, 1, 2}. If raytune is True, then verbose is set to 0.

            "log_dir": 'logs',  # (str).
            "output_dir": 'outputs',  # (str).
            "exp_name": 'exp',  # (str).
            "run_name": 'run',  # (str).
            "gpu": True,  # (bool).
            "checkpoint_dir": 'checkpoints'}  # None | str
    
        # Update default config
        for key in new_config:
            config[key] = new_config[key]
    
        # Delete unused entries
        if not config["node2vec"]:
            del config["n2v_dir"]
            del config["n2v_dim"]
            del config["n2v_pretrained"]
            del config["n2v_walk_len"]
            del config["n2v_context_size"]
            del config["n2v_walks_per_node"]
            del config["n2v_p"]
            del config["n2v_q"]
            del config["n2v_batch_size"]
            del config["n2v_lr"]
            del config["n2v_num_epochs"]
            del config["n2v_workers"]
            del config["n2v_verbose"]
        
        if config["decoder"] == "Transformer":
            del config["hidden_size"]
            del config["trainable_state"]
        else:  # GRU or LSTM
            del config["num_heads"]
            del config["dense_size"]

        #if not config["permute_vars"]: 
        #    del config["permute_seed"]
        
        if (config["baseline"] is None) or (config["baseline"] == "greedy"):
           del config["alpha_ema"]
           del config["k_samples"]
        elif config["baseline"] == "sample":
           del config["alpha_ema"]
        else:  # "ema"
           del config["k_samples"]

    
    # Set default config
    if config['run_name'] is None:
        config['run_name'] = 'run'
    if config['exp_name'] is None:
        config['exp_name'] = 'exp'
    if config['log_dir'] is None:
        config['log_dir'] = 'logs'
    if config['output_dir'] is None:
        config['output_dir'] = 'outputs'
    if config['checkpoint_dir'] is None:
        config['checkpoint_dir'] = 'checkpoints'
    
    # Generate run_id
    config['run_id'] = time.strftime("%Y%m%dT%H%M%S")

    # Set output directory
    if config["raytune"]:
        config['save_dir'] = ''  # config['output_dir']
        #os.makedirs(config['save_dir'], exist_ok=True)
    else:
        config['save_dir'] = os.path.join(config['output_dir'], config['exp_name'], f"{config['run_name']}-{config['run_id']}")
        os.makedirs(config['save_dir'], exist_ok=True)
    
        config['log_dir'] = os.path.join(config['save_dir'], config['log_dir'])
        os.makedirs(config['log_dir'], exist_ok=True)
    
        config['checkpoint_dir'] = os.path.join(config['save_dir'], config['checkpoint_dir'])
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # RayTune or Tensorboard
    if config["raytune"]:
        config["tensorboard_on"] = False
        config["extra_logging"] = False
    
    # Conext embedding size
    if config["dec_context_initializer"] == "EmptyContext":
        config['context_emb_size'] = 0
    
    # Verbose
    if config["raytune"]:
        config["verbose"] = 0
        
    return config

    