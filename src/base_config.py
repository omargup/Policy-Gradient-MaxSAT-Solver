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
            "node2vec": False,  # {False, True}
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
            "dec_var_initializer": "BasicVar",  # {"BasicVar", "Node2VecVar"}
            "dec_context_initializer": "EmptyContext",  # {"EmptyContext", "Node2VecContext"}

            # Embeddings
            "var_emb_size": 64,
            "assignment_emb_size": 64,
            "context_emb_size": 64,
            "model_dim": 256,  

            # Architecture
            "decoder": 'GRU',  # {'GRU', 'LSTM', "Transformer"}
            "num_layers": 1,  
            "output_size": 2,  #Decoder output size: 1, 2
            "dropout": 0,

            "hidden_size": 128,  #hidden_size if RNN
            "trainable_state": False,  # {False, True}

            "num_heads": 2,
            "dense_size": 256,

            # Training
            "num_samples": 15000,
            "accumulation_episodes": 1,
            "batch_size": 1,
            "permute_vars": True,
            "permute_seed": None,  # 2147483647
            "clip_grad": 1,
            "lr": 0.00015,  # 0.00015

            # Baseline
            "baseline": None,  # {None, 'greedy', 'sample'. 'ema'}
            "alpha_ema": 0.99,  # 0 <= alpha <= 1. EMA decay.
            "k_samples": 10,  # int, k >= 1. Number of samples used to obtain the baseline value
            
            # Exploration
            "logit_clipping": None,  # {None, int >= 1}
            "logit_temp": None,  # {None, float >= 1}
            "entropy_estimator": 'crude',  # {'crude', 'smooth'}
            "beta_entropy": 0,  # float, beta >= 0.

            # Regularization
            "early_stopping": False,
            "patience": 5,
            "entropy_value": 0,

            "log_interval": 100,
            "eval_interval": 200,
            "eval_strategies": [0, 5],
            "tensorboard_on": True,
            "extra_logging": False,  # log TrainableState's weights
            "raytune": False,
            "data_dir": None,
            "verbose": 1,  # {0, 1, 2}. If raytune is True, then verbose is set to 0.

            "log_dir": 'logs',
            "output_dir": 'outputs',
            "exp_name": 'exp',
            "run_name": 'run',
            "gpu": True,
            "checkpoint_dir": None}
    
        # Update default config
        for key in new_config:
            config[key] = new_config[key]
    
    if config['run_name'] is None:
        config['run_name'] = 'run'
    if config['exp_name'] is None:
        config['exp_name'] = 'exp'
    if config['log_dir'] is None:
        config['log_dir'] = 'logs'
    if config['output_dir'] is None:
        config['output_dir'] = 'outputs'
    
    # Generate run_id
    config['run_id'] = time.strftime("%Y%m%dT%H%M%S")

    # Set output directory
    config['save_dir'] = os.path.join(config['output_dir'], config['exp_name'], f"{config['run_name']}-{config['run_id']}")
    os.makedirs(config['save_dir'], exist_ok=True)

    # Conext embedding size
    if config["dec_context_initializer"] == "EmptyContext":
        config['context_emb_size'] = 0
    
    # Verbose
    if config["raytune"]:
        config["verbose"] = 0
    
    # Saving configuration
    with open(os.path.join(config['save_dir'], "config.json"), 'w') as f:
        json.dump(config, f, indent=True)
    
    return config

    