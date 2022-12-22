import os
import time
import json
import argparse
import torch

def get_config(new_config=None):
    config = {
        # Architecture
        "cell": 'GRU',  # 'GRU', 'LSTM'
        "hidden_size": 128,  
        "num_layers": 1,  
        "clip_logits_c": 0,
        "output_size": 1,  #Decoder output size: 1, 2

        # Training
        "num_episodes": 5000,
        "accumulation_episodes": 1,
        "baseline": None,  # None, -1, 1, 2, 3, 4, 5
        "batch_size": 1,
        "permute_vars": False,
        "permute_seed": None,  # 2147483647
        "clip_grad": 1,
        "entropy_weight": 10,
        "lr": 0.00015,  # 0.00015

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
        "extra_logging": False,  # log TrainableState's weights
        "raytune": False,
        "data_dir": None,

        "log_dir": 'logs',
        "output_dir": 'outputs',
        "exp_name": 'exp',
        "run_name": 'run',
        "gpu": True,
        "checkpoint_dir": None} # <--------- missing
    
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
    
    # Saving configuration
    with open(os.path.join(config['save_dir'], "config.json"), 'w') as f:
        json.dump(config, f, indent=True)
    
    return config

    