from src.solvers import pg_solver
import os

import torch
import src.utils as utils
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import optuna

############################################
# This example runs a hyperparameters search
# with optuna and raytune.
############################################

config = {
    # Encoder
    "node2vec": True,
    "n2v_dir": os.path.abspath("n2v_emb"), 
    "n2v_dim": 64,
    "n2v_pretrained": True,
    "n2v_walk_len": 10,
    "n2v_context_size": 5,
    "n2v_walks_per_node": 5,
    "n2v_p": 1,
    "n2v_q": 1,
    "n2v_batch_size": 32,
    "n2v_lr": 0.01,
    "n2v_num_epochs": 150,
    "n2v_workers": 0,  # (int). {0, 1, 2, 3, 4}
    "n2v_verbose": 0,  # (int). {0, 1, 2}

    # Initializers
    "dec_var_initializer": "Node2VecVar",  # (str). {"BasicVar", "Node2VecVar"}
    "dec_context_initializer": "EmptyContext",  # (str). {"EmptyContext", "Node2VecContext"}

    # Embeddings
    "var_emb_size": 128,
    "assignment_emb_size": 64,
    "context_emb_size": 128,
    "model_dim": 128,  # model_dim must be divisible by num_heads

    # Architecture
    "decoder": 'Transformer',  # (str). {'GRU', 'LSTM', "Transformer"}
    "num_layers": 2,  
    "output_size": 1,  # (int). Decoder output size: {1, 2}
    "dropout": 0,

    "hidden_size": 128,  # for RNN
    "trainable_state": True,  # for RNN

    "num_heads": 2,  # for Transformer. model_dim must be divisible by num_heads
    "dense_size":128,  # for Transformer

    # Training
    "num_samples": 11200,  #4000
    "accumulation_episodes": 1,
    "batch_size": 32,  #10
    "permute_vars": True,
    "permute_seed": None,  # (int). e.g.: 2147483647
    "clip_grad": 1.0,  # {None, float} e.g.:0.00015.
    "lr": 0.00015,  # (float). e.g.: 0.00015. or 0.00001

    # Baseline
    "baseline": 'sample',  # {None, 'greedy', 'sample'. 'ema'}
    "alpha_ema": 0.99,  # (float). 0 <= alpha <= 1. EMA decay.
    "k_samples": 10,  # (int). k >= 1. Number of samples used to obtain the sample baseline value.

    # Exploration
    "logit_clipping": None,  # {None, int >= 1}
    "logit_temp": None,  # {None, float >= 1}. Useful for improve exploration in evaluation.
    "entropy_estimator": 'crude',  # (str). {'crude', 'smooth'}
    "beta_entropy": 0.03,  # (float). beta >= 0.

    # Misc
    "sat_stopping": False,
    "log_interval": 20,
    "eval_interval": 50,
    "eval_strategies": [0, 32],
    "tensorboard_on": True,
    "extra_logging": False,
    "raytune": True,
    "data_dir": os.path.abspath('data/rand/0020/0080/rand_n=0020_k=03_m=0080_i=01.cnf'),
    "verbose": 0,  # {0, 1, 2}. If raytune is True, then verbose is set to 0.

    "log_dir": 'logs',
    "output_dir": 'outputs',
    "exp_name": 'exp_tune1',
    "run_name": 'run',
    "gpu": True,
    "checkpoint_dir": 'checkpoints'}

# Ensure the n2v embedding exists at n2v_dir/n2v_dim before running raytune.
node2vec_dir = os.path.join(config['n2v_dir'], str(config['n2v_dim']))
os.makedirs(node2vec_dir, exist_ok=True)
tail = os.path.split(config['data_dir'])[1]
node2vec_filename = os.path.splitext(tail)[0]
node2vec_file = os.path.join(node2vec_dir, node2vec_filename + ".pt")
n2v_emb_exist = False
if config['n2v_pretrained']:  # Tries to load pretrained embeddings
    if os.path.isfile(node2vec_file):
        n2v_emb_exist = True
if not n2v_emb_exist:  # Runs n2v algorithm if not pretrained or emb not found
    device = 'cpu'
    if config['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = utils.node2vec(dimacs_path=config['data_dir'],
                        device=device,
                        embedding_dim=config['n2v_dim'],
                        walk_length=config['n2v_walk_len'],
                        context_size=config['n2v_context_size'],
                        walks_per_node=config['n2v_walks_per_node'],
                        p=config['n2v_p'],
                        q=config['n2v_q'],
                        batch_size=config['n2v_batch_size'],
                        lr=config['n2v_lr'],
                        num_epochs=config['n2v_num_epochs'],
                        save_path=node2vec_dir,
                        file_name=node2vec_filename,
                        num_workers=config['n2v_workers'],
                        verbose=config['n2v_verbose'])

# Search space
config["k_samples"] = tune.qrandint(2, 32, 2)  # Round to multiples of 2 (includes 32)
config["entropy_estimator"] = tune.choice(['crude', 'smooth'])
config["beta_entropy"] = tune.choice([0.01, 0.02, 0.03])
config["output_size"] = tune.choice([1, 2])
config["lr"] = tune.qloguniform(1e-6, 1e-4, 5e-7)  # Round to multiples of 0.0000005

search_alg = OptunaSearch(sampler=optuna.samplers.TPESampler(n_startup_trials=10,
                                                             n_ei_candidates=24,
                                                             multivariate=True))

scheduler = ASHAScheduler(grace_period=6)

tune_config = tune.TuneConfig(mode='max',
                              metric="num_sat_sample_32",
                              num_samples=10,
                              search_alg=search_alg,
                              scheduler=scheduler,
                              max_concurrent_trials=None)

reporter = tune.CLIReporter()#metric_columns=['episode', 'samples', 'num_sat_greedy', 'num_sat_sample_32'],
                            #parameter_columns=['batch_size', 'lr'])
reporter.add_parameter_column('decoder')
reporter.add_parameter_column('lr')
reporter.add_parameter_column('batch_size')     
reporter.add_metric_column('samples')
reporter.add_metric_column('episode')
reporter.add_metric_column('num_sat_greedy')
reporter.add_metric_column('num_sat_sample_32')

run_config = air.RunConfig(local_dir="experiments",
                           name=config["exp_name"],
                           progress_reporter=reporter,  # None
                           log_to_file=True)

# We have 1 GPU and 12 cpus, this will run 2 concurrent trials at a time.
trainable_with_cpu_gpu = tune.with_resources(pg_solver, {"cpu": 6, "gpu": 0.5})
tuner = tune.Tuner(trainable_with_cpu_gpu,
                   tune_config=tune_config,
                   run_config=run_config,
                   param_space=config)

results = tuner.fit()
