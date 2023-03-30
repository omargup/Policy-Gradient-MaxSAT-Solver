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
# This example runs a conditional hyperparameters search
# with optuna and raytune.
############################################

preconf = {
    'node2vec': True,
    'n2v_dir': os.path.abspath('n2v_emb'),
    'n2v_dim': 64,
    'n2v_pretrained': True,
    'n2v_walk_len': 10,
    'n2v_context_size': 5,
    'n2v_walks_per_node': 5,
    'n2v_p': 1,
    'n2v_q': 1,
    'n2v_batch_size': 32,
    'n2v_lr': 0.01,
    'n2v_num_epochs': 150,
    'n2v_workers': 0,  # {0, 1, 2, 3, 4}
    'n2v_verbose': 0,  # {0, 1, 2}
    
    'exp_name': 'exp_tune2',
    'data_dir': os.path.abspath('data/rand/0020/0080/rand_n=0020_k=03_m=0080_i=01.cnf'),
    'gpu': True,
}

def define_by_run_func(trial):
    # Conditional search space
    decoder = trial.suggest_categorical("decoder", ['GRU', 'LSTM', 'Transformer'])
    if (decoder == 'GRU') or (decoder == 'LSTM'):
        trial.suggest_int("hidden_size", 64, 1024, log=True)
        trial.suggest_categorical("trainable_state", [True, False])
    else:  # Transformer
        trial.suggest_categorical("num_heads", [1, 2, 4, 8])  # model_dim must be divisible by num_heads
        trial.suggest_int("dense_size", 64, 1024, log=True)
    
    # Search space
    trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    trial.suggest_categorical("k_samples", [2, 4, 8, 16, 32])
    trial.suggest_categorical("entropy_estimator", ['crude', 'smooth'])
    trial.suggest_categorical("beta_entropy", [0.01, 0.02, 0.03])
    
    # Constants
    config = {
    # Encoder
    "node2vec": preconf["node2vec"],
    "n2v_dir": preconf["n2v_dir"],
    "n2v_dim": preconf["n2v_dim"],
    "n2v_pretrained": preconf["n2v_pretrained"],
    "n2v_walk_len": preconf["n2v_walk_len"],
    "n2v_context_size": preconf["n2v_context_size"],
    "n2v_walks_per_node": preconf["n2v_walks_per_node"],
    "n2v_p": preconf["n2v_p"],
    "n2v_q": preconf["n2v_q"],
    "n2v_batch_size": preconf["n2v_batch_size"],
    "n2v_lr": preconf["n2v_lr"],
    "n2v_num_epochs": preconf["n2v_num_epochs"],
    "n2v_workers": preconf["n2v_workers"],
    "n2v_verbose": preconf["n2v_verbose"],

    # Initializers
    "dec_var_initializer": "Node2VecVar",  # {"BasicVar", "Node2VecVar"}
    "dec_context_initializer": "Node2VecContext",  # {"EmptyContext", "Node2VecContext"}

    # Embeddings
    "var_emb_size": 128,
    "assignment_emb_size": 64,
    "context_emb_size": 128,
    "model_dim": 128,  # model_dim must be divisible by num_heads

    # Architecture
    #"decoder": 'GRU',  # {'GRU', 'LSTM', "Transformer"}
    "num_layers": 2,  
    "output_size": 2,  # Decoder output size: 1, 2
    "dropout": 0,

    #"hidden_size": 128,  # Useful for RNN
    #"trainable_state": True,  # {False, True}. Useful for RNN

    #"num_heads": 2,  # Useful for Transformer
    #"dense_size":128,  # Useful for Transformer

    # Training
    "num_samples": 11200,  #4000
    "accumulation_episodes": 1,
    "batch_size": 32,  #10
    "permute_vars": True,
    "permute_seed": None,  # 2147483647
    "clip_grad": 1,  # {None, float}
    #"lr": 0.00001,  # 0.00015   0.00001

    # Baseline
    "baseline": 'sample',  # {None, 'greedy', 'sample'. 'ema'}
    "alpha_ema": 0.99,  # 0 <= alpha <= 1. EMA decay, useful if baseline == 'ema'
    #"k_samples": 10,  # int, k >= 1. Number of samples used to obtain the baseline value, useful if baseline == 'sample'

    # Exploration
    "logit_clipping": None,  # {None, int >= 1}
    "logit_temp": None,  # {None, float >= 1}. Useful for improve exploration in evaluation.
    #"entropy_estimator": 'crude',  # {'crude', 'smooth'}
    #"beta_entropy": 0.03,  # float, beta >= 0.

    # Misc
    "sat_stopping": False,  # {True, False}. Stop when num_sat is equal with the num of clauses.
    "log_interval": 20,
    "eval_interval": 50,
    "eval_strategies": [0, 32],
    "tensorboard_on": True,
    "extra_logging": False,  # log TrainableState's weights
    "raytune": True,
    "data_dir": preconf['data_dir'],
    "verbose": 0,  # {0, 1, 2}. If raytune is True, then verbose is set to 0.

    "log_dir": 'logs',
    "output_dir": 'outputs',
    "exp_name": preconf['exp_name'],
    "run_name": 'run',
    "gpu": preconf['gpu'],
    "checkpoint_dir": 'checkpoints'} 

    # Optuna defined by run flag. Don't set to False
    config["optuna_by_run"] = True

    return config


# Ensure the n2v embedding exists at n2v_dir/n2v_dim before running raytune.
node2vec_dir = os.path.join(preconf['n2v_dir'], str(preconf['n2v_dim']))
os.makedirs(node2vec_dir, exist_ok=True)
tail = os.path.split(preconf['data_dir'])[1]
node2vec_filename = os.path.splitext(tail)[0]
node2vec_file = os.path.join(node2vec_dir, node2vec_filename + ".pt")
n2v_emb_exist = False
if preconf['n2v_pretrained']:  # Tries to load pretrained embeddings
    if os.path.isfile(node2vec_file):
        n2v_emb_exist = True
if not n2v_emb_exist:  # Runs n2v algorithm if not pretrained or emb not found
    device = 'cpu'
    if preconf['gpu']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _ = utils.node2vec(dimacs_path=preconf['data_dir'],
                        device=device,
                        embedding_dim=preconf['n2v_dim'],
                        walk_length=preconf['n2v_walk_len'],
                        context_size=preconf['n2v_context_size'],
                        walks_per_node=preconf['n2v_walks_per_node'],
                        p=preconf['n2v_p'],
                        q=preconf['n2v_q'],
                        batch_size=preconf['n2v_batch_size'],
                        lr=preconf['n2v_lr'],
                        num_epochs=preconf['n2v_num_epochs'],
                        save_path=node2vec_dir,
                        file_name=node2vec_filename,
                        num_workers=preconf['n2v_workers'],
                        verbose=preconf['n2v_verbose'])
        
        
sampler = optuna.samplers.TPESampler(n_startup_trials=10,
                                     n_ei_candidates=24,
                                     multivariate=True)

search_alg = OptunaSearch(sampler=sampler,
                          space=define_by_run_func,
                          mode='max',
                          metric="num_sat_sample_32",)

scheduler = ASHAScheduler(grace_period=5)

tune_config = tune.TuneConfig(mode='max',
                              metric="num_sat_sample_32",
                              num_samples=10,
                              search_alg=search_alg,
                              scheduler=scheduler,
                              max_concurrent_trials=None)

reporter = tune.CLIReporter()
reporter.add_parameter_column('decoder')
reporter.add_parameter_column('lr')
reporter.add_parameter_column('batch_size')     
reporter.add_metric_column('samples')
reporter.add_metric_column('episode')
reporter.add_metric_column('num_sat_greedy')
reporter.add_metric_column('num_sat_sample_32')

run_config = air.RunConfig(local_dir='experiments',
                           name=preconf['exp_name'],
                           progress_reporter=reporter,  # None
                           log_to_file=True)

# We have 1 GPU and 12 cpus, this will run 2 concurrent trials at a time.
trainable_with_cpu_gpu = tune.with_resources(pg_solver, {"cpu": 6, "gpu": 0.5})
tuner = tune.Tuner(trainable_with_cpu_gpu,
                   tune_config=tune_config,
                   run_config=run_config)

results = tuner.fit()