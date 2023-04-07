import torch
import src.utils as utils
from src.solvers import pg_solver

import os
import itertools

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
import optuna

import time


#####################################################
# Experiment v2                                     #
#####################################################

def paths_for_instances(num_vars=20, data_path='data/rand'):
    '''
    Returs the path for every instance with num_vars variables in the data_path.
    '''
    
    data_path = os.path.abspath(os.path.join(data_path, str(f'{num_vars:04d}')))
    paths = []
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            paths.append(os.path.join(root, filename))
    paths = sorted(paths)
    return paths


def node2vec_tune(config):
    '''
    Wrapper to handle the node2vec function with raytune.
    '''
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
                       save_path=config['n2v_dir'],
                       file_name=config['n2v_filename'],
                       num_workers=config['n2v_workers'],
                       raytune=config['n2v_raytune'],
                       verbose=config['n2v_verbose'])


def n2v_hypersearch(instance_dir,
                    n2v_dim=128,
                    n2v_num_epochs=100,
                    exp_name='node2vec',
                    raytune_trials=35,
                    raytune_dir='hypersearch',
                    grace_period=5,
                    scheduler_max_t=50,
                    resources_per_trail={"cpu": 6, "gpu": 0.5}):
    
    def define_by_run_func(trial):
        trial.suggest_categorical('n2v_batch_size', [8, 16, 32, 64, 128])
        trial.suggest_float('n2v_lr', 1e-4, 1e-1, log=True) 
        trial.suggest_categorical('n2v_p', [0.25, 0.50, 1, 2, 4])
        trial.suggest_categorical('n2v_q', [0.25, 0.50, 1, 2, 4])
        
        config = {'n2v_dim': n2v_dim,  #128
                  'n2v_walk_len': 20,  #80
                  'n2v_context_size': 10,  #10
                  'n2v_walks_per_node': 10,  #10
                  #'n2v_p': 1,
                  #'n2v_q': 1,
                  #'n2v_batch_size': 32,
                  #'n2v_lr': 0.01,
                  'n2v_num_epochs': n2v_num_epochs,
                  'n2v_workers': 0,  # {0, 1, 2, 3, 4}
                  
                  'data_dir': instance_dir,
                  'gpu': True}
        
        config['n2v_verbose'] = 0  # {0, 1, 2}
        config['n2v_raytune'] = True 
        config['n2v_dir'] = None
        config['n2v_filename'] = None
        
        return config
    
    
    #torch.cuda.empty_cache()
    sampler = optuna.samplers.TPESampler(n_startup_trials=10,
                                         n_ei_candidates=24,
                                         multivariate=True)

    search_alg = OptunaSearch(sampler=sampler,
                              space=define_by_run_func,
                              mode='min',
                              metric='loss')

    scheduler = ASHAScheduler(grace_period=grace_period,
                              max_t=scheduler_max_t)

    tune_config = tune.TuneConfig(mode='min',
                                  metric="loss",
                                  num_samples=raytune_trials,
                                  search_alg=search_alg,
                                  scheduler=scheduler,
                                  max_concurrent_trials=None)

    reporter = tune.CLIReporter()
    reporter.add_parameter_column('n2v_dim')
    reporter.add_parameter_column('n2v_batch_size')
    reporter.add_parameter_column('n2v_lr')
    reporter.add_parameter_column('n2v_p')
    reporter.add_parameter_column('n2v_q')
    reporter.add_metric_column('epoch')
    reporter.add_metric_column('loss')

    run_config = air.RunConfig(local_dir=raytune_dir,
                               name=exp_name,
                               progress_reporter=reporter,  # None
                               log_to_file=True)

    # We have 1 GPU and 12 cpus, this will run 2 concurrent trials at a time.
    trainable_with_cpu_gpu = tune.with_resources(node2vec_tune, resources_per_trail)
    tuner = tune.Tuner(trainable_with_cpu_gpu,
                        tune_config=tune_config,
                        run_config=run_config)

    results = tuner.fit()
    

#####################################################
# Running the experiment                            #
#####################################################

#lista = [20, 30, 40, 50, 60, 70, 80, 90, 100]
#for i in lista:
num_vars = 20
data_path = 'data/rand'
raytune_dir="hypersearch"

n2v_dim=128
n2v_num_epochs=30
n2v_raytune_trials=35
n2v_grace_period=5
n2v_scheduler_max_t=25
n2v_resources_per_trail={"cpu": 6, "gpu": 0.5}
n2v_exp_name='node2vec'
n2v_dir = 'n2v_emb'

pg_batch_size=32
pg_raytune_trials=35
pg_grace_period=5
pg_scheduler_max_t=25
pg_resources_per_trail={"cpu": 6, "gpu": 0.5}
pg_exp_name='pg_solver'

output_dir = 'outputs'
log_dir = 'logs'

paths = paths_for_instances(num_vars, data_path)
#paths = ['/home/omargp/Documents/Code/Learning-SAT-Solvers/data/rand/0020/0060/rand_n=0020_k=03_m=0060_i=01.cnf']


#####################################################
# First step:                                       #
# Run hyperparameters search for node2vec emb       #
#####################################################

for i, instance_dir in enumerate(paths):
    tail = os.path.split(instance_dir)[1]
    instance_filename = os.path.splitext(tail)[0]
    exp_path = os.path.join(n2v_exp_name, str(n2v_dim), instance_filename)

    torch.cuda.empty_cache()
    n2v_hypersearch(instance_dir,
                    n2v_dim=n2v_dim,
                    n2v_num_epochs=n2v_num_epochs,
                    exp_name=exp_path,
                    raytune_trials=n2v_raytune_trials,
                    raytune_dir=raytune_dir,
                    grace_period=n2v_grace_period,
                    scheduler_max_t=n2v_scheduler_max_t,
                    resources_per_trail=n2v_resources_per_trail)
        
    
#####################################################
# Second step:                                      #
# Build node2vec emb with the best hyperparameters  #
#####################################################

node2vec_dir = os.path.join(n2v_dir, str(n2v_dim))
os.makedirs(node2vec_dir, exist_ok=True)


for i, instance_dir in enumerate(paths):
    tail = os.path.split(instance_dir)[1]
    instance_filename = os.path.splitext(tail)[0]
    exp_path = os.path.join(raytune_dir, n2v_exp_name, str(n2v_dim), instance_filename)
    
    # Load best config for this instance
    print(f"\nLoading results from {exp_path} ...")
    restored_tuner = tune.Tuner.restore(path=exp_path,
                                        trainable=node2vec_tune)
    results = restored_tuner.get_results()
    best_config = results.get_best_result(metric="loss", mode="min").config
    
    best_config['n2v_verbose'] = 1
    best_config['n2v_raytune'] = False
    best_config['n2v_dir'] = node2vec_dir
    best_config['n2v_filename'] = f'{instance_filename}'

    torch.cuda.empty_cache()
    node2vec_tune(best_config)


#####################################################
# Third step:                                       #
# Run hyperparameters search for pg_solver          #
#####################################################

    
#####################################################
# Fourth step:                                       #
# Run pg_solver with the best hyperparameters       #
#####################################################
