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
#import pprint as pp

#####################################################
# Experiment v3                                     #
#####################################################

def paths_for_instances(num_vars=20, data_path='data/rand'):
    '''
    Returs the path for every instance in the data_path with num_vars variables.
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
                    resources_per_trial={"cpu": 6, "gpu": 0.5}):
    
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
    trainable_with_cpu_gpu = tune.with_resources(node2vec_tune, resources_per_trial)
    tuner = tune.Tuner(trainable_with_cpu_gpu,
                        tune_config=tune_config,
                        run_config=run_config)

    results = tuner.fit()
 
 
def pg_hypersearch(instance_dir,
                   n2v_exp_name,
                   n2v_dir,
                   num_samples=11200,
                   batch_size=32,
                   exp_name='exp',
                   raytune_trials=35,
                   raytune_dir='hypersearch',
                   grace_period=5,
                   scheduler_max_t=5000,
                   resources_per_trial={"cpu": 6, "gpu": 0.5}):
    
    assumps = ['arch', 'baseline', 'permute', 'entropy', 'node2vec']
    for assumption in assumps:
        exp_name1 = f'{exp_name}/{assumption}'
        
        # Load best node2vec config for this instance
        n2v_exp_path = os.path.join(raytune_dir, n2v_exp_name)
        print(f"\nLoading best node2vec config from {n2v_exp_path} ...")
        restored_tuner = tune.Tuner.restore(path=n2v_exp_path,
                                            trainable=node2vec_tune)
        results = restored_tuner.get_results()
        n2v_config = results.get_best_result(metric="loss", mode="min").config
        
        # Modify node2vec config
        n2v_config['n2v_verbose'] = 0
        n2v_config['n2v_dir'] = n2v_dir
        del n2v_config['n2v_raytune']
        del n2v_config['n2v_filename']
    
        def define_by_run_func(trial):
            # Constants
            config = {
                # Misc
                "sat_stopping": False,
                "log_interval": 10,
                "eval_interval": 10,
                "eval_strategies": [32],
                "tensorboard_on": False,
                "extra_logging": False,  # log TrainableState's weights
                "raytune": True,
                "data_dir": instance_dir,
                "verbose": 0,  # {0, 1, 2}. If raytune is True, then verbose is set to 0.

                "log_dir": '',  #'logs',
                "output_dir": '',  #'outputs',
                "exp_name": exp_name1,
                "run_name": 'run',
                "gpu": True,
                "checkpoint_dir": 'checkpoints'}
            
            # Search space
            # Input and embeddings
            if assumption == 'node2vec':
                node2vec = trial.suggest_categorical("node2vec", [True, False])
            else:
                node2vec = config["node2vec"] = False
                
            if not node2vec:
                config["dec_var_initializer"] = "BasicVar"
                dec_context_initializer = config["dec_context_initializer"] = "EmptyContext"
            else:
                config["n2v_dir"] = n2v_config['n2v_dir']
                config["n2v_dim"] = n2v_config["n2v_dim"]
                config["n2v_pretrained"] = True
                config["n2v_walk_len"] = n2v_config["n2v_walk_len"]
                config["n2v_context_size"] = n2v_config["n2v_context_size"]
                config["n2v_walks_per_node"] = n2v_config["n2v_walks_per_node"]
                config["n2v_p"] = n2v_config["n2v_p"]
                config["n2v_q"] = n2v_config["n2v_q"]
                config["n2v_batch_size"] = n2v_config["n2v_batch_size"]
                config["n2v_lr"] = n2v_config["n2v_lr"]
                config["n2v_num_epochs"] = n2v_config["n2v_num_epochs"]
                config["n2v_workers"] = n2v_config["n2v_workers"]
                config["n2v_verbose"] = n2v_config['n2v_verbose']
                
                config["dec_var_initializer"] = "Node2VecVar"
                dec_context_initializer = trial.suggest_categorical("dec_context_initializer", ["EmptyContext", "Node2VecContext"])
                
            trial.suggest_categorical("var_emb_size", [64, 128, 256])
            trial.suggest_categorical("assignment_emb_size", [64, 128, 256])
            if dec_context_initializer == "Node2VecContext":
                trial.suggest_categorical("context_emb_size", [32, 64, 128, 256])
            else: # EmptyContext
                config["context_emb_size"] = 0
            trial.suggest_categorical("model_dim", [64, 128, 256, 512])
                
            # Decoder
            #if assumption == 'arch':
            #    decoder = trial.suggest_categorical("decoder", ["Transformer", "GRU", "LSTM"])
            #else:
            #    decoder = config["decoder"] = "Transformer"
            decoder = trial.suggest_categorical("decoder", ["Transformer", "GRU", "LSTM"])
                
            if decoder == "Transformer":
                trial.suggest_categorical("num_heads", [1, 2, 4])
                trial.suggest_categorical("dense_size", [64, 128, 256, 512])
            else:  # rnn decoder
                trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 768, 1024])
                trial.suggest_categorical("trainable_state", [True, False])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            #trial.suggest_categorical("output_size", [1, 2])
            config["output_size"] = 1
            dropout_flag = ((decoder == "GRU") and (num_layers > 1)) or ((decoder == "LSTM") and (num_layers > 1))
            if (decoder == "Transformer") or dropout_flag:
                trial.suggest_float("dropout", 0, 0.3, step=0.05)
            else:
                config["dropout"] = 0 
            
            # Training
            config["num_samples"] = num_samples
            config["accumulation_episodes"] = 1
            config["batch_size"] = batch_size
            
            # Training - Permutation
            if (assumption == 'arch') or (assumption == 'baseline'):
                config["vars_permutation"] = 'fixed'
            else:
                trial.suggest_categorical("vars_permutation", ["fixed", "importance", "random", "batch"])
            
            # Training - Baseline
            if assumption == 'arch':
                baseline = config["baseline"] = 'zero'
            else:
                baseline = trial.suggest_categorical("baseline", ['zero', 'greedy', 'sample', 'ema'])
                
            if baseline == "sample":
                trial.suggest_categorical("k_samples", [2, 4, 8, 16, 32])  # int, k >= 1
                trial.suggest_float("sampling_temp", 1.0, 2.6, step=0.2)  # {float >= 1}
            elif baseline == "ema":
                trial.suggest_float("alpha_ema", 0.95, 0.99, step=0.01)  # 0 <= alpha <= 1
        
            trial.suggest_categorical("clip_grad", [0, 0.5, 1, 1.5, 2])
            trial.suggest_float("lr", 1e-6, 1e-4, log=True)  # 0.00015   0.00001
            
            # Exploration 
            trial.suggest_categorical("logit_clipping", [0, 1, 2, 5, 10])  # {int >= 0}
            trial.suggest_float("logit_temp", 1.0, 2.6, step=0.2)  # {float >= 1}
            
            # Exploration - Entropy
            if (assumption == 'arch') or (assumption == 'baseline') or (assumption == 'permute'):
                config["entropy_estimator"] = 'crude'
                config["beta_entropy"] = 0
            else:  # assumption == 'entropy' or assumption == 'node2vec'
                trial.suggest_categorical("entropy_estimator", ['crude', 'smooth'])
                trial.suggest_categorical("beta_entropy", [0, 0.01, 0.03, 0.05, 0.1])  # float, beta >= 0
            
            
            # Flag for optuna defined by run. Don't set to False.
            config["optuna_by_run"] = True
            return config
        
        
        #torch.cuda.empty_cache()
        sampler = optuna.samplers.TPESampler(n_startup_trials=10,
                                            n_ei_candidates=24,
                                            multivariate=True)

        search_alg = OptunaSearch(sampler=sampler,
                                space=define_by_run_func,
                                mode='max',
                                metric='num_sat_eval')

        scheduler = ASHAScheduler(time_attr='samples',
                                grace_period=grace_period,
                                max_t=scheduler_max_t,
                                stop_last_trials=True)

        tune_config = tune.TuneConfig(mode='max',
                                    metric='num_sat_eval',
                                    num_samples=raytune_trials,
                                    search_alg=search_alg,
                                    scheduler=scheduler,
                                    max_concurrent_trials=None)

        reporter = tune.CLIReporter()
        reporter.add_parameter_column('decoder')
        reporter.add_parameter_column('baseline')
        reporter.add_parameter_column('vars_permutation')
        reporter.add_parameter_column('node2vec')
        reporter.add_metric_column('samples')
        reporter.add_metric_column('episode')
        reporter.add_metric_column('num_sat')
        reporter.add_metric_column('num_sat_eval')

        run_config = air.RunConfig(local_dir=raytune_dir,
                                name=exp_name1,
                                progress_reporter=reporter,  # None
                                log_to_file=True)#,
                                #checkpoint_config=air.CheckpointConfig(checkpoint_score_attribute="num_sat",
                                #                                       checkpoint_score_order="max",
                                #                                       num_to_keep=1))

        # We have 1 GPU and 8 cpus, this will run 2 concurrent trials at a time.
        trainable_with_cpu_gpu = tune.with_resources(pg_solver, resources_per_trial)
        tuner = tune.Tuner(trainable_with_cpu_gpu,
                            tune_config=tune_config,
                            run_config=run_config)
        results = tuner.fit()
    
     
#####################################################
# Running the experiment                            #
#####################################################

lista = [30, 40, 50]
for i in lista:
    num_vars = i
    data_path = 'data/rand'
    raytune_dir="hypersearch"

    n2v_dim=128
    n2v_num_epochs=30
    n2v_raytune_trials=35
    n2v_grace_period=5
    n2v_scheduler_max_t=25
    n2v_resources_per_trial={"cpu": 4, "gpu": 0.5}
    n2v_exp_name='node2vec'
    n2v_dir = 'node2vec_emb'

    pg_batch_size=32
    pg_raytune_trials=50  # 34 # 50
    #pg_grace_period=((2*n)+m)*8
    #pg_num_samples=((2*n)+m)*128
    #pg_scheduler_max_t=((2*n)+m)*64
    pg_resources_per_trial={"cpu": 8, "gpu": 1.0}
    pg_exp_name='pg_solver'

    output_dir = 'outputs'
    log_dir = 'logs'

    paths = paths_for_instances(num_vars, data_path)
    first_instances = []
    for inst in paths:
        if inst[-6:-4] == '01':
            first_instances.append(inst)
    paths = first_instances
    #paths = ['/home/omargp/Documents/Code/Learning-SAT-Solvers/data/rand/0020/0092/rand_n=0020_k=03_m=0092_i=01.cnf']


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
                        resources_per_trial=n2v_resources_per_trial)
            
        
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
        print(f"\nLoading best node2vec config from {exp_path} ...")
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

    for i, instance_dir in enumerate(paths):
        # Get instance's filename
        tail = os.path.split(instance_dir)[1]
        instance_filename = os.path.splitext(tail)[0]
        
        # Ensure the n2v embedding exists at n2v_dir/n2v_dim before running raytune.
        node2vec_dir = os.path.join(n2v_dir, str(n2v_dim))
        n2v_file = os.path.join(node2vec_dir, instance_filename + ".pt")
        if not os.path.isfile(n2v_file):
            raise Exception(f"No node2vec emb was found at {n2v_file}.")
        
        # n2v experiment path + name
        n2v_exp_path = os.path.join(n2v_exp_name, str(n2v_dim), instance_filename)
        
        # pg_solver experiment path + name
        n, m, _ = utils.dimacs2list(instance_dir)
        exp_path = f'{pg_exp_name}/{n:04d}/{m:04d}/{instance_filename}'
        
        # Run the hyperparameters search for this instance
        torch.cuda.empty_cache()
        pg_hypersearch(instance_dir,
                    n2v_exp_name = n2v_exp_path,
                    n2v_dir = os.path.abspath(n2v_dir),
                    num_samples=((2*n)+m)*128,
                    batch_size=pg_batch_size,
                    exp_name=exp_path,
                    raytune_trials=pg_raytune_trials,
                    raytune_dir=raytune_dir,
                    grace_period=((2*n)+m)*4,
                    scheduler_max_t=((2*n)+m)*64,
                    resources_per_trial=pg_resources_per_trial)
        
    
    #####################################################
    # Fourth step:                                      #
    # Run pg_solver with the best hyperparameters       #
    # for the rest of the instances.                    #
    #####################################################
