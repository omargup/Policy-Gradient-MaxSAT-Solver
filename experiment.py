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

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

############################################
# Experiment
############################################

def hyper_search(instance_dir,
                 num_samples=11200,
                 batch_size=32,
                 exp_name='exp',
                 local_dir="experiments"):
    
    assumps = list(itertools.product([True, False], repeat=3))
    for assump in assumps:
        # Assumptions
        transformer_assump, node2vec_assump, baseline_assump = assump
        exp_name1 = f'{exp_name}/t-{int(transformer_assump)}_n-{int(node2vec_assump)}_b-{int(baseline_assump)}'


        preconf = {
            #'node2vec': True,
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
            
            'exp_name': exp_name1,
            'data_dir': instance_dir,
            'gpu': True,
        } 

        
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
            torch.cuda.empty_cache()
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


        def define_by_run_func(trial):
            # Constants
            config = {
                # Misc
                "sat_stopping": False,
                "log_interval": 20,
                "eval_interval": 20,
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

            # Search space
            # Input and embeddings
            if node2vec_assump:
                config["dec_var_initializer"] = "Node2VecVar"
                dec_context_initializer = trial.suggest_categorical("dec_context_initializer", ["EmptyContext", "Node2VecContext"])
                config["node2vec"] = True
                config["n2v_dir"] = preconf["n2v_dir"]
                config["n2v_dim"] = preconf["n2v_dim"]
                config["n2v_pretrained"] = preconf["n2v_pretrained"]
                config["n2v_walk_len"] = preconf["n2v_walk_len"]
                config["n2v_context_size"] = preconf["n2v_context_size"]
                config["n2v_walks_per_node"] = preconf["n2v_walks_per_node"]
                config["n2v_p"] = preconf["n2v_p"]
                config["n2v_q"] = preconf["n2v_q"]
                config["n2v_batch_size"] = preconf["n2v_batch_size"]
                config["n2v_lr"] = preconf["n2v_lr"]
                config["n2v_num_epochs"] = preconf["n2v_num_epochs"]
                config["n2v_workers"] = preconf["n2v_workers"]
                config["n2v_verbose"] = preconf["n2v_verbose"]
            else:
                config["node2vec"] = False
                config["dec_var_initializer"] = "BasicVar"
                config["dec_context_initializer"] = "EmptyContext"
                dec_context_initializer = "EmptyContext"
            trial.suggest_categorical("var_emb_size", [64, 128, 256])
            trial.suggest_categorical("assignment_emb_size", [64, 128, 256])
            if dec_context_initializer == "Node2VecContext":
                trial.suggest_categorical("context_emb_size", [64, 128, 256])
            else: # EmptyContext
                config["context_emb_size"] = 0
            trial.suggest_categorical("model_dim", [64, 128, 256, 512])

            # Decoder
            if transformer_assump:
                config["decoder"] = "Transformer"
                trial.suggest_categorical("num_heads", [1, 2, 4, 8])
                trial.suggest_categorical("dense_size", [64, 128, 256, 512, 768, 1024])
            else:  # rnn decoder
                trial.suggest_categorical("decoder", ['GRU', 'LSTM'])
                trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 768, 1024])
                trial.suggest_categorical("trainable_state", [True, False])
            trial.suggest_int("num_layers", 1, 6, log=True)
            trial.suggest_categorical("output_size", [1, 2])
            trial.suggest_float("dropout", 0, 0.3, step=0.05)

            # Training
            config["num_samples"] = num_samples
            config["accumulation_episodes"] = 1
            config["batch_size"] = batch_size
            config["permute_vars"] = True
            config["permute_seed"] = None  # 2147483647
            trial.suggest_categorical("clip_grad", [None, 0.5, 1, 1.5, 2])
            trial.suggest_float("lr", 1e-6, 1e-4, log=True)  # 0.00015   0.00001
            
            # Baseline
            if baseline_assump:
                baseline = trial.suggest_categorical("baseline", ["greedy", "sample", "ema"])
                if baseline == "sample":
                    trial.suggest_categorical("k_samples", [2, 4, 8, 16, 32])  # int, k >= 1
                elif baseline == "ema":
                    trial.suggest_float("alpha_ema", 0.95, 0.99, step=0.01)  # 0 <= alpha <= 1
            else:  # No baseline
                config["baseline"] = None

            # Exploration
            trial.suggest_categorical("logit_clipping", [None, 1, 2, 5, 10])  # {None, int >= 1}
            trial.suggest_categorical("logit_temp", [None, 1.5, 2, 2.5])  # {None, int >= 1}
            trial.suggest_categorical("entropy_estimator", ['crude', 'smooth'])
            trial.suggest_categorical("beta_entropy", [0, 0.01, 0.03, 0.05, 0.1])  # float, beta >= 0

            # Optuna defined by run flag. Don't set to False
            config["optuna_by_run"] = True
            return config
        
        torch.cuda.empty_cache()
        sampler = optuna.samplers.TPESampler(n_startup_trials=10,
                                            n_ei_candidates=24,
                                            multivariate=True)

        search_alg = OptunaSearch(sampler=sampler,
                                space=define_by_run_func,
                                mode='max',
                                metric="num_sat_sample_32")

        scheduler = ASHAScheduler(grace_period=6)

        tune_config = tune.TuneConfig(mode='max',
                                    metric="num_sat_sample_32",
                                    num_samples=2,
                                    search_alg=search_alg,
                                    scheduler=scheduler,
                                    max_concurrent_trials=None)

        reporter = tune.CLIReporter()
        reporter.add_parameter_column('decoder')
        reporter.add_parameter_column('baseline',)
        reporter.add_parameter_column('logit_clipping')
        reporter.add_parameter_column('logit_temp')
        reporter.add_parameter_column('entropy_estimator')
        reporter.add_parameter_column('beta_entropy')
        reporter.add_metric_column('samples')
        reporter.add_metric_column('episode')
        reporter.add_metric_column('num_sat_greedy')
        reporter.add_metric_column('num_sat_sample_32')

        run_config = air.RunConfig(local_dir=local_dir,
                                name=preconf['exp_name'],
                                progress_reporter=reporter,  # None
                                log_to_file=True)

        # We have 1 GPU and 12 cpus, this will run 2 concurrent trials at a time.
        trainable_with_cpu_gpu = tune.with_resources(pg_solver, {"cpu": 6, "gpu": 0.5})
        tuner = tune.Tuner(trainable_with_cpu_gpu,
                           tune_config=tune_config,
                           run_config=run_config)

        results = tuner.fit()


local_dir = "experiments"
data_path = 'data/rand'
num_vars = 40
n_path = os.path.abspath(os.path.join(data_path, str(f'{num_vars:04d}')))

#####################################################
# Running the hyperparameters searches              #
#####################################################

# # Finding all m values for that n
# m_paths = []
# for folder in os.listdir(n_path):
#     m_paths.append(os.path.join(n_path, folder))
# m_paths = sorted(m_paths)

# # For each m value
# for m_path in m_paths:
#     # Find all instances
#     n_m_paths = []
#     for n_m_file in os.listdir(m_path):
#         n_m_paths.append(os.path.join(m_path, n_m_file))
#     n_m_paths = sorted(n_m_paths)
    
#     # Get the first instance (i=1) with that m and n
#     instance_path = n_m_paths[0]
#     n, m, _ = utils.dimacs2list(instance_path)
    
#     # Run hyper search for instance i=1 with n variables and m clauses
#     exp_name = f'exp_{n:04d}/{m:04d}'
#     hyper_search(instance_dir=instance_path,
#                  num_samples= ((2*n)+m)*64,
#                  batch_size=32,
#                  exp_name=exp_name,
#                  local_dir=local_dir)
    

#####################################################
# Running pg_solver with the best hyperparameters   #
# for the rest of the instances.                    #
#####################################################

start_time = time.time()

# Finding all m values for that n
m_paths = []
for folder in os.listdir(n_path):
    m_paths.append(os.path.join(n_path, folder))
m_paths = sorted(m_paths)

# For each m value
for m_path in m_paths:
    # Find all instances
    n_m_paths = []
    for n_m_file in os.listdir(m_path):
        n_m_paths.append(os.path.join(m_path, n_m_file))
    n_m_paths = sorted(n_m_paths)
    
    # Get the first instance (i=1) with that m and n
    instance_path = n_m_paths[0]
    n, m, _ = utils.dimacs2list(instance_path)
    
    # Finding the 8 hyperparameters searches for that instace
    exp_name = f'exp_{n:04d}/{m:04d}'
    assumps = list(itertools.product([True, False], repeat=3))
    for assump in assumps:
        # Assumptions
        transformer_assump, node2vec_assump, baseline_assump = assump
        exp_path = os.path.join(local_dir, f'{exp_name}/t-{int(transformer_assump)}_n-{int(node2vec_assump)}_b-{int(baseline_assump)}')
        
        # Load best config for this configuration
        print(f"Loading results from {exp_path}...")
        restored_tuner = tune.Tuner.restore(path=exp_path,
                                            trainable=pg_solver)
        results = restored_tuner.get_results()
        best_config = results.get_best_result(metric="num_sat_sample_32", mode="max").config

        # Run pg_solver for the rest of the instances
        for i, path_i in enumerate(n_m_paths[1:6]):
            exp_path2 = f'exp_{n:04d}/{m:04d}/t-{int(transformer_assump)}_n-{int(node2vec_assump)}_b-{int(baseline_assump)}'
            run_name = f'run_i-{i+2}'
            
            # Update config
            best_config['num_samples'] = ((2 * n) + m) * 128
            best_config['sat_stopping'] = True
            best_config['log_interval'] = 20
            best_config['eval_interval'] = 20
            best_config['eval_strategies'] = [0, 32]
            best_config['tensorboard_on'] = True
            best_config['raytune'] = False
            best_config['data_dir'] = path_i
            best_config['verbose'] = 1
            best_config['log_dir'] = 'logs'
            best_config['output_dir'] = 'outputs'
            best_config['exp_name'] = exp_path2 
            best_config['run_name'] = run_name
            best_config['gpu'] = True
            best_config['checkpoint_dir'] = 'checkpoints'
            best_config['optuna_by_run'] = False
            best_config['n2v_pretrained'] = True
            
            print()
            print("-" * 80)
            print("-" * 80)
            print(f"Experiment: {exp_path2}")
            print(f"Instance: {i+2}")
            print("-" * 80)
            torch.cuda.empty_cache()
            pg_solver(best_config)

end_time = time.time()
mins, secs = epoch_time(start_time, end_time)
print(f'Total time: {mins}m {secs}s')