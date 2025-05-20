import numpy as np
import time
import copy
import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd 

from pose_retargeting.retargeting_problem import PoseRetargetingProblem
from pose_retargeting.evolutionary_runner import EvolutionaryRunner


class BayesianHyperparameterOptimizer:
    def __init__(self, base_config, source_train, target_train, log_func):
        self.base_config = copy.deepcopy(base_config)
        self.source_train = source_train
        self.target_train = target_train
        self.log_func = log_func

        self.bo_config = self.base_config['bayesian_optimizer_settings']
        self.search_space_config_raw = self.bo_config['search_space']
        
        self.hyperopt_space = self._prepare_hyperopt_space()
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self.trial_count = 0

    def _prepare_hyperopt_space(self):
        space = {}
        for name, conf in self.search_space_config_raw.items():
            stype = conf['type']
            if stype == "uniform":
                space[name] = hp.uniform(name, conf['low'], conf['high'])
            elif stype == "quniform":
                space[name] = hp.quniform(name, conf['low'], conf['high'], conf['q'])
            elif stype == "loguniform":
                space[name] = hp.loguniform(name, np.log(conf['low']), np.log(conf['high']))
            elif stype == "choice":
                space[name] = hp.choice(name, conf['options'])
            elif stype == "randint":
                space[name] = hp.randint(name, conf['upper'])
            else:
                self.log_func(f"[BO Warning] Unsupported hyperparameter type '{stype}' for '{name}' in Hyperopt search_space.")
        
        if not space:
            self.log_func("[BO Error] No valid dimensions found for Hyperopt. Check 'search_space' in config.")
            raise ValueError("Hyperopt search space is empty or invalid.")
        return space

    def _objective_for_hyperopt(self, current_hyperparams):
        self.trial_count += 1

        trial_nsga3_config = copy.deepcopy(self.base_config['nsga3_optimizer'])
        bo_eval_ea_config = self.bo_config.get('base_nsga3_config_for_bo_eval', {})
        eval_generations_bo = bo_eval_ea_config.get('num_generations', 8)
        trial_nsga3_config['num_generations'] = int(eval_generations_bo)

        for key, value in current_hyperparams.items():
            param_detail = self.search_space_config_raw.get(key)
            if param_detail and param_detail['type'] == "quniform" and param_detail.get('q') == 1:
                trial_nsga3_config[key] = int(value)
            else:
                trial_nsga3_config[key] = value
        
        source_train_eval = self.source_train
        target_train_eval = self.target_train
        subset_ratio_bo = self.bo_config.get('training_data_subset_ratio_bo', 1.0)
        subset_info_str = ""

        if subset_ratio_bo < 1.0 and len(self.source_train) > 1:
            num_train_samples = len(self.source_train)
            subset_size = max(1, int(num_train_samples * subset_ratio_bo))
            indices = np.arange(subset_size)
            source_train_eval = [self.source_train[i] for i in indices]
            target_train_eval = [self.target_train[i] for i in indices]
            if subset_size < num_train_samples:
                 subset_info_str = f"(Subset: {subset_size}/{num_train_samples})"

        pr_config = self.base_config['pose_retargeting']
        genome_bounds_cfg = self.base_config.get('genome_definition')

        problem = PoseRetargetingProblem(
            source_sequences_train=source_train_eval,
            target_sequences_train=target_train_eval,
            source_dim_config=(pr_config['source_num_keypoints'], 3),
            target_dim_config=(pr_config['target_num_keypoints'], 3),
            genome_param_bounds=genome_bounds_cfg
        )
        
        runner_seed = self.base_config['settings'].get('global_random_seed')
        if runner_seed is not None: runner_seed = int(runner_seed)
            
        runner = EvolutionaryRunner(problem, trial_nsga3_config, runner_seed, verbose_level=0)

        metric_value = 1e9 
        duration_ea_run = 0
        ea_run_details_str = f"(N_gen={trial_nsga3_config['num_generations']}, Pop={trial_nsga3_config['population_size']})"
        
        try:
            start_time_ea = time.time()
            results = runner.run()
            duration_ea_run = time.time() - start_time_ea

            if results is not None and hasattr(results, 'F') and results.F is not None and results.F.shape[0] > 0:
                pareto_front_F = results.F
                min_mpjpe = np.min(pareto_front_F[:, 0])
                min_temp_consist = np.min(pareto_front_F[:, 1])
                weights_cfg = self.base_config['nsga3_optimizer'].get('objective_weights', {'accuracy': 1.0, 'temporal_consistency': 1.0})
                metric_value = (weights_cfg.get('accuracy', 1.0) * min_mpjpe + 
                                weights_cfg.get('temporal_consistency', 1.0) * min_temp_consist)
            
            self.log_func(f"[BO Eval #{self.trial_count}] Loss: {metric_value:.6f}, EA: {duration_ea_run:.2f}s {ea_run_details_str} {subset_info_str}. Params: {current_hyperparams}")
            return {'loss': metric_value, 'status': STATUS_OK, 'params': current_hyperparams, 'duration': duration_ea_run}

        except Exception as e:
            self.log_func(f"[BO Eval #{self.trial_count} Error] EA: {duration_ea_run:.2f}s {ea_run_details_str} {subset_info_str}. Params: {current_hyperparams}. Exception: {e}. Penalizing.")
            return {'loss': 1e10, 'status': STATUS_OK, 'duration': duration_ea_run, 'params': current_hyperparams}


    def _log_insights_no_plots(self, trials):
        """Logs top trials and saves trial data to CSV if pandas is available."""
        self.log_func("\n[BO Insights] Top 5 Trials (lower loss is better):")
        valid_results = [res for res in trials.results if 'loss' in res]
        if not valid_results:
            self.log_func("  No valid trial results to process.")
            return

        sorted_trials_results = sorted(valid_results, key=lambda x: x['loss'])
        for i, trial_res in enumerate(sorted_trials_results[:5]):
            self.log_func(f"  Rank {i+1}: Loss={trial_res['loss']:.4f}, Params={trial_res.get('params', 'N/A')}, Duration={trial_res.get('duration',0):.2f}s")

        try:
            trial_data_list = []
            for t in trials.trials:
                if 'result' in t and 'params' in t['result']:
                     trial_data_list.append({'loss': t['result']['loss'], 
                                             'duration': t['result'].get('duration', 0), 
                                             **t['result']['params']})
            if not trial_data_list:
                self.log_func("[BO Insights] No trial data with parameters found to create CSV.")
                return

            df_trials = pd.DataFrame(trial_data_list)
            csv_path = os.path.join(self.reports_dir, "bo_hyperopt_trials.csv")
            df_trials.to_csv(csv_path, index=False)
            self.log_func(f"[BO Insights] All trial data saved to {csv_path}")
        except Exception as e_csv: # Catch specific pandas/CSV errors if any
            self.log_func(f"[BO Insights] Error during CSV saving: {e_csv}")


    def optimize(self):
        if not self.hyperopt_space:
             self.log_func("[BO Error] Cannot start: Hyperopt search space is not set.")
             return {}, np.inf
        self.trial_count = 0

        self.log_func(f"[BO Hyperopt] Starting TPE with {self.bo_config['max_evals_bo']} evaluations...")
        self.log_func(f"[BO Hyperopt] Search space (raw config): {self.search_space_config_raw}")
        
        trials = Trials()
        bo_random_seed = self.base_config['settings'].get('global_random_seed')
        if bo_random_seed is not None: bo_random_seed = int(bo_random_seed)

        algo_rstate = np.random.default_rng(seed=bo_random_seed) if bo_random_seed is not None else None

        start_time_bo = time.time()
        fmin( 
            fn=self._objective_for_hyperopt,
            space=self.hyperopt_space,
            algo=tpe.suggest,
            max_evals=self.bo_config['max_evals_bo'],
            trials=trials,
            rstate=algo_rstate,
            show_progressbar=True
        )
        duration_bo = time.time() - start_time_bo
        self.log_func(f"\n[BO Hyperopt] TPE completed in {duration_bo:.2f}s.")

        self._log_insights_no_plots(trials) # Call the version without plotting

        best_hyperparams_dict = {}
        best_loss_val = np.inf
        
        if trials.best_trial and 'result' in trials.best_trial and 'params' in trials.best_trial['result']:
             best_hyperparams_dict = trials.best_trial['result']['params']
             best_loss_val = trials.best_trial['result']['loss']
        else:
            self.log_func("[BO Warning] Could not reliably retrieve best trial details from Hyperopt Trials.")

        for name, conf_detail in self.search_space_config_raw.items():
            if name in best_hyperparams_dict:
                if conf_detail['type'] == "quniform" and conf_detail.get('q') == 1:
                    best_hyperparams_dict[name] = int(best_hyperparams_dict[name])
        
        self.log_func(f"[BO Hyperopt] Best hyperparameters found: {best_hyperparams_dict}")
        
        # Corrected f-string for logging best_loss_val
        if best_loss_val != np.inf:
            self.log_func(f"[BO Hyperopt] Best objective value (loss): {best_loss_val:.6f}")
        else:
            self.log_func("[BO Hyperopt] Best objective value (loss): N/A (no successful trials or all failed)")
        
        return best_hyperparams_dict, best_loss_val