import time
import yaml
import os
import warnings
import numpy as np
from rich.console import Console
from rich.panel import Panel
import matplotlib.pyplot as plt
import pickle
import re

from pose_retargeting.pose_data_loader import load_pose_data
from pose_retargeting.retargeting_problem import PoseRetargetingProblem
from pose_retargeting.evolutionary_runner import EvolutionaryRunner
from pose_retargeting.genome_and_transform import Genome


console = Console()

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(REPORTS_DIR, "evolution_log.txt")
HISTORY_FILE_PATH = os.path.join(REPORTS_DIR, "evolution_history.pkl")
PARETO_PLOT_PATH = os.path.join(REPORTS_DIR, "pareto_front_plot.png")
EVOLUTION_PLOT_PATH = os.path.join(REPORTS_DIR, "evolution_progress_plot.png") 


def log_message(message_obj, to_console=True, to_file=True):
    """
    Logs a message to the console (as Rich object) and to a plain text file.
    """
    if to_console:
        console.print(message_obj)

    if to_file:
        plain_text_message = ""
        if isinstance(message_obj, str):
            plain_text_message = re.sub(r'\[(/?[a-zA-Z_]\w*?.*?|/?\s*?)\]', '', message_obj)
            replacements = {'✓': 'OK', '📂': 'Folder:', '🧬': 'Genome:', '🚀': 'Run:', '📊': 'Results:'}
            for old, new in replacements.items():
                plain_text_message = plain_text_message.replace(old, new)
        elif hasattr(message_obj, '__rich_console__'): 
            capture_console = Console(file=open(os.devnull, "w"), width=120, force_terminal=False, legacy_windows=False)
            with capture_console.capture() as capture:
                capture_console.print(message_obj)
            plain_text_message = capture.get()
            plain_text_message = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', plain_text_message) # Strip ANSI
            plain_text_message = plain_text_message.strip()
        else: 
            plain_text_message = str(message_obj)

        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(plain_text_message + "\n")
        except Exception as e:
            console.print(f"[bold red]Error writing to log file:[/bold red] {e}")


def main():
    if os.path.exists(LOG_FILE_PATH):
        try:
            os.remove(LOG_FILE_PATH)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove old log file: {e}[/yellow]")

    log_message(Panel.fit("[bold blue]Starting Pose Retargeting Pipeline with NSGA-III...[/bold blue]", border_style="blue"))
    
    config_path = 'config.yaml'
    config = None
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            config = yaml.safe_load(f)
        log_message(f"[green]✓ Configuration loaded from [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        log_message(f"[bold red]ERROR:[/bold red] Configuration file '[bold]{config_path}[/bold]' not found.")
        exit(1)
    except yaml.YAMLError as e:
        log_message(f"[bold red]ERROR:[/bold red] YAML parsing error in '[bold]{config_path}[/bold]': {e}")
        exit(1)
    except Exception as e:
        log_message(f"[bold red]ERROR:[/bold red] Unknown error loading '[bold]{config_path}[/bold]': {e}")
        exit(1)

    if config is None:
        log_message("[bold red]CRITICAL: Configuration not loaded. Aborting.[/bold red]")
        exit(1)

    verbose = config.get('settings', {}).get('verbose', 0)
    
    log_message(Panel("[yellow]📂 Loading and Preprocessing Pose Data...[/yellow]", border_style="yellow"))
    pr_config = config.get('pose_retargeting')
    if not pr_config:
        log_message("[bold red]ERROR: 'pose_retargeting' section missing in config.yaml[/bold red]")
        exit(1)

    source_train, target_train, source_test, target_test = load_pose_data(config, verbose)

    if not source_train or not target_train: # Check if training data is loaded
        log_message("[bold red]ERROR: Training data (source_train or target_train) is empty. Aborting.[/bold red]")
        exit(1)

    log_message(f"[green]✓ Data loaded: {len(source_train)} training sequences, {len(source_test)} testing sequences.[/green]")
    if source_train: 
         log_message(f"  Example train source sequence shape: {source_train[0].shape}")
         log_message(f"  Example train target sequence shape: {target_train[0].shape}")

    log_message(Panel("[magenta]🧬 Defining Optimization Problem...[/magenta]", border_style="magenta"))
    source_dim_cfg = (pr_config['source_num_keypoints'], 3)
    target_dim_cfg = (pr_config['target_num_keypoints'], 3)
    
    genome_bounds_cfg = config.get('genome_definition')
    if not genome_bounds_cfg:
        log_message("[bold red]ERROR: 'genome_definition' section missing in config.yaml[/bold red]")
        exit(1)

    problem = PoseRetargetingProblem(
        source_sequences_train=source_train,
        target_sequences_train=target_train,
        source_dim_config=source_dim_cfg,
        target_dim_config=target_dim_cfg,
        genome_param_bounds=genome_bounds_cfg
    )
    log_message("[green]✓ Optimization problem defined.[/green]")

    log_message(Panel("[cyan]🚀 Running NSGA-III Optimization...[/cyan]", border_style="cyan"))
    nsga3_cfg = config.get('nsga3_optimizer')
    if not nsga3_cfg:
        log_message("[bold red]ERROR: 'nsga3_optimizer' section missing in config.yaml[/bold red]")
        exit(1)

    # return 

    runner = EvolutionaryRunner(problem, nsga3_cfg, config['settings'].get('global_random_seed'))

    start_time = time.time()
    results = runner.run()
    end_time = time.time()
    log_message(f"Optimization duration: {(end_time - start_time):.2f} seconds.")

    log_message(Panel("[green]📊 Analyzing Optimization Results...[/green]", border_style="green"))

    if results is not None and hasattr(results, 'F') and results.F is not None and results.F.shape[0] > 0 :
        log_message(f"Found [bold]{len(results.X)}[/bold] non-dominated solutions (Pareto front).")

        log_message("\nObjective values (F) of non-dominated solutions:")
        log_message("---------------------------------------------------")
        log_message("Idx | MPJPE (f1)   | Temp. Consist. (f2)")
        log_message("----|--------------|--------------------")
        for i, (f1, f2) in enumerate(results.F):
            log_message(f"{i:<3} | {f1:<12.6f} | {f2:<18.6f}")
        log_message("---------------------------------------------------\n")

        try:
            with open(HISTORY_FILE_PATH, 'wb') as f_hist:
                pickle.dump(results, f_hist)
            log_message(f"Full evolution history saved to: [cyan]{HISTORY_FILE_PATH}[/cyan]")
        except Exception as e:
            log_message(f"[yellow]Warning: Could not save evolution history: {e}[/yellow]")

        # --- Enhanced Pareto Front Plot ---
        plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics
        plt.figure(figsize=(10, 7))
        
        # Scatter plot of all non-dominated solutions
        plt.scatter(results.F[:, 0], results.F[:, 1], s=70,
                      facecolors='royalblue', edgecolors='black', alpha=0.8, label="Pareto Optimal Solutions")

        # Sort solutions by the first objective to draw the front line
        sorted_indices = np.argsort(results.F[:, 0])
        F_sorted = results.F[sorted_indices]
        
        if F_sorted.shape[0] > 1: # Only draw line if more than one point
            plt.plot(F_sorted[:, 0], F_sorted[:, 1], color='orangered', linestyle='--', linewidth=1.5, label="Pareto Front")

        # Annotate some specific solutions (indices are from the original results.F)
        for i, (f1, f2) in enumerate(results.F):
            plt.annotate(f"{i}", (f1, f2), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)

        best_mpjpe_idx = np.argmin(results.F[:, 0])
        plt.scatter(results.F[best_mpjpe_idx, 0], results.F[best_mpjpe_idx, 1],
                    s=150, facecolors='none', edgecolors='crimson', linewidth=2,
                    label=f"Lowest MPJPE (Sol: {best_mpjpe_idx})")

        if len(results.F) > 2: 
            # Find a compromise solution (e.g., median f1 after sorting, ensuring it's distinct)
            # This example compromise logic might need to be smarter depending on front shape
            # For simplicity, pick one near the "knee" or middle if sorted by f1.
            sorted_indices_f1_original = np.argsort(results.F[:, 0])
            # Try to pick a compromise solution that is not the best MPJPE one.
            # This could be the median of sorted by f1, or median of sorted by f2, or other heuristic.
            compromise_idx_candidate_orig_idx = sorted_indices_f1_original[len(sorted_indices_f1_original) // 2]
            
            if compromise_idx_candidate_orig_idx != best_mpjpe_idx :
                 plt.scatter(results.F[compromise_idx_candidate_orig_idx, 0], results.F[compromise_idx_candidate_orig_idx, 1],
                            s=150, facecolors='none', edgecolors='forestgreen', linewidth=2,
                            label=f"Example Compromise (Sol: {compromise_idx_candidate_orig_idx})")

        plt.title('Pareto Front (NSGA-III)', fontsize=16, fontweight='bold')
        plt.xlabel('f1: MPJPE (Lower is Better)', fontsize=12)
        plt.ylabel('f2: Temporal Consistency Error (Lower is Better)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()

        try:
            plt.savefig(PARETO_PLOT_PATH)
            log_message(f"Pareto front plot saved to: [cyan]{PARETO_PLOT_PATH}[/cyan]")
        except Exception as e:
            log_message(f"[yellow]Warning: Could not save Pareto plot: {e}[/yellow]")
        # plt.show() # Usually commented out for automated runs

        # --- Evolution Plot ---
        if hasattr(results, 'history') and results.history is not None and len(results.history) > 0:
            num_generations = len(results.history)
            generations = np.arange(1, num_generations + 1)

            best_f1_per_gen = np.full(num_generations, np.inf)
            avg_f1_per_gen = np.full(num_generations, np.inf)
            best_f2_per_gen = np.full(num_generations, np.inf)
            avg_f2_per_gen = np.full(num_generations, np.inf)

            for i, algorithm_gen in enumerate(results.history):
                if algorithm_gen.opt is not None and algorithm_gen.opt.has("F"): # opt stores non-dominated solutions
                    opt_F = algorithm_gen.opt.get("F")
                    if opt_F.ndim == 2 and opt_F.shape[0] > 0 and opt_F.shape[1] >= 2:
                        best_f1_per_gen[i] = np.min(opt_F[:, 0])
                        avg_f1_per_gen[i] = np.mean(opt_F[:, 0])
                        best_f2_per_gen[i] = np.min(opt_F[:, 1])
                        avg_f2_per_gen[i] = np.mean(opt_F[:, 1])
                    elif opt_F.ndim == 1 and opt_F.shape[0] >=2 : # single solution in opt
                        best_f1_per_gen[i] = opt_F[0]
                        avg_f1_per_gen[i] = opt_F[0]
                        best_f2_per_gen[i] = opt_F[1]
                        avg_f2_per_gen[i] = opt_F[1]


            plt.figure(figsize=(12, 8))
            plt.suptitle('Evolution of Objectives Over Generations', fontsize=16, fontweight='bold')

            # Plot for f1 (MPJPE)
            plt.subplot(2, 1, 1)
            plt.plot(generations, best_f1_per_gen, marker='o', linestyle='-', color='dodgerblue', markersize=5, label='Best MPJPE (f1)')
            plt.plot(generations, avg_f1_per_gen, marker='x', linestyle='--', color='skyblue', markersize=5, label='Average MPJPE (f1) in Pareto Set')
            plt.title('MPJPE (f1) Evolution', fontsize=14)
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel('MPJPE Value', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.ylim(bottom=0) # MPJPE should not be negative

            # Plot for f2 (Temporal Consistency)
            plt.subplot(2, 1, 2)
            plt.plot(generations, best_f2_per_gen, marker='o', linestyle='-', color='forestgreen', markersize=5, label='Best Temp. Consistency (f2)')
            plt.plot(generations, avg_f2_per_gen, marker='x', linestyle='--', color='lightgreen', markersize=5, label='Average Temp. Consistency (f2) in Pareto Set')
            plt.title('Temporal Consistency (f2) Evolution', fontsize=14)
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel('Temporal Consistency Error', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.ylim(bottom=0) # Error should not be negative


            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            
            try:
                plt.savefig(EVOLUTION_PLOT_PATH)
                log_message(f"Evolution progress plot saved to: [cyan]{EVOLUTION_PLOT_PATH}[/cyan]")
            except Exception as e:
                log_message(f"[yellow]Warning: Could not save evolution plot: {e}[/yellow]")
            # plt.show()
        else:
            log_message("[yellow]Evolution history not available or empty, skipping evolution plot.[/yellow]")


        selected_solution_X = results.X[best_mpjpe_idx]
        selected_solution_F = results.F[best_mpjpe_idx]
        log_message(f"\nExample solution (lowest MPJPE - Index {best_mpjpe_idx}):")
        log_message(f"  Objectives: MPJPE={selected_solution_F[0]:.6f}, TemporalConsist={selected_solution_F[1]:.6f}")

        genome = Genome.from_flat_representation(selected_solution_X, source_dim_cfg, target_dim_cfg)
        log_message(f"  Decoded C1 matrix (sample of Sol {best_mpjpe_idx}): \n{genome.C1[:2,:5]}")

    else:
        log_message("[yellow]No valid solutions found or optimization did not produce results.F attribute.[/yellow]")

    log_message(Panel("[bold blue]Pose Retargeting Pipeline completed.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning) 

    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERROR:[/bold red] Configuration file 'config.yaml' does not exist.", style="red")
        exit(1)

    main()