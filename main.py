import time
import yaml
import os
import warnings
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import matplotlib.pyplot as plt
import pickle
import re

from pose_retargeting.pose_data_loader import load_pose_data
from pose_retargeting.retargeting_problem import PoseRetargetingProblem
from pose_retargeting.evolutionary_runner import EvolutionaryRunner
from pose_retargeting.genome_and_transform import Genome
from bayesian_evo_opt.optimizer import BayesianHyperparameterOptimizer


console = Console()

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(REPORTS_DIR, "evolution_log.txt")
HISTORY_FILE_PATH = os.path.join(REPORTS_DIR, "evolution_history.pkl")
PARETO_PLOT_PATH = os.path.join(REPORTS_DIR, "pareto_front_plot.png")
EVOLUTION_PLOT_PATH = os.path.join(REPORTS_DIR, "evolution_progress_plot.png") 
METRICS_CSV_PATH = os.path.join(REPORTS_DIR, "final_metrics.csv")


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


def display_metrics_table(results, problem=None):
    """Create and display a comprehensive metrics table."""
    if results is None or not hasattr(results, 'F') or results.F is None:
        return
    
    # If we have access to the problem, we can unscale the objectives
    display_values = results.F.copy()
    if problem is not None and problem.objective_scales is not None:
        # Unscale objectives for display
        display_values = results.F * problem.objective_scales
        log_message("[cyan]Note: Displaying unscaled objective values (actual meters/values)[/cyan]")
    else:
        log_message("[yellow]Warning: Displaying scaled objective values (used for optimization)[/yellow]")
    
    # Define objective names
    objective_names = [
        "MPJPE (m)",
        "MPJPE (norm)",
        "Trimmed MPJPE (m)",
        "Median PJPE (m)",
        "Temporal Consist.",
        "90th %ile (m)"
    ]
    
    # Create table
    table = Table(title="[bold]Pareto Front Solutions - All Metrics[/bold]", 
                  show_header=True, header_style="bold magenta")
    
    table.add_column("Solution", style="cyan", justify="center")
    for obj_name in objective_names:
        table.add_column(obj_name, justify="right")
    
    # Add rows for each solution
    for i, objectives in enumerate(display_values):
        row_data = [f"#{i}"]
        for j, value in enumerate(objectives):
            # Format based on metric type
            if j == 1:  # Normalized MPJPE (dimensionless)
                row_data.append(f"{value:.4f}")
            elif j == 4:  # Temporal consistency (very small values)
                row_data.append(f"{value:.6f}")
            else:  # All others in meters
                row_data.append(f"{value:.4f}")
        table.add_row(*row_data)
    
    # Add summary statistics
    table.add_row("", "", "", "", "", "", "")  # Empty row
    best_row = ["[bold]Best[/bold]"]
    mean_row = ["[bold]Mean[/bold]"]
    
    for i in range(display_values.shape[1]):
        best_val = np.min(display_values[:, i])
        mean_val = np.mean(display_values[:, i])
        
        if i == 4:  # Temporal consistency
            best_row.append(f"{best_val:.6f}")
            mean_row.append(f"{mean_val:.6f}")
        else:
            best_row.append(f"{best_val:.4f}")
            mean_row.append(f"{mean_val:.4f}")
    
    table.add_row(*best_row, style="green")
    table.add_row(*mean_row, style="yellow")
    
    log_message(table)
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(display_values, columns=objective_names)
    df.index.name = 'Solution'
    df.to_csv(METRICS_CSV_PATH)
    log_message(f"[green]Metrics saved to: {METRICS_CSV_PATH}[/green]")


def plot_multi_objective_pareto(results, problem=None):
    """Create visualization for multi-objective optimization results."""
    if results is None or not hasattr(results, 'F') or results.F is None:
        return
    
    # Unscale objectives if possible
    display_values = results.F.copy()
    if problem is not None and problem.objective_scales is not None:
        display_values = results.F * problem.objective_scales
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Objective Optimization Results (Unscaled Values)', fontsize=16, fontweight='bold')
    
    # Plot 1: MPJPE (meters) vs Normalized MPJPE
    ax = axes[0, 0]
    ax.scatter(display_values[:, 0], display_values[:, 1], s=50, alpha=0.6, c='blue')
    ax.set_xlabel('MPJPE in meters')
    ax.set_ylabel('MPJPE normalized')
    ax.set_title('Standard vs Normalized MPJPE')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MPJPE vs Trimmed MPJPE
    ax = axes[0, 1]
    ax.scatter(display_values[:, 0], display_values[:, 2], s=50, alpha=0.6, c='green')
    min_val = min(display_values[:, 0].min(), display_values[:, 2].min())
    max_val = max(display_values[:, 0].max(), display_values[:, 2].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('MPJPE in meters')
    ax.set_ylabel('Trimmed MPJPE in meters')
    ax.set_title('Effect of Trimming (10% outliers removed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: MPJPE vs Median PJPE
    ax = axes[1, 0]
    ax.scatter(display_values[:, 0], display_values[:, 3], s=50, alpha=0.6, c='red')
    min_val = min(display_values[:, 0].min(), display_values[:, 3].min())
    max_val = max(display_values[:, 0].max(), display_values[:, 3].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('MPJPE in meters')
    ax.set_ylabel('Median PJPE in meters')
    ax.set_title('Mean vs Median (Robustness Check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: MPJPE vs Temporal Consistency
    ax = axes[1, 1]
    scatter = ax.scatter(display_values[:, 0], display_values[:, 4], 
                        c=display_values[:, 5], s=50, alpha=0.6, cmap='viridis')
    ax.set_xlabel('MPJPE in meters')
    ax.set_ylabel('Temporal Consistency')
    ax.set_title('Accuracy vs Smoothness (color: 90th percentile)')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('90th %ile error (m)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    pareto_path = os.path.join(REPORTS_DIR, "multi_objective_results.png")
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    log_message(f"[green]Multi-objective plots saved to: {pareto_path}[/green]")
    

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

    if not source_train or not target_train:
        log_message("[bold red]ERROR: Training data (source_train or target_train) is empty. Aborting.[/bold red]")
        exit(1)

    log_message(f"[green]✓ Data loaded: {len(source_train)} training sequences, {len(source_test)} testing sequences.[/green]")
    if source_train: 
         log_message(f"  Example train source sequence shape: {source_train[0].shape}")
         log_message(f"  Example train target sequence shape: {target_train[0].shape}")
         # Check normalization status
         norm_count = sum(1 for seq in source_train if seq.is_normalized)
         log_message(f"  Normalized sequences: {norm_count}/{len(source_train)}")

    
    # --- Bayesian Optimization for Hyperparameters ---
    bo_settings = config.get('bayesian_optimizer_settings')
    if bo_settings and bo_settings.get('run_bayesian_opt'):
        log_message(Panel("[purple]⚙️ Running Bayesian Optimization (Hyperopt TPE) for NSGA-III Hyperparameters...[/purple]", border_style="purple"))
        
        if 'nsga3_optimizer' not in config:
            log_message("[bold red]ERROR: 'nsga3_optimizer' section missing in config.yaml, needed for Bayesian Optimization base. Aborting.[/bold red]")
            exit(1)

        hyper_optimizer = BayesianHyperparameterOptimizer(
            base_config=config, 
            source_train=source_train, 
            target_train=target_train, 
            log_func=log_message 
        )
        try:
            best_hyperparams, best_metric = hyper_optimizer.optimize()

            log_message(Panel(f"[purple]Bayesian Optimization (Hyperopt TPE) Completed.[/purple]\n"
                              f"  Best Hyperparameters: {best_hyperparams}\n"
                              f"  Best Objective Metric (loss): {best_metric:.6f}",
                              border_style="purple"))
            
            log_message("[purple]⚙️ Updating 'nsga3_optimizer' config with found hyperparameters for the main run.[/purple]")
            for key, value in best_hyperparams.items():
                config['nsga3_optimizer'][key] = value
            
        except Exception as e_bo:
            log_message(f"[bold red]ERROR during Bayesian Optimization (Hyperopt): {e_bo}. Proceeding with original hyperparameters.[/bold red]")
            import traceback 
            log_message(traceback.format_exc())
    
    log_message(Panel("[magenta]🧬 Defining Optimization Problem...[/magenta]", border_style="magenta"))
    source_dim_cfg = (pr_config['source_num_keypoints'], 3)
    target_dim_cfg = (pr_config['target_num_keypoints'], 3)
    
    genome_bounds_cfg = config.get('genome_definition')
    if not genome_bounds_cfg:
        log_message("[bold red]ERROR: 'genome_definition' section missing in config.yaml[/bold red]")
        exit(1)

    # Get trim percentage from config or use default
    trim_percentage = config.get('nsga3_optimizer', {}).get('trim_percentage', 0.1)
    
    problem = PoseRetargetingProblem(
        source_sequences_train=source_train,
        target_sequences_train=target_train,
        source_dim_config=source_dim_cfg,
        target_dim_config=target_dim_cfg,
        genome_param_bounds=genome_bounds_cfg,
        trim_percentage=trim_percentage
    )
    log_message("[green]✓ Optimization problem defined with 6 objectives (including robust metrics).[/green]")

    log_message(Panel("[cyan]🚀 Running NSGA-III Optimization...[/cyan]", border_style="cyan"))
    nsga3_cfg = config.get('nsga3_optimizer') 
    if not nsga3_cfg:
        log_message("[bold red]ERROR: 'nsga3_optimizer' section missing in config.yaml[/bold red]")
        exit(1)

    main_ea_verbose_level = config.get('settings', {}).get('verbose_ea_runner', 1) 
    runner = EvolutionaryRunner(problem, nsga3_cfg, config['settings'].get('global_random_seed'), verbose_level=main_ea_verbose_level)

    start_time = time.time()
    results = runner.run()
    end_time = time.time()
    log_message(f"Optimization duration: {(end_time - start_time):.2f} seconds.")

    log_message(Panel("[green]📊 Analyzing Optimization Results...[/green]", border_style="green"))

    if results is not None and hasattr(results, 'F') and results.F is not None and results.F.shape[0] > 0:
        log_message(f"Found [bold]{len(results.X)}[/bold] non-dominated solutions (Pareto front).")

        # Display comprehensive metrics table with unscaled values
        display_metrics_table(results, problem)
        
        # Get unscaled values for analysis
        if problem.objective_scales is not None:
            unscaled_F = results.F * problem.objective_scales
        else:
            unscaled_F = results.F
        
        # Find best solutions according to different criteria (using unscaled values)
        best_mpjpe_idx = np.argmin(unscaled_F[:, 0])
        best_trimmed_idx = np.argmin(unscaled_F[:, 2])
        best_median_idx = np.argmin(unscaled_F[:, 3])
        
        log_message("\n[bold]Best Solutions by Different Criteria:[/bold]")
        log_message(f"  Best MPJPE (meters): Solution #{best_mpjpe_idx} = {unscaled_F[best_mpjpe_idx, 0]:.4f}m")
        log_message(f"  Best Trimmed MPJPE: Solution #{best_trimmed_idx} = {unscaled_F[best_trimmed_idx, 2]:.4f}m")
        log_message(f"  Best Median PJPE: Solution #{best_median_idx} = {unscaled_F[best_median_idx, 3]:.4f}m")
        
        # Check robustness
        log_message("\n[bold]Robustness Analysis:[/bold]")
        for i in [best_mpjpe_idx, best_trimmed_idx, best_median_idx]:
            mpjpe = unscaled_F[i, 0]
            trimmed = unscaled_F[i, 2]
            median = unscaled_F[i, 3]
            p90 = unscaled_F[i, 5]
            robustness_score = (trimmed - mpjpe) / mpjpe * 100  # % difference
            log_message(f"  Solution #{i}: MPJPE={mpjpe:.4f}m, Trimmed={trimmed:.4f}m "
                       f"(diff: {robustness_score:+.1f}%), 90th%ile={p90:.4f}m")

        try:
            # Save results with problem instance for later retrieval
            save_data = {
                'results': results,
                'problem_scales': problem.objective_scales,
                'problem_config': {
                    'source_dim': problem.source_dim,
                    'target_dim': problem.target_dim,
                    'trim_percentage': problem.trim_percentage
                }
            }
            # Use binary write mode with proper path handling
            history_path = os.path.normpath(HISTORY_FILE_PATH)
            with open(history_path, 'wb') as f_hist:
                pickle.dump(save_data, f_hist)
            log_message(f"Full evolution history saved to: [cyan]{history_path}[/cyan]")
        except Exception as e:
            log_message(f"[yellow]Warning: Could not save evolution history: {e}[/yellow]")
            # Try alternative approach for Windows
            if os.name == 'nt':
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir=REPORTS_DIR) as tmp:
                        pickle.dump(save_data, tmp)
                        log_message(f"[yellow]Saved to temporary file: {tmp.name}[/yellow]")
                except Exception as e2:
                    log_message(f"[red]Failed to save even to temp file: {e2}[/red]")

        # Create multi-objective visualization with unscaled values
        plot_multi_objective_pareto(results, problem)

        # Example: decode best solution
        selected_solution_X = results.X[best_trimmed_idx]  # Using best trimmed MPJPE
        selected_solution_F = unscaled_F[best_trimmed_idx]
        log_message(f"\n[bold]Selected Solution (Best Trimmed MPJPE - Index {best_trimmed_idx}):[/bold]")
        log_message(f"  MPJPE (meters): {selected_solution_F[0]:.6f}")
        log_message(f"  MPJPE (normalized): {selected_solution_F[1]:.6f}")
        log_message(f"  Trimmed MPJPE: {selected_solution_F[2]:.6f}")
        log_message(f"  Median PJPE: {selected_solution_F[3]:.6f}")
        log_message(f"  Temporal Consistency: {selected_solution_F[4]:.6f}")
        log_message(f"  90th Percentile: {selected_solution_F[5]:.6f}")

        genome = Genome.from_flat_representation(selected_solution_X, source_dim_cfg, target_dim_cfg)
        log_message(f"  Decoded C1 matrix (sample): \n{genome.C1[:2,:5]}")

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