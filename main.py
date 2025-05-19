import time
import yaml
import os
import warnings
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint 
import matplotlib.pyplot as plt


# Import new modules for pose retargeting
from pose_retargeting import load_pose_data
from pose_retargeting import PoseRetargetingProblem
from pose_retargeting import EvolutionaryRunner
from pose_retargeting import Genome 

# Create a console instance
console = Console()

def main_pose_retargeting():
    console.print(Panel.fit("[bold blue]Starting Pose Retargeting Pipeline with NSGA-III...[/bold blue]", 
                           border_style="blue"))

    # Load configuration from YAML file
    config_path = 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        console.print(f"[green]✓ Configuration loaded from [bold]{config_path}[/bold][/green]")
    except FileNotFoundError:
        console.print(f"[bold red]ERROR:[/bold red] Configuration file '[bold]{config_path}[/bold]' not found.", style="red")
        exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]ERROR:[/bold red] YAML parsing error in '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Unknown error loading '[bold]{config_path}[/bold]': {e}", style="red")
        exit(1)

    # --- 1. Load and Preprocess Pose Data ---
    console.print(Panel("[yellow]📂 Loading and Preprocessing Pose Data...[/yellow]", border_style="yellow"))
    pr_config = config.get('pose_retargeting')
    if not pr_config:
        console.print("[bold red]ERROR: 'pose_retargeting' section missing in config.yaml[/bold red]")
        exit(1)

    source_train, target_train, source_test, target_test = load_pose_data(config)

    if not source_train or not target_train:
        console.print("[bold red]ERROR: Training data could not be loaded. Aborting.[/bold red]")
        exit(1)
    
    console.print(f"[green]✓ Data loaded: {len(source_train)} training sequences, {len(source_test)} testing sequences.[/green]")
    if source_train:
         console.print(f"  Example train source sequence shape: {source_train[0].shape}")
         console.print(f"  Example train target sequence shape: {target_train[0].shape}")


    # --- 2. Define the Optimization Problem ---
    console.print(Panel("[magenta]🧬 Defining Optimization Problem...[/magenta]", border_style="magenta"))
    source_dim_cfg = (pr_config['source_num_keypoints'], 3)
    target_dim_cfg = (pr_config['target_num_keypoints'], 3)
    
    genome_bounds_cfg = config.get('genome_definition')
    if not genome_bounds_cfg:
        console.print("[bold red]ERROR: 'genome_definition' section missing in config.yaml[/bold red]")
        exit(1)

    problem = PoseRetargetingProblem(
        source_sequences_train=source_train,
        target_sequences_train=target_train,
        source_dim_config=source_dim_cfg,
        target_dim_config=target_dim_cfg,
        genome_param_bounds=genome_bounds_cfg
    )
    console.print("[green]✓ Optimization problem defined.[/green]")

    # --- 3. Run Evolutionary Algorithm (NSGA-III) ---
    console.print(Panel("[cyan]🚀 Running NSGA-III Optimization...[/cyan]", border_style="cyan"))
    nsga3_cfg = config.get('nsga3_optimizer')
    if not nsga3_cfg:
        console.print("[bold red]ERROR: 'nsga3_optimizer' section missing in config.yaml[/bold red]")
        exit(1)
    
    runner = EvolutionaryRunner(problem, nsga3_cfg, config['settings'].get('global_random_seed'))
    
    start_time = time.time()
    results = runner.run()
    end_time = time.time()
    console.print(f"Optimization duration: {(end_time - start_time):.2f} seconds.")

    # --- 4. Analyze Results ---
    console.print(Panel("[green]📊 Analyzing Optimization Results...[/green]", border_style="green"))
    if results.X is not None and results.F is not None:
        console.print(f"Found [bold]{len(results.X)}[/bold] non-dominated solutions (Pareto front).")
        console.print("Objective values (F) of non-dominated solutions (first 5):")
        rprint(results.F[:5]) # F contains [MPJPE, TemporalConsistencyError]

        # Example: Plot Pareto Front
        plt.figure(figsize=(8, 6))
        plt.scatter(results.F[:, 0], results.F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.title('Pareto Front (NSGA-III)')
        plt.xlabel('f1: MPJPE (Lower is Better)')
        plt.ylabel('f2: Temporal Consistency Error (Lower is Better)')
        plt.grid(True)
        
        # Save plot
        output_dir = "reports"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "pareto_front_plot.png")
        plt.savefig(plot_path)
        console.print(f"Pareto front plot saved to: [cyan]{plot_path}[/cyan]")
        # plt.show() # Uncomment to display plot interactively

        # Here you would typically select one or more solutions from the Pareto front
        # for further evaluation on the test set.
        # For simplicity, let's pick the one with the lowest MPJPE for a quick check (not ideal for multi-objective)
        best_mpjpe_idx = np.argmin(results.F[:, 0])
        selected_solution_X = results.X[best_mpjpe_idx]
        selected_solution_F = results.F[best_mpjpe_idx]
        console.print(f"\nExample solution (lowest MPJPE): Objectives = {selected_solution_F}")

        # Decode the selected solution to see C1, S, B (optional)
        # selected_genome = Genome.from_flat_representation(selected_solution_X, source_dim_cfg, target_dim_cfg)
        # console.print(f"  Decoded C1 matrix (sample): \n{selected_genome.C1[:2,:5]}") # Print a small part

        # TODO: Implement evaluation of selected solution(s) on the test set (source_test, target_test)
        # This would involve using the `transform_source_to_target` function with the selected_genome
        # and calculating MPJPE and temporal consistency on the test data.

    else:
        console.print("[yellow]No solutions found or optimization did not run correctly.[/yellow]")

    console.print(Panel("[bold blue]Pose Retargeting Pipeline completed.[/bold blue]", border_style="blue"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning) # Pymoo might raise some user warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    
    if not os.path.exists('config.yaml'):
        console.print("[bold red]ERROR:[/bold red] Configuration file 'config.yaml' does not exist.", style="red")
        exit(1)
    
    main_pose_retargeting()