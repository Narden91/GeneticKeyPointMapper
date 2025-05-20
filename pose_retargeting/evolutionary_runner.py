from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.sampling.lhs import LHS # Latin Hypercube Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM # Polynomial Mutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from rich.console import Console

console = Console()

class EvolutionaryRunner:
    # Added verbose_level parameter with default 1
    def __init__(self, problem, config_nsga3, random_seed=None, verbose_level=1):
        self.problem = problem
        self.config = config_nsga3
        self.random_seed = random_seed
        self.verbose_level = verbose_level # Store verbose_level

    def run(self):
        if self.verbose_level > 0: # Conditional printing
            console.print("\n[bold cyan]Initializing NSGA-III Algorithm...[/bold cyan]")

        n_partitions_val = self.config.get('reference_point_partitions')
        if n_partitions_val is None: # If not explicitly set by BO or config
            pop_size = self.config['population_size']
            n_partitions_val = pop_size - 1 if pop_size > 1 and self.problem.n_obj == 2 else 12 
            # A common heuristic for 2 objectives is pop_size-1.
            # For >2 objectives, n_partitions relates to how many points are on the hyperplane.
            # pymoo's get_reference_directions handles this well.
            # A fixed default like 12 is often used if specific pop_size relation is not desired.

        if self.problem.n_obj > 1:
            ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=n_partitions_val)
        else:
             ref_dirs = None

        sampling = LHS()
        
        # Use configured mutation_prob if available, otherwise None (for 1/n_var)
        mutation_probability = self.config.get('mutation_prob', None)
        if mutation_probability is not None and not (0 <= mutation_probability <= 1):
            # If mutation_prob is optimized, it might come as a float.
            # If it was intended to be 1/n_var (by being None in config), EvolutionaryRunner shouldn't change that.
            # The BO will set a float value if 'mutation_prob' is in its search_space.
             if self.verbose_level > 0:
                console.print(f"[yellow]Warning: mutation_prob {mutation_probability} is outside [0,1]. Using pymoo default (1/n_var).[/yellow]")
             mutation_probability = None # Fallback to pymoo default for PM if invalid

        crossover = SBX(prob=self.config.get('crossover_prob', 0.9), 
                        eta=self.config.get('crossover_eta', 15))

        mutation = PM(prob=mutation_probability, # Use the potentially optimized value
                      eta=self.config.get('mutation_eta', 20))

        algorithm = NSGA3(
            pop_size=self.config['population_size'],
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.config['num_generations'])

        if self.verbose_level > 0: # Conditional printing
            console.print(f"Starting optimization for [bold]{self.config['num_generations']} generations[/bold] "
                          f"with population size [bold]{self.config['population_size']}[/bold]...")
        
        results = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.random_seed,
            save_history=self.verbose_level > 0, # Only save history for main runs, not BO evals to save memory
            verbose=self.verbose_level > 1 # pymoo's verbose output (per generation) if verbose_level is high
        )
        
        if self.verbose_level > 0: # Conditional printing
            console.print("[bold green]Optimization finished.[/bold green]")
        
        return results