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
    def __init__(self, problem, config_nsga3, random_seed=None, verbose_level=1):
        self.problem = problem
        self.config = config_nsga3
        self.random_seed = random_seed
        self.verbose_level = verbose_level

    def run(self):
        if self.verbose_level > 0:
            console.print("\n[bold cyan]Initializing NSGA-III Algorithm...[/bold cyan]")
            console.print(f"Problem has {self.problem.n_obj} objectives")

        # For NSGA-III with many objectives, we need appropriate reference directions
        n_partitions_val = self.config.get('reference_point_partitions')
        if n_partitions_val is None:
            pop_size = self.config['population_size']
            # For 6 objectives, we need more sophisticated reference direction generation
            if self.problem.n_obj == 6:
                # Use a lower number of partitions for many objectives to avoid explosion
                n_partitions_val = 3  # This gives (3+6-1)!/(3!*(6-1)!) = 56 reference points
                if pop_size < 56:
                    n_partitions_val = 2  # This gives 28 reference points
            elif self.problem.n_obj == 2:
                n_partitions_val = pop_size - 1
            else:
                n_partitions_val = 12

        if self.problem.n_obj > 1:
            ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=n_partitions_val)
            if self.verbose_level > 0:
                console.print(f"Generated {len(ref_dirs)} reference directions for {self.problem.n_obj} objectives")
        else:
            ref_dirs = None

        sampling = LHS()
        
        # Use configured mutation_prob if available, otherwise None (for 1/n_var)
        mutation_probability = self.config.get('mutation_prob', None)
        if mutation_probability is not None and not (0 <= mutation_probability <= 1):
            if self.verbose_level > 0:
                console.print(f"[yellow]Warning: mutation_prob {mutation_probability} is outside [0,1]. Using pymoo default (1/n_var).[/yellow]")
            mutation_probability = None

        crossover = SBX(prob=self.config.get('crossover_prob', 0.9), 
                        eta=self.config.get('crossover_eta', 15))

        mutation = PM(prob=mutation_probability,
                      eta=self.config.get('mutation_eta', 20))

        # Ensure population size is at least as large as number of reference directions
        pop_size = self.config['population_size']
        if ref_dirs is not None and pop_size < len(ref_dirs):
            if self.verbose_level > 0:
                console.print(f"[yellow]Warning: Population size ({pop_size}) is smaller than number of reference directions ({len(ref_dirs)}). "
                            f"Adjusting population size to {len(ref_dirs)}.[/yellow]")
            pop_size = len(ref_dirs)

        algorithm = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", self.config['num_generations'])

        if self.verbose_level > 0:
            console.print(f"Starting optimization for [bold]{self.config['num_generations']} generations[/bold] "
                          f"with population size [bold]{pop_size}[/bold]...")
        
        results = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.random_seed,
            save_history=self.verbose_level > 0,
            verbose=self.verbose_level > 1
        )
        
        if self.verbose_level > 0:
            console.print("[bold green]Optimization finished.[/bold green]")
        
        return results