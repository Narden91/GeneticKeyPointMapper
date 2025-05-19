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
    def __init__(self, problem, config_nsga3, random_seed=None):
        self.problem = problem
        self.config = config_nsga3
        self.random_seed = random_seed

    def run(self):
        console.print("\n[bold cyan]Initializing NSGA-III Algorithm...[/bold cyan]")

        # Reference directions: Important for NSGA-III's diversity preservation
        # n_partitions is related to population size for many objectives. For 2 obj, 
        # pop_size - 1 usually works, or a value like 11-15 for good spread.
        # The pop_size should ideally be a multiple of a number related to n_partitions.
        # For 2 objectives, ref_dirs would be pop_size points on a line.
        # A common choice for n_partitions for n_obj=2 is pop_size - 1 or a fixed reasonable number.
        n_partitions = self.config.get('reference_point_partitions', self.config['population_size'] -1 if self.config['population_size'] > 1 else 12) # Heuristic, may need tuning
        if self.problem.n_obj > 1:
            ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=n_partitions)
        else: # Single objective
             ref_dirs = None # Not used for single-objective

        # Sampling method for initial population
        sampling = LHS() # Latin Hypercube Sampling - often good for continuous vars

        # Crossover
        crossover = SBX(prob=self.config.get('crossover_prob', 0.9), 
                        eta=self.config.get('crossover_eta', 15))

        # Mutation
        mutation = PM(prob=self.config.get('mutation_prob', None), # if None, 1/n_var
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

        console.print(f"Starting optimization for [bold]{self.config['num_generations']} generations[/bold] "
                      f"with population size [bold]{self.config['population_size']}[/bold]...")
        
        results = minimize(
            self.problem,
            algorithm,
            termination,
            seed=self.random_seed,
            save_history=True, # Useful for analysis later
            verbose=True # Pymoo's own verbosity
        )
        
        console.print("[bold green]Optimization finished.[/bold green]")
        
        # res.X contains the decision variables of the non-dominated solutions
        # res.F contains the objective values of the non-dominated solutions
        return results