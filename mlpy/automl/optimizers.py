"""
Hyperparameter Optimizers for AutoML
=====================================

Different optimization strategies for hyperparameter search.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
import random
from collections import deque

try:
    from scipy.stats import norm
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from .search_spaces import SearchSpace, ModelSearchSpace, CategoricalSpace, NumericSpace

logger = logging.getLogger(__name__)


@dataclass
class Trial:
    """Represents a single optimization trial."""
    
    config: Dict[str, Any]
    score: float
    model_name: str
    iteration: int
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(
        self,
        search_space: Dict[str, ModelSearchSpace],
        n_trials: int = 100,
        random_state: Optional[int] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            search_space: Dictionary of model search spaces
            n_trials: Maximum number of trials
            random_state: Random seed
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.random_state = random_state
        
        self.trials: List[Trial] = []
        self.best_score = -np.inf
        self.best_params = None
        self.best_model = None
        self.iteration = 0
        
        self.rng = np.random.RandomState(random_state)
    
    @abstractmethod
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get next configuration to evaluate."""
        pass
    
    def update(self, config: Dict[str, Any], score: float):
        """
        Update optimizer with trial result.
        
        Args:
            config: Configuration that was evaluated
            score: Score achieved
        """
        trial = Trial(
            config=config,
            score=score,
            model_name=config.get('model_name', 'Unknown'),
            iteration=self.iteration
        )
        
        self.trials.append(trial)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = config.copy()
            self.best_model = config.get('model')
        
        self.iteration += 1
    
    def get_best(self) -> Tuple[Dict[str, Any], float]:
        """Get best configuration and score."""
        return self.best_params, self.best_score
    
    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        return self.iteration >= self.n_trials


class RandomSearchOptimizer(BaseOptimizer):
    """Random search optimizer."""
    
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get random configuration."""
        if self.should_stop():
            return None
        
        # Randomly select a model
        model_names = list(self.search_space.keys())
        model_name = self.rng.choice(model_names)
        model_space = self.search_space[model_name]
        
        # Sample configuration
        params = model_space.sample_config(self.rng.randint(0, 2**31 - 1))
        
        # Create model
        model = model_space.create_model(params)
        
        return {
            'model': model,
            'params': params,
            'model_name': model_name
        }


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer."""
    
    def __init__(
        self,
        search_space: Dict[str, ModelSearchSpace],
        n_trials: int = 100,
        random_state: Optional[int] = None,
        grid_density: int = 3
    ):
        """
        Initialize grid search.
        
        Args:
            search_space: Search space
            n_trials: Max trials
            random_state: Random seed
            grid_density: Points per dimension
        """
        super().__init__(search_space, n_trials, random_state)
        self.grid_density = grid_density
        self.grid_configs = self._generate_grid()
        self.current_index = 0
    
    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate grid of configurations."""
        configs = []
        
        for model_name, model_space in self.search_space.items():
            # Generate grid for this model
            param_grids = {}
            
            for param_name, space in model_space.parameters.items():
                if isinstance(space, CategoricalSpace):
                    param_grids[param_name] = space.choices
                elif isinstance(space, NumericSpace):
                    low, high = space.get_bounds()
                    if space.dtype == int:
                        values = np.linspace(low, high, self.grid_density, dtype=int)
                    else:
                        values = np.linspace(low, high, self.grid_density)
                    param_grids[param_name] = values.tolist()
            
            # Create all combinations
            import itertools
            keys = param_grids.keys()
            values = param_grids.values()
            
            for combination in itertools.product(*values):
                params = dict(zip(keys, combination))
                model = model_space.create_model(params)
                
                configs.append({
                    'model': model,
                    'params': params,
                    'model_name': model_name
                })
        
        # Shuffle to avoid bias
        self.rng.shuffle(configs)
        
        return configs
    
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get next grid configuration."""
        if self.current_index >= len(self.grid_configs) or self.should_stop():
            return None
        
        config = self.grid_configs[self.current_index]
        self.current_index += 1
        
        return config


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Gaussian Processes."""
    
    def __init__(
        self,
        search_space: Dict[str, ModelSearchSpace],
        n_trials: int = 100,
        random_state: Optional[int] = None,
        n_initial: int = 10,
        acquisition: str = "ei"  # expected improvement
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            search_space: Search space
            n_trials: Max trials
            random_state: Random seed
            n_initial: Initial random samples
            acquisition: Acquisition function ("ei", "ucb", "poi")
        """
        super().__init__(search_space, n_trials, random_state)
        self.n_initial = n_initial
        self.acquisition = acquisition
        
        if not HAS_SCIPY:
            logger.warning("scipy not available, falling back to random search")
            self.fallback_optimizer = RandomSearchOptimizer(
                search_space, n_trials, random_state
            )
    
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get next configuration using Bayesian optimization."""
        if not HAS_SCIPY:
            return self.fallback_optimizer.get_next_config()
        
        if self.should_stop():
            return None
        
        # Use random search for initial points
        if self.iteration < self.n_initial:
            return self._get_random_config()
        
        # Use acquisition function
        return self._optimize_acquisition()
    
    def _get_random_config(self) -> Dict[str, Any]:
        """Get random configuration."""
        model_name = self.rng.choice(list(self.search_space.keys()))
        model_space = self.search_space[model_name]
        
        params = model_space.sample_config(self.rng.randint(0, 2**32))
        model = model_space.create_model(params)
        
        return {
            'model': model,
            'params': params,
            'model_name': model_name
        }
    
    def _optimize_acquisition(self) -> Dict[str, Any]:
        """Optimize acquisition function."""
        # Build surrogate model from trials
        X = []  # configurations
        y = []  # scores
        
        for trial in self.trials:
            # Convert config to numeric vector
            x_vec = self._config_to_vector(trial.config)
            if x_vec is not None:
                X.append(x_vec)
                y.append(trial.score)
        
        if not X:
            return self._get_random_config()
        
        X = np.array(X)
        y = np.array(y)
        
        # Simple Gaussian Process approximation
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            random_state=self.random_state
        )
        
        gp.fit(X, y)
        
        # Optimize acquisition function
        best_config = None
        best_acquisition = -np.inf
        
        # Sample random points and evaluate acquisition
        for _ in range(100):
            config = self._get_random_config()
            x_vec = self._config_to_vector(config)
            
            if x_vec is not None:
                x_vec = x_vec.reshape(1, -1)
                
                # Calculate acquisition value
                mu, sigma = gp.predict(x_vec, return_std=True)
                
                if self.acquisition == "ei":
                    # Expected Improvement
                    improvement = mu - np.max(y)
                    Z = improvement / (sigma + 1e-9)
                    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
                    acq_value = ei[0]
                    
                elif self.acquisition == "ucb":
                    # Upper Confidence Bound
                    acq_value = mu[0] + 2.0 * sigma[0]
                    
                else:  # poi
                    # Probability of Improvement
                    improvement = mu - np.max(y)
                    Z = improvement / (sigma + 1e-9)
                    acq_value = norm.cdf(Z)[0]
                
                if acq_value > best_acquisition:
                    best_acquisition = acq_value
                    best_config = config
        
        return best_config or self._get_random_config()
    
    def _config_to_vector(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert configuration to numeric vector."""
        try:
            vector = []
            params = config.get('params', {})
            
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    vector.append(value)
                elif isinstance(value, bool):
                    vector.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    # Simple hash encoding for strings
                    vector.append(hash(value) % 100)
                    
            return np.array(vector) if vector else None
        except:
            return None


class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary/Genetic algorithm optimizer."""
    
    def __init__(
        self,
        search_space: Dict[str, ModelSearchSpace],
        n_trials: int = 100,
        random_state: Optional[int] = None,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        """
        Initialize evolutionary optimizer.
        
        Args:
            search_space: Search space
            n_trials: Max trials
            random_state: Random seed
            population_size: Size of population
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
        """
        super().__init__(search_space, n_trials, random_state)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitness_scores = []
    
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get next configuration using evolution."""
        if self.should_stop():
            return None
        
        # Initialize population
        if len(self.population) < self.population_size:
            return self._get_random_config()
        
        # Evolve population
        return self._evolve()
    
    def update(self, config: Dict[str, Any], score: float):
        """Update with new result."""
        super().update(config, score)
        
        if len(self.population) < self.population_size:
            self.population.append(config)
            self.fitness_scores.append(score)
        else:
            # Replace worst individual
            min_idx = np.argmin(self.fitness_scores)
            if score > self.fitness_scores[min_idx]:
                self.population[min_idx] = config
                self.fitness_scores[min_idx] = score
    
    def _get_random_config(self) -> Dict[str, Any]:
        """Get random configuration."""
        model_name = self.rng.choice(list(self.search_space.keys()))
        model_space = self.search_space[model_name]
        
        params = model_space.sample_config(self.rng.randint(0, 2**32))
        model = model_space.create_model(params)
        
        return {
            'model': model,
            'params': params,
            'model_name': model_name
        }
    
    def _evolve(self) -> Dict[str, Any]:
        """Evolve population to get next config."""
        # Selection (tournament)
        parent1 = self._tournament_selection()
        parent2 = self._tournament_selection()
        
        # Crossover
        if self.rng.random() < self.crossover_rate:
            child = self._crossover(parent1, parent2)
        else:
            child = parent1.copy()
        
        # Mutation
        if self.rng.random() < self.mutation_rate:
            child = self._mutate(child)
        
        return child
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """Select individual using tournament."""
        indices = self.rng.choice(len(self.population), tournament_size, replace=False)
        scores = [self.fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(scores)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two configurations."""
        # Use same model type
        if parent1['model_name'] != parent2['model_name']:
            return parent1.copy()
        
        child_params = {}
        for key in parent1['params']:
            if self.rng.random() < 0.5:
                child_params[key] = parent1['params'][key]
            else:
                child_params[key] = parent2['params'][key]
        
        model_space = self.search_space[parent1['model_name']]
        model = model_space.create_model(child_params)
        
        return {
            'model': model,
            'params': child_params,
            'model_name': parent1['model_name']
        }
    
    def _mutate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate configuration."""
        model_space = self.search_space[config['model_name']]
        mutated_params = config['params'].copy()
        
        # Mutate one random parameter
        param_names = list(mutated_params.keys())
        if param_names:
            param_to_mutate = self.rng.choice(param_names)
            space = model_space.parameters[param_to_mutate]
            mutated_params[param_to_mutate] = space.sample(self.rng.randint(0, 2**32))
        
        model = model_space.create_model(mutated_params)
        
        return {
            'model': model,
            'params': mutated_params,
            'model_name': config['model_name']
        }


class OptunaOptimizer(BaseOptimizer):
    """Optuna-based optimizer (if available)."""
    
    def __init__(
        self,
        search_space: Dict[str, ModelSearchSpace],
        n_trials: int = 100,
        random_state: Optional[int] = None
    ):
        """Initialize Optuna optimizer."""
        super().__init__(search_space, n_trials, random_state)
        
        if not HAS_OPTUNA:
            logger.warning("Optuna not available, falling back to Bayesian optimization")
            self.fallback_optimizer = BayesianOptimizer(
                search_space, n_trials, random_state
            )
            return
        
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        self.current_trial = None
    
    def get_next_config(self) -> Optional[Dict[str, Any]]:
        """Get next configuration from Optuna."""
        if not HAS_OPTUNA:
            return self.fallback_optimizer.get_next_config()
        
        if self.should_stop():
            return None
        
        # Create Optuna trial
        trial = self.study.ask()
        
        # Sample model and parameters
        model_name = trial.suggest_categorical('model', list(self.search_space.keys()))
        model_space = self.search_space[model_name]
        
        params = {}
        for param_name, space in model_space.parameters.items():
            if isinstance(space, CategoricalSpace):
                params[param_name] = trial.suggest_categorical(
                    f"{model_name}_{param_name}",
                    space.choices
                )
            elif isinstance(space, NumericSpace):
                low, high = space.get_bounds()
                if space.dtype == int:
                    params[param_name] = trial.suggest_int(
                        f"{model_name}_{param_name}",
                        int(low), int(high),
                        log=space.log_scale
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        f"{model_name}_{param_name}",
                        low, high,
                        log=space.log_scale
                    )
        
        model = model_space.create_model(params)
        
        self.current_trial = trial
        
        return {
            'model': model,
            'params': params,
            'model_name': model_name,
            'trial': trial
        }
    
    def update(self, config: Dict[str, Any], score: float):
        """Update Optuna with result."""
        super().update(config, score)
        
        if HAS_OPTUNA and self.current_trial:
            self.study.tell(self.current_trial, score)
            self.current_trial = None


def get_optimizer(
    name: str,
    search_space: Dict[str, ModelSearchSpace],
    n_trials: int = 100,
    random_state: Optional[int] = None,
    **kwargs
) -> BaseOptimizer:
    """
    Get optimizer by name.
    
    Args:
        name: Optimizer name
        search_space: Search space
        n_trials: Max trials
        random_state: Random seed
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'random': RandomSearchOptimizer,
        'grid': GridSearchOptimizer,
        'bayesian': BayesianOptimizer,
        'evolutionary': EvolutionaryOptimizer,
        'optuna': OptunaOptimizer
    }
    
    if name not in optimizers:
        logger.warning(f"Unknown optimizer {name}, using random search")
        name = 'random'
    
    optimizer_class = optimizers[name]
    
    return optimizer_class(
        search_space=search_space,
        n_trials=n_trials,
        random_state=random_state,
        **kwargs
    )