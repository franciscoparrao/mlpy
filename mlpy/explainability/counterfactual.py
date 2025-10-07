"""
Counterfactual Explanations
===========================

Generate counterfactual examples to explain model decisions.
"What would need to change for a different outcome?"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import warnings
import logging
from scipy.spatial.distance import euclidean, cosine
from scipy.optimize import minimize
import copy

logger = logging.getLogger(__name__)


@dataclass
class Counterfactual:
    """Container for counterfactual explanation."""
    
    original: np.ndarray
    counterfactual: np.ndarray
    original_prediction: Union[float, int, str]
    counterfactual_prediction: Union[float, int, str]
    feature_changes: Dict[str, Tuple[float, float]]  # feature -> (original, new)
    distance: float
    sparsity: int  # Number of features changed
    validity: bool  # Whether desired outcome achieved
    feature_names: List[str]
    
    def get_changes_summary(self) -> str:
        """Get human-readable summary of changes."""
        changes = []
        for feature, (orig, new) in self.feature_changes.items():
            change = new - orig
            if abs(change) > 1e-6:
                changes.append(f"{feature}: {orig:.3f} → {new:.3f} ({change:+.3f})")
        return "\n".join(changes)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for easy viewing."""
        data = {
            'Feature': [],
            'Original': [],
            'Counterfactual': [],
            'Change': [],
            'Change%': []
        }
        
        for feature, (orig, new) in self.feature_changes.items():
            data['Feature'].append(feature)
            data['Original'].append(orig)
            data['Counterfactual'].append(new)
            data['Change'].append(new - orig)
            data['Change%'].append(((new - orig) / (abs(orig) + 1e-10)) * 100)
        
        return pd.DataFrame(data)


class CounterfactualExplainer:
    """Generate counterfactual explanations for model decisions."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        categorical_features: Optional[List[str]] = None,
        immutable_features: Optional[List[str]] = None,
        distance_metric: str = "euclidean",
        sparsity_weight: float = 0.1,
        diversity_weight: float = 0.1
    ):
        """
        Initialize counterfactual explainer.
        
        Args:
            model: Trained model
            feature_names: Names of features
            feature_ranges: Valid ranges for features
            categorical_features: List of categorical feature names
            immutable_features: Features that cannot be changed
            distance_metric: Distance metric to use
            sparsity_weight: Weight for sparsity in objective
            diversity_weight: Weight for diversity among counterfactuals
        """
        self.model = model
        self.feature_names = feature_names or []
        self.feature_ranges = feature_ranges or {}
        self.categorical_features = categorical_features or []
        self.immutable_features = immutable_features or []
        self.distance_metric = distance_metric
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        
        # Create feature indices
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        self.immutable_indices = [
            self.feature_indices[f] for f in self.immutable_features 
            if f in self.feature_indices
        ]
        self.categorical_indices = [
            self.feature_indices[f] for f in self.categorical_features
            if f in self.feature_indices
        ]
    
    def generate(
        self,
        instance: Union[pd.Series, np.ndarray],
        desired_outcome: Optional[Union[int, float]] = None,
        desired_range: Optional[Tuple[float, float]] = None,
        max_features_changed: Optional[int] = None,
        num_counterfactuals: int = 1,
        method: str = "optimization"
    ) -> Union[Counterfactual, List[Counterfactual]]:
        """
        Generate counterfactual explanation(s).
        
        Args:
            instance: Instance to explain
            desired_outcome: Desired prediction value
            desired_range: Desired prediction range (min, max)
            max_features_changed: Maximum features to change
            num_counterfactuals: Number of counterfactuals to generate
            method: "optimization", "genetic", or "random"
            
        Returns:
            Single Counterfactual or list of Counterfactuals
        """
        if isinstance(instance, pd.Series):
            instance = instance.values
        
        # Get original prediction
        original_pred = self._predict(instance)
        
        # Determine target
        if desired_outcome is None and desired_range is None:
            # For classification, flip the class
            if hasattr(self.model, 'predict_proba'):
                desired_outcome = 1 - int(original_pred)
            else:
                # For regression, aim for significant change
                desired_outcome = original_pred * 1.5
        
        # Generate counterfactuals
        if method == "optimization":
            cfs = self._generate_optimization(
                instance, desired_outcome, desired_range,
                max_features_changed, num_counterfactuals
            )
        elif method == "genetic":
            cfs = self._generate_genetic(
                instance, desired_outcome, desired_range,
                max_features_changed, num_counterfactuals
            )
        elif method == "random":
            cfs = self._generate_random(
                instance, desired_outcome, desired_range,
                max_features_changed, num_counterfactuals
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return cfs[0] if num_counterfactuals == 1 else cfs
    
    def _predict(self, instance: np.ndarray) -> Union[float, int]:
        """Get model prediction for instance."""
        instance = instance.reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            # Classification: return class
            proba = self.model.predict_proba(instance)[0]
            return np.argmax(proba)
        else:
            # Regression: return value
            return self.model.predict(instance)[0]
    
    def _generate_optimization(
        self,
        instance: np.ndarray,
        desired_outcome: Optional[Union[int, float]],
        desired_range: Optional[Tuple[float, float]],
        max_features_changed: Optional[int],
        num_counterfactuals: int
    ) -> List[Counterfactual]:
        """Generate counterfactuals using optimization."""
        counterfactuals = []
        
        for i in range(num_counterfactuals):
            # Define objective function
            def objective(x):
                # Prediction loss
                pred = self._predict(x)
                
                if desired_outcome is not None:
                    pred_loss = abs(pred - desired_outcome)
                elif desired_range is not None:
                    if desired_range[0] <= pred <= desired_range[1]:
                        pred_loss = 0
                    else:
                        pred_loss = min(abs(pred - desired_range[0]), 
                                      abs(pred - desired_range[1]))
                else:
                    pred_loss = 0
                
                # Distance loss
                if self.distance_metric == "euclidean":
                    dist_loss = euclidean(instance, x)
                elif self.distance_metric == "cosine":
                    dist_loss = cosine(instance, x)
                else:
                    dist_loss = np.linalg.norm(instance - x)
                
                # Sparsity loss (L0 norm approximation)
                sparsity_loss = np.sum(np.abs(instance - x) > 1e-6)
                
                # Diversity loss (if generating multiple)
                diversity_loss = 0
                if i > 0 and counterfactuals:
                    for cf in counterfactuals:
                        diversity_loss -= self._calculate_distance(
                            x, cf.counterfactual
                        )
                
                # Combined loss
                loss = (pred_loss + 
                       dist_loss + 
                       self.sparsity_weight * sparsity_loss +
                       self.diversity_weight * diversity_loss)
                
                return loss
            
            # Set bounds
            bounds = []
            for j in range(len(instance)):
                if j in self.immutable_indices:
                    # Immutable feature: fix to original value
                    bounds.append((instance[j], instance[j]))
                elif self.feature_names and self.feature_names[j] in self.feature_ranges:
                    # Use specified range
                    bounds.append(self.feature_ranges[self.feature_names[j]])
                else:
                    # Default bounds
                    bounds.append((instance[j] - 2, instance[j] + 2))
            
            # Constraint for max features changed
            constraints = []
            if max_features_changed is not None:
                def sparsity_constraint(x):
                    return max_features_changed - np.sum(np.abs(instance - x) > 1e-6)
                constraints.append({'type': 'ineq', 'fun': sparsity_constraint})
            
            # Optimize
            result = minimize(
                objective,
                instance,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints
            )
            
            # Create counterfactual
            cf_instance = result.x
            cf_pred = self._predict(cf_instance)
            
            # Check validity
            if desired_outcome is not None:
                valid = abs(cf_pred - desired_outcome) < 0.1
            elif desired_range is not None:
                valid = desired_range[0] <= cf_pred <= desired_range[1]
            else:
                valid = True
            
            # Calculate changes
            feature_changes = {}
            for j, name in enumerate(self.feature_names):
                if abs(instance[j] - cf_instance[j]) > 1e-6:
                    feature_changes[name] = (instance[j], cf_instance[j])
            
            cf = Counterfactual(
                original=instance,
                counterfactual=cf_instance,
                original_prediction=self._predict(instance),
                counterfactual_prediction=cf_pred,
                feature_changes=feature_changes,
                distance=self._calculate_distance(instance, cf_instance),
                sparsity=len(feature_changes),
                validity=valid,
                feature_names=self.feature_names
            )
            
            counterfactuals.append(cf)
        
        return counterfactuals
    
    def _generate_genetic(
        self,
        instance: np.ndarray,
        desired_outcome: Optional[Union[int, float]],
        desired_range: Optional[Tuple[float, float]],
        max_features_changed: Optional[int],
        num_counterfactuals: int,
        population_size: int = 100,
        generations: int = 50
    ) -> List[Counterfactual]:
        """Generate counterfactuals using genetic algorithm."""
        counterfactuals = []
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = instance.copy()
            
            # Randomly mutate some features
            num_changes = np.random.randint(1, min(5, len(instance)))
            changeable_indices = [
                i for i in range(len(instance)) 
                if i not in self.immutable_indices
            ]
            
            if changeable_indices:
                indices_to_change = np.random.choice(
                    changeable_indices, 
                    size=min(num_changes, len(changeable_indices)),
                    replace=False
                )
                
                for idx in indices_to_change:
                    if self.feature_names and self.feature_names[idx] in self.feature_ranges:
                        low, high = self.feature_ranges[self.feature_names[idx]]
                    else:
                        low, high = instance[idx] - 2, instance[idx] + 2
                    
                    individual[idx] = np.random.uniform(low, high)
            
            population.append(individual)
        
        # Evolution
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                pred = self._predict(individual)
                
                # Calculate fitness (inverse of loss)
                if desired_outcome is not None:
                    outcome_fitness = 1.0 / (1.0 + abs(pred - desired_outcome))
                elif desired_range is not None:
                    if desired_range[0] <= pred <= desired_range[1]:
                        outcome_fitness = 1.0
                    else:
                        dist = min(abs(pred - desired_range[0]), 
                                 abs(pred - desired_range[1]))
                        outcome_fitness = 1.0 / (1.0 + dist)
                else:
                    outcome_fitness = 1.0
                
                distance_fitness = 1.0 / (1.0 + self._calculate_distance(instance, individual))
                sparsity_fitness = 1.0 / (1.0 + np.sum(np.abs(instance - individual) > 1e-6))
                
                fitness = outcome_fitness * distance_fitness * sparsity_fitness
                fitness_scores.append(fitness)
            
            # Selection
            fitness_scores = np.array(fitness_scores)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            
            # Keep top individuals
            new_population = [population[i] for i in sorted_indices[:population_size//2]]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Select parents
                parent1 = new_population[np.random.randint(len(new_population))]
                parent2 = new_population[np.random.randint(len(new_population))]
                
                # Crossover
                child = parent1.copy()
                mask = np.random.random(len(child)) < 0.5
                child[mask] = parent2[mask]
                
                # Mutation
                if np.random.random() < 0.1:
                    idx = np.random.choice([
                        i for i in range(len(child)) 
                        if i not in self.immutable_indices
                    ])
                    if self.feature_names and self.feature_names[idx] in self.feature_ranges:
                        low, high = self.feature_ranges[self.feature_names[idx]]
                    else:
                        low, high = instance[idx] - 2, instance[idx] + 2
                    child[idx] = np.random.uniform(low, high)
                
                new_population.append(child)
            
            population = new_population
        
        # Select best individuals as counterfactuals
        final_fitness = []
        for individual in population:
            pred = self._predict(individual)
            if desired_outcome is not None:
                valid = abs(pred - desired_outcome) < 0.1
            elif desired_range is not None:
                valid = desired_range[0] <= pred <= desired_range[1]
            else:
                valid = True
            
            if valid:
                final_fitness.append((individual, self._calculate_distance(instance, individual)))
        
        # Sort by distance and select top n
        final_fitness.sort(key=lambda x: x[1])
        
        for i in range(min(num_counterfactuals, len(final_fitness))):
            cf_instance = final_fitness[i][0]
            cf_pred = self._predict(cf_instance)
            
            # Calculate changes
            feature_changes = {}
            for j, name in enumerate(self.feature_names):
                if abs(instance[j] - cf_instance[j]) > 1e-6:
                    feature_changes[name] = (instance[j], cf_instance[j])
            
            cf = Counterfactual(
                original=instance,
                counterfactual=cf_instance,
                original_prediction=self._predict(instance),
                counterfactual_prediction=cf_pred,
                feature_changes=feature_changes,
                distance=final_fitness[i][1],
                sparsity=len(feature_changes),
                validity=True,
                feature_names=self.feature_names
            )
            
            counterfactuals.append(cf)
        
        # Fill with random if not enough valid ones found
        while len(counterfactuals) < num_counterfactuals:
            counterfactuals.append(self._generate_random(
                instance, desired_outcome, desired_range, 
                max_features_changed, 1
            )[0])
        
        return counterfactuals
    
    def _generate_random(
        self,
        instance: np.ndarray,
        desired_outcome: Optional[Union[int, float]],
        desired_range: Optional[Tuple[float, float]],
        max_features_changed: Optional[int],
        num_counterfactuals: int,
        max_attempts: int = 1000
    ) -> List[Counterfactual]:
        """Generate counterfactuals using random search."""
        counterfactuals = []
        
        for _ in range(num_counterfactuals):
            best_cf = None
            best_distance = float('inf')
            
            for attempt in range(max_attempts):
                cf_instance = instance.copy()
                
                # Randomly select features to change
                changeable_indices = [
                    i for i in range(len(instance)) 
                    if i not in self.immutable_indices
                ]
                
                if max_features_changed:
                    num_changes = min(max_features_changed, len(changeable_indices))
                else:
                    num_changes = np.random.randint(1, max(2, len(changeable_indices)//2))
                
                if changeable_indices:
                    indices_to_change = np.random.choice(
                        changeable_indices,
                        size=min(num_changes, len(changeable_indices)),
                        replace=False
                    )
                    
                    for idx in indices_to_change:
                        if self.feature_names and self.feature_names[idx] in self.feature_ranges:
                            low, high = self.feature_ranges[self.feature_names[idx]]
                        else:
                            # Use wider range for random search
                            low, high = instance[idx] - 3, instance[idx] + 3
                        
                        cf_instance[idx] = np.random.uniform(low, high)
                
                # Check if valid
                cf_pred = self._predict(cf_instance)
                
                if desired_outcome is not None:
                    valid = abs(cf_pred - desired_outcome) < 0.1
                elif desired_range is not None:
                    valid = desired_range[0] <= cf_pred <= desired_range[1]
                else:
                    valid = True
                
                if valid:
                    distance = self._calculate_distance(instance, cf_instance)
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = cf_instance.copy()
            
            # Use best found or last attempt
            if best_cf is None:
                best_cf = cf_instance
            
            cf_pred = self._predict(best_cf)
            
            # Calculate changes
            feature_changes = {}
            for j, name in enumerate(self.feature_names):
                if abs(instance[j] - best_cf[j]) > 1e-6:
                    feature_changes[name] = (instance[j], best_cf[j])
            
            cf = Counterfactual(
                original=instance,
                counterfactual=best_cf,
                original_prediction=self._predict(instance),
                counterfactual_prediction=cf_pred,
                feature_changes=feature_changes,
                distance=self._calculate_distance(instance, best_cf),
                sparsity=len(feature_changes),
                validity=(best_cf is not cf_instance),  # Valid if we found a good one
                feature_names=self.feature_names
            )
            
            counterfactuals.append(cf)
        
        return counterfactuals
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance between two instances."""
        if self.distance_metric == "euclidean":
            return euclidean(x1, x2)
        elif self.distance_metric == "cosine":
            return cosine(x1, x2)
        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        else:
            return np.linalg.norm(x1 - x2)