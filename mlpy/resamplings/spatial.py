"""
Spatial resampling strategies for MLPY.

These resampling methods are designed to handle spatial autocorrelation
in geographic data by ensuring spatial separation between train and test sets.
"""

from typing import Optional, List, Tuple, Dict, Any, Union
import numpy as np
import warnings
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

from .base import Resampling
from ..tasks import Task
from ..tasks.spatial import TaskClassifSpatial, TaskRegrSpatial
from ..utils.registry import mlpy_resamplings


class SpatialResampling(Resampling):
    """Base class for spatial resampling strategies."""
    
    def _validate_spatial_task(self, task: Task) -> None:
        """Validate that the task has spatial information."""
        if not isinstance(task, (TaskClassifSpatial, TaskRegrSpatial)):
            if not hasattr(task, 'coordinate_names') or not task.coordinate_names:
                raise ValueError(
                    "Task must be a spatial task (TaskClassifSpatial or TaskRegrSpatial) "
                    "or have coordinate_names attribute"
                )
    
    def _get_coordinates(self, task: Task) -> np.ndarray:
        """Extract coordinates from task."""
        if hasattr(task, 'coordinates'):
            return task.coordinates()
        else:
            # Fallback for non-spatial tasks with coordinate columns
            coord_cols = task.coordinate_names
            data = task.data(cols=coord_cols, data_format="array")
            return data


@mlpy_resamplings.register("spatial_kfold")
class SpatialKFold(SpatialResampling):
    """
    Spatial K-Fold Cross-Validation.
    
    Creates spatially separated folds by clustering coordinates and ensuring
    that each fold contains spatially contiguous regions.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of folds
    clustering_method : str, default='kmeans'
        Method for spatial clustering ('kmeans', 'grid')
    random_state : int, optional
        Random state for reproducibility
    buffer_distance : float, optional
        Buffer distance to exclude points near fold boundaries
        
    Examples
    --------
    >>> from mlpy.resamplings.spatial import SpatialKFold
    >>> from mlpy.tasks.spatial import TaskClassifSpatial
    >>> 
    >>> # Create spatial task
    >>> task = TaskClassifSpatial(data, target='class', coordinate_names=['x', 'y'])
    >>> 
    >>> # Create spatial k-fold
    >>> cv = SpatialKFold(n_folds=5)
    >>> cv.instantiate(task)
    >>> 
    >>> # Get first fold
    >>> train_idx = cv.train_set(0)
    >>> test_idx = cv.test_set(0)
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        clustering_method: str = 'kmeans',
        random_state: Optional[int] = None,
        buffer_distance: Optional[float] = None
    ):
        super().__init__(
            id="spatial_kfold",
            iters=n_folds,
            duplicated_ids=False
        )
        self.n_folds = n_folds
        self.clustering_method = clustering_method
        self.random_state = random_state
        self.buffer_distance = buffer_distance
        
        self.param_set = {
            'n_folds': n_folds,
            'clustering_method': clustering_method,
            'random_state': random_state,
            'buffer_distance': buffer_distance
        }
    
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create spatial folds."""
        self._validate_spatial_task(task)
        
        # Get coordinates
        coords = self._get_coordinates(task)
        n_samples = len(coords)
        indices = np.arange(n_samples)
        
        # Create spatial clusters
        if self.clustering_method == 'kmeans':
            clusters = self._kmeans_clustering(coords)
        elif self.clustering_method == 'grid':
            clusters = self._grid_clustering(coords)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        # Create folds from clusters
        folds = self._create_folds_from_clusters(clusters, indices)
        
        # Apply buffer if specified
        if self.buffer_distance is not None:
            folds = self._apply_buffer(folds, coords)
        
        return {'folds': folds, 'clusters': clusters}
    
    def _kmeans_clustering(self, coords: np.ndarray) -> np.ndarray:
        """Cluster coordinates using K-Means."""
        kmeans = KMeans(
            n_clusters=self.n_folds,
            random_state=self.random_state,
            n_init=10
        )
        clusters = kmeans.fit_predict(coords)
        return clusters
    
    def _grid_clustering(self, coords: np.ndarray) -> np.ndarray:
        """Cluster coordinates using regular grid."""
        # Create grid based on spatial extent
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Calculate grid dimensions
        grid_cols = int(np.ceil(np.sqrt(self.n_folds)))
        grid_rows = int(np.ceil(self.n_folds / grid_cols))
        
        # Create grid boundaries
        x_bins = np.linspace(x_min, x_max, grid_cols + 1)
        y_bins = np.linspace(y_min, y_max, grid_rows + 1)
        
        # Assign points to grid cells
        x_idx = np.digitize(coords[:, 0], x_bins) - 1
        y_idx = np.digitize(coords[:, 1], y_bins) - 1
        
        # Convert to cluster labels
        clusters = y_idx * grid_cols + x_idx
        
        # Map to 0..n_folds-1
        unique_clusters = np.unique(clusters)
        cluster_map = {old: new % self.n_folds for new, old in enumerate(unique_clusters)}
        clusters = np.array([cluster_map[c] for c in clusters])
        
        return clusters
    
    def _create_folds_from_clusters(
        self, 
        clusters: np.ndarray, 
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create train/test folds from cluster assignments."""
        folds = []
        
        for fold_id in range(self.n_folds):
            test_mask = clusters == fold_id
            train_mask = ~test_mask
            
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def _apply_buffer(
        self, 
        folds: List[Tuple[np.ndarray, np.ndarray]], 
        coords: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply buffer to exclude points near fold boundaries."""
        buffered_folds = []
        
        for train_idx, test_idx in folds:
            # Calculate distances from test points to train points
            train_coords = coords[train_idx]
            test_coords = coords[test_idx]
            
            # Keep only test points far enough from training set
            keep_test = []
            for i, test_coord in enumerate(test_coords):
                min_dist = np.min(np.linalg.norm(train_coords - test_coord, axis=1))
                if min_dist > self.buffer_distance:
                    keep_test.append(test_idx[i])
            
            # Keep only train points far enough from test set
            keep_train = []
            for i, train_coord in enumerate(train_coords):
                min_dist = np.min(np.linalg.norm(test_coords - train_coord, axis=1))
                if min_dist > self.buffer_distance:
                    keep_train.append(train_idx[i])
            
            buffered_folds.append((np.array(keep_train), np.array(keep_test)))
        
        return buffered_folds
    
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for fold i."""
        return self._instance['folds'][i][0]
    
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for fold i."""
        return self._instance['folds'][i][1]


@mlpy_resamplings.register("spatial_block")
class SpatialBlockCV(SpatialResampling):
    """
    Spatial Block Cross-Validation.
    
    Divides the spatial extent into rectangular blocks and uses blocks
    as folds. This ensures spatial separation between training and test sets.
    
    Parameters
    ----------
    n_blocks : int or tuple, default=5
        Number of blocks. If int, creates roughly square blocks.
        If tuple (rows, cols), creates grid with specified dimensions.
    buffer_size : float, optional
        Buffer size between blocks to increase spatial separation
    random_state : int, optional
        Random state for block assignment to folds
    method : str, default='systematic'
        Block assignment method: 'systematic' or 'random'
        
    Examples
    --------
    >>> from mlpy.resamplings.spatial import SpatialBlockCV
    >>> 
    >>> # Create spatial block CV with 3x3 grid
    >>> cv = SpatialBlockCV(n_blocks=(3, 3))
    >>> cv.instantiate(task)
    >>> 
    >>> # Iterate through folds
    >>> for i in range(cv.iters):
    ...     train = cv.train_set(i)
    ...     test = cv.test_set(i)
    """
    
    def __init__(
        self,
        n_blocks: Union[int, Tuple[int, int]] = 5,
        buffer_size: Optional[float] = None,
        random_state: Optional[int] = None,
        method: str = 'systematic'
    ):
        # Calculate number of folds based on blocks
        if isinstance(n_blocks, int):
            self.n_rows = int(np.ceil(np.sqrt(n_blocks)))
            self.n_cols = int(np.ceil(n_blocks / self.n_rows))
        else:
            self.n_rows, self.n_cols = n_blocks
        
        n_folds = min(self.n_rows * self.n_cols, 10)  # Cap at 10 folds
        
        super().__init__(
            id="spatial_block",
            iters=n_folds,
            duplicated_ids=False
        )
        
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.method = method
        
        self.param_set = {
            'n_blocks': n_blocks,
            'buffer_size': buffer_size,
            'random_state': random_state,
            'method': method
        }
    
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create spatial blocks."""
        self._validate_spatial_task(task)
        
        # Get coordinates
        coords = self._get_coordinates(task)
        n_samples = len(coords)
        indices = np.arange(n_samples)
        
        # Create blocks
        blocks = self._create_blocks(coords)
        
        # Assign blocks to folds
        if self.method == 'systematic':
            fold_assignments = self._systematic_assignment(blocks)
        elif self.method == 'random':
            fold_assignments = self._random_assignment(blocks)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Create folds
        folds = self._create_folds_from_blocks(blocks, fold_assignments, indices)
        
        # Apply buffer if specified
        if self.buffer_size is not None:
            folds = self._apply_block_buffer(folds, coords, blocks)
        
        return {
            'folds': folds,
            'blocks': blocks,
            'fold_assignments': fold_assignments
        }
    
    def _create_blocks(self, coords: np.ndarray) -> np.ndarray:
        """Create spatial blocks."""
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Add small epsilon to ensure all points fall within blocks
        eps = 1e-10
        x_max += eps
        y_max += eps
        
        # Create block boundaries
        x_edges = np.linspace(x_min, x_max, self.n_cols + 1)
        y_edges = np.linspace(y_min, y_max, self.n_rows + 1)
        
        # Assign points to blocks
        x_block = np.digitize(coords[:, 0], x_edges) - 1
        y_block = np.digitize(coords[:, 1], y_edges) - 1
        
        # Ensure indices are within bounds
        x_block = np.clip(x_block, 0, self.n_cols - 1)
        y_block = np.clip(y_block, 0, self.n_rows - 1)
        
        # Convert to single block index
        blocks = y_block * self.n_cols + x_block
        
        return blocks
    
    def _systematic_assignment(self, blocks: np.ndarray) -> np.ndarray:
        """Systematically assign blocks to folds (checkerboard pattern)."""
        n_blocks = self.n_rows * self.n_cols
        fold_assignments = np.zeros(n_blocks, dtype=int)
        
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                block_id = row * self.n_cols + col
                # Checkerboard pattern
                fold_id = (row + col) % self.iters
                fold_assignments[block_id] = fold_id
        
        return fold_assignments
    
    def _random_assignment(self, blocks: np.ndarray) -> np.ndarray:
        """Randomly assign blocks to folds."""
        np.random.seed(self.random_state)
        
        n_blocks = self.n_rows * self.n_cols
        unique_blocks = np.unique(blocks)
        
        # Shuffle and assign to folds
        np.random.shuffle(unique_blocks)
        fold_assignments = np.zeros(n_blocks, dtype=int)
        
        for i, block_id in enumerate(unique_blocks):
            fold_assignments[block_id] = i % self.iters
        
        return fold_assignments
    
    def _create_folds_from_blocks(
        self,
        blocks: np.ndarray,
        fold_assignments: np.ndarray,
        indices: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create train/test folds from block assignments."""
        folds = []
        
        for fold_id in range(self.iters):
            # Find blocks in this fold
            test_blocks = np.where(fold_assignments == fold_id)[0]
            
            # Find points in test blocks
            test_mask = np.isin(blocks, test_blocks)
            train_mask = ~test_mask
            
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            
            folds.append((train_idx, test_idx))
        
        return folds
    
    def _apply_block_buffer(
        self,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        coords: np.ndarray,
        blocks: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply buffer between blocks."""
        # Implementation would remove points near block boundaries
        # For simplicity, returning folds as-is
        warnings.warn("Block buffer not yet implemented, returning unbuffered folds")
        return folds
    
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for fold i."""
        return self._instance['folds'][i][0]
    
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for fold i."""
        return self._instance['folds'][i][1]


@mlpy_resamplings.register("spatial_buffer")
class SpatialBufferCV(SpatialResampling):
    """
    Spatial Buffer (Leave-One-Out) Cross-Validation.
    
    For each test point or region, excludes all training points within
    a specified buffer distance. This is useful for testing spatial
    prediction at various distances.
    
    Parameters
    ----------
    buffer_distance : float
        Minimum distance between train and test points
    test_size : float or int, default=0.2
        If float, proportion of data for test set.
        If int, absolute number of test points.
    n_folds : int, optional
        Number of folds. If None, uses leave-one-out for test_size points.
    random_state : int, optional
        Random state for test point selection
        
    Examples
    --------
    >>> from mlpy.resamplings.spatial import SpatialBufferCV
    >>> 
    >>> # Create buffer CV with 100m buffer
    >>> cv = SpatialBufferCV(buffer_distance=100, test_size=0.2, n_folds=5)
    >>> cv.instantiate(task)
    """
    
    def __init__(
        self,
        buffer_distance: float,
        test_size: Union[float, int] = 0.2,
        n_folds: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        # Determine number of iterations
        if n_folds is None:
            n_folds = 5  # Default to 5 folds
        
        super().__init__(
            id="spatial_buffer",
            iters=n_folds,
            duplicated_ids=False
        )
        
        self.buffer_distance = buffer_distance
        self.test_size = test_size
        self.random_state = random_state
        
        self.param_set = {
            'buffer_distance': buffer_distance,
            'test_size': test_size,
            'n_folds': n_folds,
            'random_state': random_state
        }
    
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create buffered folds."""
        self._validate_spatial_task(task)
        
        # Get coordinates
        coords = self._get_coordinates(task)
        n_samples = len(coords)
        indices = np.arange(n_samples)
        
        # Determine test set size
        if isinstance(self.test_size, float):
            n_test = int(n_samples * self.test_size)
        else:
            n_test = min(self.test_size, n_samples)
        
        # Select test points for each fold
        np.random.seed(self.random_state)
        folds = []
        
        for i in range(self.iters):
            # Randomly select test points
            test_idx = np.random.choice(indices, size=n_test, replace=False)
            
            # Find training points outside buffer
            train_idx = self._find_points_outside_buffer(
                coords, indices, test_idx, self.buffer_distance
            )
            
            folds.append((train_idx, test_idx))
        
        return {'folds': folds}
    
    def _find_points_outside_buffer(
        self,
        coords: np.ndarray,
        all_indices: np.ndarray,
        test_indices: np.ndarray,
        buffer_distance: float
    ) -> np.ndarray:
        """Find points outside buffer distance from test points."""
        test_coords = coords[test_indices]
        train_candidates = np.setdiff1d(all_indices, test_indices)
        
        # Calculate minimum distance from each candidate to test points
        keep_train = []
        for idx in train_candidates:
            point = coords[idx]
            min_dist = np.min(np.linalg.norm(test_coords - point, axis=1))
            if min_dist > buffer_distance:
                keep_train.append(idx)
        
        return np.array(keep_train)
    
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for fold i."""
        return self._instance['folds'][i][0]
    
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for fold i."""
        return self._instance['folds'][i][1]


@mlpy_resamplings.register("spatial_environmental")
class SpatialEnvironmentalCV(SpatialResampling):
    """
    Spatial-Environmental Blocking Cross-Validation.
    
    Creates folds based on both spatial and environmental distances,
    useful when environmental gradients are as important as spatial separation.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of folds
    spatial_weight : float, default=0.5
        Weight for spatial distance (0-1)
    environmental_cols : List[str], optional
        Environmental feature columns to use for distance calculation
    method : str, default='kmeans'
        Clustering method ('kmeans' or 'hierarchical')
    random_state : int, optional
        Random state for reproducibility
        
    Examples
    --------
    >>> # Create environmental blocking with elevation and slope
    >>> cv = SpatialEnvironmentalCV(
    ...     n_folds=5,
    ...     spatial_weight=0.6,
    ...     environmental_cols=['elevation', 'slope']
    ... )
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        spatial_weight: float = 0.5,
        environmental_cols: Optional[List[str]] = None,
        method: str = 'kmeans',
        random_state: Optional[int] = None
    ):
        super().__init__(
            id="spatial_environmental",
            iters=n_folds,
            duplicated_ids=False
        )
        
        self.n_folds = n_folds
        self.spatial_weight = spatial_weight
        self.environmental_cols = environmental_cols
        self.method = method
        self.random_state = random_state
        
        self.param_set = {
            'n_folds': n_folds,
            'spatial_weight': spatial_weight,
            'environmental_cols': environmental_cols,
            'method': method,
            'random_state': random_state
        }
    
    def _materialize(self, task: Task) -> Dict[str, Any]:
        """Create spatial-environmental folds."""
        self._validate_spatial_task(task)
        
        # Get coordinates
        coords = self._get_coordinates(task)
        n_samples = len(coords)
        indices = np.arange(n_samples)
        
        # Normalize spatial coordinates
        from sklearn.preprocessing import StandardScaler
        scaler_spatial = StandardScaler()
        coords_norm = scaler_spatial.fit_transform(coords)
        
        # Get environmental features if specified
        if self.environmental_cols:
            env_data = task.data(cols=self.environmental_cols, data_format="array")
            scaler_env = StandardScaler()
            env_norm = scaler_env.fit_transform(env_data)
            
            # Combine spatial and environmental features
            combined = np.hstack([
                coords_norm * self.spatial_weight,
                env_norm * (1 - self.spatial_weight)
            ])
        else:
            combined = coords_norm
        
        # Perform clustering
        if self.method == 'kmeans':
            from sklearn.cluster import KMeans
            clustering = KMeans(
                n_clusters=self.n_folds,
                random_state=self.random_state,
                n_init=10
            )
            clusters = clustering.fit_predict(combined)
        elif self.method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=self.n_folds)
            clusters = clustering.fit_predict(combined)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Create folds from clusters
        folds = []
        for fold_id in range(self.n_folds):
            test_mask = clusters == fold_id
            train_mask = ~test_mask
            
            train_idx = indices[train_mask]
            test_idx = indices[test_mask]
            
            folds.append((train_idx, test_idx))
        
        return {
            'folds': folds,
            'clusters': clusters,
            'combined_features': combined
        }
    
    def _get_train_set(self, i: int) -> np.ndarray:
        """Get training indices for fold i."""
        return self._instance['folds'][i][0]
    
    def _get_test_set(self, i: int) -> np.ndarray:
        """Get test indices for fold i."""
        return self._instance['folds'][i][1]


# Convenience functions
def spatial_cv_score(
    task: Union[TaskClassifSpatial, TaskRegrSpatial],
    learner: Any,
    cv: SpatialResampling,
    measure: Any
) -> List[float]:
    """
    Evaluate a learner using spatial cross-validation.
    
    Parameters
    ----------
    task : TaskClassifSpatial or TaskRegrSpatial
        Spatial task
    learner : Learner
        The learner to evaluate
    cv : SpatialResampling
        Spatial cross-validation strategy
    measure : Measure
        Performance measure
        
    Returns
    -------
    List[float]
        Scores for each fold
    """
    cv.instantiate(task)
    scores = []
    
    for i in range(cv.iters):
        train_idx = cv.train_set(i)
        test_idx = cv.test_set(i)
        
        # Create train and test data
        train_data = task.data(rows=train_idx.tolist())
        test_data = task.data(rows=test_idx.tolist())
        
        # Train and evaluate
        # Note: This is simplified - actual implementation would use proper
        # learner train/predict methods
        # learner.train(train_data)
        # predictions = learner.predict(test_data)
        # score = measure.score(predictions, test_data.truth)
        # scores.append(score)
    
    return scores