"""
Spatial machine learning tasks for MLPY.

These tasks extend the base supervised tasks with spatial awareness,
enabling spatial cross-validation and coordinate-based operations.
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd

from .supervised import TaskClassif, TaskRegr
from mlpy.backends.base import DataBackend
from mlpy.backends.pandas_backend import DataBackendPandas
from mlpy.utils.registry import mlpy_tasks


class TaskSpatial(ABC):
    """
    Mixin class for spatial tasks.
    
    Adds spatial awareness to supervised tasks, including:
    - Coordinate columns management
    - CRS (Coordinate Reference System) handling
    - Spatial metadata
    - Support for spatial resampling methods
    """
    
    def __init__(
        self,
        coordinate_names: Optional[List[str]] = None,
        crs: Optional[Union[str, int]] = None,
        coords_as_features: bool = False,
        spatial_extent: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Initialize spatial task components.
        
        Parameters
        ----------
        coordinate_names : List[str], optional
            Names of coordinate columns (e.g., ['x', 'y'] or ['lon', 'lat'])
        crs : str or int, optional
            Coordinate Reference System (e.g., 'EPSG:4326' or 4326)
        coords_as_features : bool, default=False
            Whether to use coordinates as features in modeling
        spatial_extent : dict, optional
            Spatial bounding box {'xmin': , 'xmax': , 'ymin': , 'ymax': }
        """
        self._coordinate_names = coordinate_names or []
        self._crs = crs
        self._coords_as_features = coords_as_features
        self._spatial_extent = spatial_extent
        
        # Store original kwargs for parent class
        self._spatial_kwargs = kwargs
    
    def _setup_spatial_roles(self):
        """Setup column roles for spatial data."""
        if self._coordinate_names:
            # Add coordinates to column roles
            if "coordinate" not in self._col_roles:
                self._col_roles["coordinate"] = set()
            
            # Add coordinate columns to the coordinate role
            for coord in self._coordinate_names:
                if coord in self._backend.colnames:
                    self._col_roles["coordinate"].add(coord)
                    
                    # Remove from features unless coords_as_features is True
                    if not self._coords_as_features and coord in self._col_roles["feature"]:
                        self._col_roles["feature"].discard(coord)
                else:
                    warnings.warn(f"Coordinate column '{coord}' not found in data")
    
    @property
    def coordinate_names(self) -> List[str]:
        """Names of coordinate columns."""
        return self._coordinate_names
    
    @property
    def crs(self) -> Optional[Union[str, int]]:
        """Coordinate Reference System."""
        return self._crs
    
    @property
    def coords_as_features(self) -> bool:
        """Whether coordinates are used as features."""
        return self._coords_as_features
    
    @property
    def spatial_extent(self) -> Optional[Dict[str, float]]:
        """Spatial bounding box of the data."""
        if self._spatial_extent is None and self._coordinate_names:
            # Calculate extent from data
            coords = self.coordinates()
            if len(coords) > 0:
                self._spatial_extent = {
                    'xmin': coords[:, 0].min(),
                    'xmax': coords[:, 0].max(),
                    'ymin': coords[:, 1].min(),
                    'ymax': coords[:, 1].max()
                }
        return self._spatial_extent
    
    def coordinates(self, rows: Optional[List[int]] = None) -> np.ndarray:
        """
        Get coordinate values.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        np.ndarray
            Coordinate array of shape (n_samples, n_coords)
        """
        if not self._coordinate_names:
            raise ValueError("No coordinate columns specified")
        
        return self.data(
            rows=rows,
            cols=self._coordinate_names,
            data_format="array"
        )
    
    def distance_matrix(self, rows: Optional[List[int]] = None) -> np.ndarray:
        """
        Calculate pairwise distance matrix between points.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        np.ndarray
            Distance matrix of shape (n_samples, n_samples)
        """
        coords = self.coordinates(rows)
        
        # Calculate Euclidean distances (can be extended for geographic distances)
        from scipy.spatial.distance import cdist
        return cdist(coords, coords, metric='euclidean')
    
    def spatial_neighbors(
        self,
        n_neighbors: int = 5,
        max_distance: Optional[float] = None,
        rows: Optional[List[int]] = None
    ) -> Dict[int, List[int]]:
        """
        Find spatial neighbors for each point.
        
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of nearest neighbors
        max_distance : float, optional
            Maximum distance for neighbors
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        dict
            Mapping from row index to list of neighbor indices
        """
        coords = self.coordinates(rows)
        
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
        nn.fit(coords)
        
        distances, indices = nn.kneighbors(coords)
        
        neighbors = {}
        for i in range(len(coords)):
            # Exclude self (first neighbor)
            neighbor_idx = indices[i, 1:].tolist()
            
            # Filter by max_distance if specified
            if max_distance is not None:
                neighbor_dist = distances[i, 1:]
                neighbor_idx = [
                    idx for idx, dist in zip(neighbor_idx, neighbor_dist)
                    if dist <= max_distance
                ]
            
            neighbors[i] = neighbor_idx
        
        return neighbors
    
    def to_geopandas(self, rows: Optional[List[int]] = None):
        """
        Convert task data to GeoPandas DataFrame.
        
        Parameters
        ----------
        rows : List[int], optional
            Row indices. If None, all rows in use.
            
        Returns
        -------
        gpd.GeoDataFrame
            GeoPandas DataFrame with geometry column
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point
        except ImportError:
            raise ImportError(
                "GeoPandas is required for spatial operations. "
                "Install it with: pip install geopandas"
            )
        
        # Get all data including coordinates
        all_cols = self.feature_names + self.target_names + self._coordinate_names
        df = self.data(rows=rows, cols=all_cols, data_format="dataframe")
        
        # Create geometry from coordinates
        if len(self._coordinate_names) >= 2:
            coord_x = self._coordinate_names[0]
            coord_y = self._coordinate_names[1]
            geometry = [
                Point(row[coord_x], row[coord_y])
                for _, row in df.iterrows()
            ]
        else:
            raise ValueError("Need at least 2 coordinate columns for geometry")
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # Set CRS if available
        if self._crs:
            if isinstance(self._crs, int):
                gdf.crs = f"EPSG:{self._crs}"
            else:
                gdf.crs = self._crs
        
        return gdf


@mlpy_tasks.register("classif_spatial")
class TaskClassifSpatial(TaskClassif, TaskSpatial):
    """
    Spatial classification task.
    
    Extends TaskClassif with spatial awareness for geographic data.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    target : str
        Name of the target column (must be categorical)
    coordinate_names : List[str], optional
        Names of coordinate columns (e.g., ['x', 'y'])
    crs : str or int, optional
        Coordinate Reference System (e.g., 'EPSG:4326' or 4326)
    coords_as_features : bool, default=False
        Whether to use coordinates as features
    positive : str, optional
        Name of the positive class for binary classification
    id : str, optional
        Task identifier
    label : str, optional
        Task label
        
    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks.spatial import TaskClassifSpatial
    >>> 
    >>> # Create sample spatial data
    >>> df = pd.DataFrame({
    ...     'x': [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     'y': [1.0, 2.0, 3.0, 4.0, 5.0],
    ...     'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    ...     'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    ...     'landslide': [0, 0, 1, 1, 0]
    ... })
    >>> 
    >>> # Create spatial classification task
    >>> task = TaskClassifSpatial(
    ...     data=df,
    ...     target='landslide',
    ...     coordinate_names=['x', 'y'],
    ...     crs='EPSG:4326',
    ...     coords_as_features=False
    ... )
    >>> 
    >>> # Access spatial properties
    >>> print(task.coordinate_names)
    ['x', 'y']
    >>> print(task.spatial_extent)
    {'xmin': 1.0, 'xmax': 5.0, 'ymin': 1.0, 'ymax': 5.0}
    >>> 
    >>> # Get coordinates
    >>> coords = task.coordinates()
    >>> print(coords.shape)
    (5, 2)
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        target: str,
        coordinate_names: Optional[List[str]] = None,
        crs: Optional[Union[str, int]] = None,
        coords_as_features: bool = False,
        positive: Optional[str] = None,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        # Initialize spatial components
        TaskSpatial.__init__(
            self,
            coordinate_names=coordinate_names,
            crs=crs,
            coords_as_features=coords_as_features,
            **kwargs
        )
        
        # Initialize classification task
        TaskClassif.__init__(
            self,
            data=data,
            target=target,
            positive=positive,
            id=id or "spatial_classif",
            label=label,
            **self._spatial_kwargs
        )
        
        # Setup spatial column roles
        self._setup_spatial_roles()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "classif_spatial"
    
    def __repr__(self) -> str:
        return (
            f"<TaskClassifSpatial({self.nrow} x {self.ncol})"
            f" [coords: {self.coordinate_names}]"
            f" [crs: {self.crs}]>"
        )


@mlpy_tasks.register("regr_spatial") 
class TaskRegrSpatial(TaskRegr, TaskSpatial):
    """
    Spatial regression task.
    
    Extends TaskRegr with spatial awareness for geographic data.
    
    Parameters
    ----------
    data : pd.DataFrame, DataBackend, or dict
        The data for the task
    target : str or List[str]
        Name(s) of the target column(s) (must be numeric)
    coordinate_names : List[str], optional
        Names of coordinate columns (e.g., ['x', 'y'])
    crs : str or int, optional
        Coordinate Reference System (e.g., 'EPSG:4326' or 4326)
    coords_as_features : bool, default=False
        Whether to use coordinates as features
    id : str, optional
        Task identifier
    label : str, optional
        Task label
        
    Examples
    --------
    >>> import pandas as pd
    >>> from mlpy.tasks.spatial import TaskRegrSpatial
    >>> 
    >>> # Create sample spatial data
    >>> df = pd.DataFrame({
    ...     'lon': [-70.1, -70.2, -70.3, -70.4],
    ...     'lat': [-23.1, -23.2, -23.3, -23.4],
    ...     'elevation': [100, 200, 150, 250],
    ...     'slope': [10, 20, 15, 25],
    ...     'erosion_rate': [0.5, 1.2, 0.8, 1.5]
    ... })
    >>> 
    >>> # Create spatial regression task
    >>> task = TaskRegrSpatial(
    ...     data=df,
    ...     target='erosion_rate',
    ...     coordinate_names=['lon', 'lat'],
    ...     crs='EPSG:4326'
    ... )
    >>> 
    >>> # Calculate distance matrix
    >>> distances = task.distance_matrix()
    >>> print(distances.shape)
    (4, 4)
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, DataBackend, Dict[str, Any]],
        target: Union[str, List[str]],
        coordinate_names: Optional[List[str]] = None,
        crs: Optional[Union[str, int]] = None,
        coords_as_features: bool = False,
        id: Optional[str] = None,
        label: Optional[str] = None,
        **kwargs
    ):
        # Initialize spatial components
        TaskSpatial.__init__(
            self,
            coordinate_names=coordinate_names,
            crs=crs,
            coords_as_features=coords_as_features,
            **kwargs
        )
        
        # Initialize regression task
        TaskRegr.__init__(
            self,
            data=data,
            target=target,
            id=id or "spatial_regr",
            label=label,
            **self._spatial_kwargs
        )
        
        # Setup spatial column roles
        self._setup_spatial_roles()
    
    @property
    def task_type(self) -> str:
        """Task type identifier."""
        return "regr_spatial"
    
    def __repr__(self) -> str:
        return (
            f"<TaskRegrSpatial({self.nrow} x {self.ncol})"
            f" [coords: {self.coordinate_names}]"
            f" [crs: {self.crs}]>"
        )


# Convenience functions
def create_spatial_task(
    data: Union[pd.DataFrame, Dict[str, Any]],
    target: Union[str, List[str]],
    task_type: str = "auto",
    coordinate_names: Optional[List[str]] = None,
    crs: Optional[Union[str, int]] = None,
    coords_as_features: bool = False,
    **kwargs
) -> Union[TaskClassifSpatial, TaskRegrSpatial]:
    """
    Create a spatial task automatically detecting the type.
    
    Parameters
    ----------
    data : pd.DataFrame or dict
        The data
    target : str or List[str]
        Target column(s)
    task_type : str, default="auto"
        Task type ("classif", "regr", or "auto" for automatic detection)
    coordinate_names : List[str], optional
        Coordinate column names
    crs : str or int, optional
        Coordinate Reference System
    coords_as_features : bool, default=False
        Use coordinates as features
    **kwargs
        Additional arguments passed to task constructor
        
    Returns
    -------
    TaskClassifSpatial or TaskRegrSpatial
        The appropriate spatial task
    """
    df = pd.DataFrame(data) if isinstance(data, dict) else data
    
    if task_type == "auto":
        # Auto-detect based on target type
        if isinstance(target, str):
            target_col = df[target]
        else:
            target_col = df[target[0]]
        
        # Check if categorical or numeric
        if pd.api.types.is_categorical_dtype(target_col) or \
           pd.api.types.is_object_dtype(target_col) or \
           target_col.nunique() < 10:  # Few unique values suggests classification
            task_type = "classif"
        else:
            task_type = "regr"
    
    if task_type == "classif":
        return TaskClassifSpatial(
            data=df,
            target=target,
            coordinate_names=coordinate_names,
            crs=crs,
            coords_as_features=coords_as_features,
            **kwargs
        )
    elif task_type == "regr":
        return TaskRegrSpatial(
            data=df,
            target=target,
            coordinate_names=coordinate_names,
            crs=crs,
            coords_as_features=coords_as_features,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")