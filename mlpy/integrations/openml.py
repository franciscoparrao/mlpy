"""
Integración con OpenML para MLPY.

OpenML es una plataforma colaborativa para compartir datasets,
tareas y resultados de machine learning.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import tempfile
import warnings

from ..tasks import TaskClassif, TaskRegr, Task
from ..learners.base import Learner
from ..benchmark import Benchmark
from ..resample import Resample
from ..measures import Measure

logger = logging.getLogger(__name__)


@dataclass
class OpenMLConfig:
    """Configuración para OpenML.
    
    Attributes
    ----------
    api_key : Optional[str]
        API key para OpenML (requerido para uploads).
    server : str
        Servidor de OpenML.
    cachedir : Optional[Path]
        Directorio para caché de datasets.
    verbosity : int
        Nivel de verbosidad (0-2).
    retry_policy : int
        Número de reintentos para requests.
    avoid_duplicate_runs : bool
        Si evitar runs duplicados.
    """
    api_key: Optional[str] = None
    server: str = "https://www.openml.org/api/v1"
    cachedir: Optional[Path] = None
    verbosity: int = 0
    retry_policy: int = 3
    avoid_duplicate_runs: bool = True


class OpenMLClient:
    """Cliente para interactuar con OpenML."""
    
    def __init__(self, config: Optional[OpenMLConfig] = None):
        """Inicializa el cliente de OpenML.
        
        Parameters
        ----------
        config : Optional[OpenMLConfig]
            Configuración del cliente.
        """
        self.config = config or OpenMLConfig()
        self._setup()
    
    def _setup(self):
        """Configura OpenML."""
        try:
            import openml
            self.openml = openml
            
            # Configurar OpenML
            if self.config.api_key:
                openml.config.apikey = self.config.api_key
            
            if self.config.server:
                openml.config.server = self.config.server
            
            if self.config.cachedir:
                openml.config.cache_directory = str(self.config.cachedir)
            
            openml.config.retry_policy = self.config.retry_policy
            openml.config.avoid_duplicate_runs = self.config.avoid_duplicate_runs
            
            # Configurar verbosidad
            if self.config.verbosity == 0:
                warnings.filterwarnings('ignore')
            
            logger.info("OpenML client initialized")
            
        except ImportError:
            raise ImportError("OpenML not installed. Install with: pip install openml")
    
    def download_dataset(
        self,
        dataset_id: Optional[int] = None,
        name: Optional[str] = None,
        version: Optional[int] = None,
        data_format: str = 'dataframe'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Descarga un dataset de OpenML.
        
        Parameters
        ----------
        dataset_id : Optional[int]
            ID del dataset en OpenML.
        name : Optional[str]
            Nombre del dataset.
        version : Optional[int]
            Versión del dataset.
        data_format : str
            Formato de datos ('dataframe' o 'array').
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, Any]]
            Datos y metadata del dataset.
        """
        try:
            # Obtener dataset
            if dataset_id:
                dataset = self.openml.datasets.get_dataset(
                    dataset_id,
                    download_data=True,
                    download_qualities=True,
                    download_features_meta_data=True
                )
            elif name:
                datasets = self.openml.datasets.list_datasets(
                    output_format='dataframe'
                )
                matches = datasets[datasets['name'] == name]
                
                if version:
                    matches = matches[matches['version'] == version]
                
                if len(matches) == 0:
                    raise ValueError(f"Dataset '{name}' not found")
                
                dataset_id = matches.iloc[0]['did']
                dataset = self.openml.datasets.get_dataset(dataset_id)
            else:
                raise ValueError("Either dataset_id or name must be provided")
            
            # Obtener datos
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format=data_format
            )
            
            # Crear DataFrame si es necesario
            if data_format == 'array' and isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=attribute_names)
            
            # Combinar X e y
            if y is not None:
                if isinstance(y, pd.Series):
                    data = pd.concat([X, y.to_frame('target')], axis=1)
                else:
                    data = pd.concat([X, pd.Series(y, name='target').to_frame()], axis=1)
            else:
                data = X
            
            # Metadata
            metadata = {
                'dataset_id': dataset.dataset_id,
                'name': dataset.name,
                'version': dataset.version,
                'description': dataset.description,
                'format': dataset.format,
                'url': dataset.url,
                'default_target': dataset.default_target_attribute,
                'categorical_features': [
                    attr for attr, is_cat in zip(attribute_names, categorical_indicator) 
                    if is_cat
                ],
                'qualities': dataset.qualities
            }
            
            logger.info(f"Downloaded dataset: {dataset.name} (ID: {dataset.dataset_id})")
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def download_task(
        self,
        task_id: int
    ) -> Tuple[Task, Dict[str, Any]]:
        """Descarga una tarea de OpenML y la convierte a Task de MLPY.
        
        Parameters
        ----------
        task_id : int
            ID de la tarea en OpenML.
            
        Returns
        -------
        Tuple[Task, Dict[str, Any]]
            Tarea de MLPY y metadata.
        """
        try:
            # Obtener tarea
            openml_task = self.openml.tasks.get_task(task_id)
            
            # Obtener dataset
            dataset = openml_task.get_dataset()
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=openml_task.target_name,
                dataset_format='dataframe'
            )
            
            # Combinar datos
            if isinstance(y, pd.Series):
                data = pd.concat([X, y.to_frame(openml_task.target_name)], axis=1)
            else:
                data = pd.concat([X, pd.Series(y, name=openml_task.target_name).to_frame()], axis=1)
            
            # Determinar tipo de tarea
            task_type = openml_task.task_type
            
            if task_type in ['Supervised Classification', 'Learning Curve']:
                # Tarea de clasificación
                mlpy_task = TaskClassif(
                    id=f"openml_task_{task_id}",
                    data=data,
                    target_col=openml_task.target_name
                )
            elif task_type in ['Supervised Regression']:
                # Tarea de regresión
                mlpy_task = TaskRegr(
                    id=f"openml_task_{task_id}",
                    data=data,
                    target_col=openml_task.target_name
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Metadata
            metadata = {
                'task_id': openml_task.task_id,
                'task_type': task_type,
                'dataset_id': dataset.dataset_id,
                'dataset_name': dataset.name,
                'target_name': openml_task.target_name,
                'estimation_procedure': openml_task.estimation_procedure,
                'evaluation_measure': openml_task.evaluation_measure,
                'cost_matrix': openml_task.cost_matrix
            }
            
            # Añadir splits si existen
            if hasattr(openml_task, 'get_train_test_split_indices'):
                try:
                    train_idx, test_idx = openml_task.get_train_test_split_indices()
                    metadata['train_indices'] = train_idx
                    metadata['test_indices'] = test_idx
                except:
                    pass
            
            logger.info(f"Downloaded task: {task_id} ({task_type})")
            
            return mlpy_task, metadata
            
        except Exception as e:
            logger.error(f"Error downloading task: {e}")
            raise
    
    def upload_run(
        self,
        task_id: int,
        learner: Learner,
        predictions: np.ndarray,
        runtime: Optional[float] = None,
        setup_string: Optional[str] = None,
        parameter_settings: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        upload_model: bool = False
    ) -> int:
        """Sube un run a OpenML.
        
        Parameters
        ----------
        task_id : int
            ID de la tarea en OpenML.
        learner : Learner
            Learner usado.
        predictions : np.ndarray
            Predicciones del modelo.
        runtime : Optional[float]
            Tiempo de ejecución en segundos.
        setup_string : Optional[str]
            Descripción del setup.
        parameter_settings : Optional[Dict[str, Any]]
            Parámetros del modelo.
        tags : Optional[List[str]]
            Tags para el run.
        upload_model : bool
            Si subir el modelo serializado.
            
        Returns
        -------
        int
            ID del run en OpenML.
        """
        if not self.config.api_key:
            raise ValueError("API key required for uploading runs")
        
        try:
            # Obtener tarea
            task = self.openml.tasks.get_task(task_id)
            
            # Crear run
            # Nota: Esto es una simplificación, OpenML requiere un formato específico
            # para los runs dependiendo del tipo de tarea
            
            # Para clasificación
            if task.task_type in ['Supervised Classification']:
                # Crear objeto de run
                from openml.runs import Run
                
                # Necesitamos convertir el learner a un modelo compatible con OpenML
                # Por ahora, usamos un wrapper simple
                model_description = setup_string or f"MLPY_{learner.__class__.__name__}"
                
                # Crear flow (descripción del modelo)
                flow = self.openml.flows.create_flow(
                    name=model_description,
                    description=f"MLPY learner: {learner.__class__.__name__}",
                    model=learner,  # Esto puede necesitar un wrapper
                    components=None,
                    parameters=parameter_settings or {},
                    tags=tags
                )
                
                # Publicar flow si no existe
                try:
                    flow_id = self.openml.flows.flow_exists(flow)
                    if not flow_id:
                        flow = flow.publish()
                        flow_id = flow.flow_id
                except:
                    flow = flow.publish()
                    flow_id = flow.flow_id
                
                # Crear y publicar run
                run = Run(
                    task_id=task_id,
                    flow_id=flow_id,
                    dataset_id=task.dataset_id,
                    parameter_settings=parameter_settings,
                    tags=tags
                )
                
                # Añadir predicciones y métricas
                run.data_content = predictions
                
                if runtime:
                    run.runtime = runtime
                
                # Publicar run
                run = run.publish()
                
                logger.info(f"Uploaded run: {run.run_id} for task {task_id}")
                
                return run.run_id
                
        except Exception as e:
            logger.error(f"Error uploading run: {e}")
            raise
    
    def list_datasets(
        self,
        tag: Optional[str] = None,
        output_format: str = 'dataframe',
        **kwargs
    ) -> Union[pd.DataFrame, Dict]:
        """Lista datasets disponibles en OpenML.
        
        Parameters
        ----------
        tag : Optional[str]
            Tag para filtrar datasets.
        output_format : str
            Formato de salida ('dataframe' o 'dict').
        **kwargs
            Filtros adicionales.
            
        Returns
        -------
        Union[pd.DataFrame, Dict]
            Lista de datasets.
        """
        try:
            datasets = self.openml.datasets.list_datasets(
                tag=tag,
                output_format=output_format,
                **kwargs
            )
            
            if output_format == 'dataframe' and not datasets.empty:
                # Ordenar por popularidad (número de runs)
                if 'NumberOfInstances' in datasets.columns:
                    datasets = datasets.sort_values('NumberOfInstances', ascending=False)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            raise
    
    def list_tasks(
        self,
        task_type: Optional[str] = None,
        tag: Optional[str] = None,
        output_format: str = 'dataframe',
        **kwargs
    ) -> Union[pd.DataFrame, Dict]:
        """Lista tareas disponibles en OpenML.
        
        Parameters
        ----------
        task_type : Optional[str]
            Tipo de tarea para filtrar.
        tag : Optional[str]
            Tag para filtrar tareas.
        output_format : str
            Formato de salida ('dataframe' o 'dict').
        **kwargs
            Filtros adicionales.
            
        Returns
        -------
        Union[pd.DataFrame, Dict]
            Lista de tareas.
        """
        try:
            # Mapear tipos de tarea
            task_type_map = {
                'classification': 'Supervised Classification',
                'regression': 'Supervised Regression',
                'clustering': 'Clustering',
                'learning_curve': 'Learning Curve'
            }
            
            if task_type and task_type.lower() in task_type_map:
                task_type = task_type_map[task_type.lower()]
            
            tasks = self.openml.tasks.list_tasks(
                task_type=task_type,
                tag=tag,
                output_format=output_format,
                **kwargs
            )
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            raise
    
    def get_benchmark_suite(
        self,
        suite_id: Optional[int] = None,
        suite_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Obtiene una suite de benchmark de OpenML.
        
        Parameters
        ----------
        suite_id : Optional[int]
            ID de la suite.
        suite_name : Optional[str]
            Nombre de la suite.
            
        Returns
        -------
        Dict[str, Any]
            Información de la suite.
        """
        try:
            if suite_id:
                suite = self.openml.study.get_suite(suite_id)
            elif suite_name:
                # Buscar por nombre
                suites = self.openml.study.list_suites(output_format='dataframe')
                matches = suites[suites['name'] == suite_name]
                
                if len(matches) == 0:
                    raise ValueError(f"Suite '{suite_name}' not found")
                
                suite_id = matches.iloc[0]['id']
                suite = self.openml.study.get_suite(suite_id)
            else:
                raise ValueError("Either suite_id or suite_name must be provided")
            
            # Obtener información de la suite
            suite_info = {
                'id': suite.suite_id,
                'alias': suite.alias,
                'name': suite.name,
                'description': suite.description,
                'task_ids': suite.tasks,
                'n_tasks': len(suite.tasks)
            }
            
            logger.info(f"Retrieved suite: {suite.name} with {len(suite.tasks)} tasks")
            
            return suite_info
            
        except Exception as e:
            logger.error(f"Error getting benchmark suite: {e}")
            raise
    
    def run_benchmark(
        self,
        learners: List[Learner],
        suite_id: Optional[int] = None,
        task_ids: Optional[List[int]] = None,
        resampling: Optional[Resample] = None,
        measures: Optional[List[Measure]] = None,
        upload_results: bool = False,
        parallel: bool = True,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """Ejecuta un benchmark en tareas de OpenML.
        
        Parameters
        ----------
        learners : List[Learner]
            Lista de learners a evaluar.
        suite_id : Optional[int]
            ID de la suite de benchmark.
        task_ids : Optional[List[int]]
            IDs de tareas específicas.
        resampling : Optional[Resample]
            Estrategia de resampling.
        measures : Optional[List[Measure]]
            Medidas de evaluación.
        upload_results : bool
            Si subir resultados a OpenML.
        parallel : bool
            Si ejecutar en paralelo.
        n_jobs : int
            Número de jobs paralelos.
            
        Returns
        -------
        pd.DataFrame
            Resultados del benchmark.
        """
        # Obtener tareas
        if suite_id:
            suite = self.openml.study.get_suite(suite_id)
            task_ids = suite.tasks
        elif not task_ids:
            raise ValueError("Either suite_id or task_ids must be provided")
        
        # Descargar tareas
        tasks = []
        for task_id in task_ids:
            try:
                task, metadata = self.download_task(task_id)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Could not download task {task_id}: {e}")
        
        if not tasks:
            raise ValueError("No tasks could be downloaded")
        
        # Crear benchmark de MLPY
        from ..benchmark import Benchmark
        
        benchmark = Benchmark(
            learners=learners,
            tasks=tasks,
            resamplings=[resampling] if resampling else None,
            measures=measures
        )
        
        # Ejecutar benchmark
        results = benchmark.run(parallel=parallel, n_jobs=n_jobs)
        
        # Subir resultados si se solicita
        if upload_results and self.config.api_key:
            for _, row in results.iterrows():
                try:
                    # Extraer información necesaria
                    task_id = int(row['task'].split('_')[-1])  # Asumiendo formato openml_task_ID
                    learner_idx = row['learner']
                    
                    # Obtener predicciones (simplificado)
                    # En práctica, necesitaríamos guardar las predicciones durante el benchmark
                    
                    self.upload_run(
                        task_id=task_id,
                        learner=learners[learner_idx],
                        predictions=np.array([]),  # Placeholder
                        runtime=row.get('runtime'),
                        parameter_settings={}
                    )
                except Exception as e:
                    logger.warning(f"Could not upload run: {e}")
        
        return results


# Funciones de conveniencia
def download_dataset(
    dataset_id: Optional[int] = None,
    name: Optional[str] = None,
    version: Optional[int] = None,
    as_task: bool = True,
    target_col: Optional[str] = None,
    task_type: str = 'classification'
) -> Union[Task, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Descarga un dataset de OpenML.
    
    Parameters
    ----------
    dataset_id : Optional[int]
        ID del dataset.
    name : Optional[str]
        Nombre del dataset.
    version : Optional[int]
        Versión del dataset.
    as_task : bool
        Si retornar como Task de MLPY.
    target_col : Optional[str]
        Columna objetivo para la tarea.
    task_type : str
        Tipo de tarea ('classification' o 'regression').
        
    Returns
    -------
    Union[Task, Tuple[pd.DataFrame, Dict[str, Any]]]
        Task de MLPY o tupla (datos, metadata).
    """
    client = OpenMLClient()
    data, metadata = client.download_dataset(dataset_id, name, version)
    
    if as_task:
        # Determinar columna objetivo
        if not target_col:
            target_col = metadata.get('default_target', 'target')
        
        # Verificar que existe la columna
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Crear tarea según tipo
        if task_type == 'classification':
            return TaskClassif(
                id=f"openml_{metadata['name']}",
                data=data,
                target_col=target_col
            )
        else:
            return TaskRegr(
                id=f"openml_{metadata['name']}",
                data=data,
                target_col=target_col
            )
    
    return data, metadata


def download_task(task_id: int) -> Task:
    """Descarga una tarea de OpenML.
    
    Parameters
    ----------
    task_id : int
        ID de la tarea.
        
    Returns
    -------
    Task
        Tarea de MLPY.
    """
    client = OpenMLClient()
    task, _ = client.download_task(task_id)
    return task


def upload_run(
    task_id: int,
    learner: Learner,
    predictions: np.ndarray,
    api_key: str,
    **kwargs
) -> int:
    """Sube un run a OpenML.
    
    Parameters
    ----------
    task_id : int
        ID de la tarea.
    learner : Learner
        Learner usado.
    predictions : np.ndarray
        Predicciones.
    api_key : str
        API key de OpenML.
    **kwargs
        Argumentos adicionales.
        
    Returns
    -------
    int
        ID del run.
    """
    config = OpenMLConfig(api_key=api_key)
    client = OpenMLClient(config)
    return client.upload_run(task_id, learner, predictions, **kwargs)


def list_datasets(
    tag: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Lista datasets de OpenML.
    
    Parameters
    ----------
    tag : Optional[str]
        Tag para filtrar.
    **kwargs
        Filtros adicionales.
        
    Returns
    -------
    pd.DataFrame
        Lista de datasets.
    """
    client = OpenMLClient()
    return client.list_datasets(tag=tag, **kwargs)


def list_tasks(
    task_type: Optional[str] = None,
    tag: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """Lista tareas de OpenML.
    
    Parameters
    ----------
    task_type : Optional[str]
        Tipo de tarea.
    tag : Optional[str]
        Tag para filtrar.
    **kwargs
        Filtros adicionales.
        
    Returns
    -------
    pd.DataFrame
        Lista de tareas.
    """
    client = OpenMLClient()
    return client.list_tasks(task_type=task_type, tag=tag, **kwargs)


def get_benchmark_suite(
    suite_id: Optional[int] = None,
    suite_name: Optional[str] = None
) -> Dict[str, Any]:
    """Obtiene una suite de benchmark.
    
    Parameters
    ----------
    suite_id : Optional[int]
        ID de la suite.
    suite_name : Optional[str]
        Nombre de la suite.
        
    Returns
    -------
    Dict[str, Any]
        Información de la suite.
    """
    client = OpenMLClient()
    return client.get_benchmark_suite(suite_id, suite_name)


def run_benchmark(
    learners: List[Learner],
    suite_id: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
    **kwargs
) -> pd.DataFrame:
    """Ejecuta un benchmark en OpenML.
    
    Parameters
    ----------
    learners : List[Learner]
        Learners a evaluar.
    suite_id : Optional[int]
        ID de la suite.
    task_ids : Optional[List[int]]
        IDs de tareas.
    **kwargs
        Argumentos adicionales.
        
    Returns
    -------
    pd.DataFrame
        Resultados del benchmark.
    """
    client = OpenMLClient()
    return client.run_benchmark(learners, suite_id, task_ids, **kwargs)