"""
Módulo de integraciones externas para MLPY.

Integración con plataformas y servicios externos como OpenML, 
cloud providers, y otras herramientas del ecosistema ML.
"""

from .openml import (
    OpenMLClient,
    download_dataset,
    download_task,
    upload_run,
    list_datasets,
    list_tasks,
    get_benchmark_suite,
    run_benchmark
)

__all__ = [
    # OpenML
    'OpenMLClient',
    'download_dataset',
    'download_task', 
    'upload_run',
    'list_datasets',
    'list_tasks',
    'get_benchmark_suite',
    'run_benchmark'
]