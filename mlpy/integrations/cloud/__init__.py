"""
Integración con proveedores de cloud para MLPY.

Soporte para AWS, Google Cloud Platform y Microsoft Azure.
"""

from .base import CloudProvider, CloudStorage, CloudCompute
from .aws import AWSProvider, S3Storage, SageMakerCompute
from .gcp import GCPProvider, GCSStorage, VertexAICompute
from .azure import AzureProvider, BlobStorage, AzureMLCompute

__all__ = [
    # Base
    'CloudProvider',
    'CloudStorage',
    'CloudCompute',
    
    # AWS
    'AWSProvider',
    'S3Storage',
    'SageMakerCompute',
    
    # GCP
    'GCPProvider',
    'GCSStorage',
    'VertexAICompute',
    
    # Azure
    'AzureProvider',
    'BlobStorage',
    'AzureMLCompute'
]