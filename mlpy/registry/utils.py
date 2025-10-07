"""
Utility functions for Model Registry.

This module provides utility functions for working with the model registry.
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from .base import ModelVersion, ModelStage


def generate_model_id(
    name: str,
    version: str,
    timestamp: Optional[datetime] = None
) -> str:
    """Generate a unique model ID.
    
    Parameters
    ----------
    name : str
        Model name.
    version : str
        Model version.
    timestamp : Optional[datetime]
        Timestamp to include. If None, uses current time.
        
    Returns
    -------
    str
        Unique model ID.
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    content = f"{name}_{version}_{timestamp.isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compare_models(
    models: List[ModelVersion],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compare multiple models.
    
    Parameters
    ----------
    models : List[ModelVersion]
        List of models to compare.
    metrics : Optional[List[str]]
        Specific metrics to compare. If None, uses all available.
        
    Returns
    -------
    pd.DataFrame
        Comparison table of models.
    """
    comparison_data = []
    
    for model in models:
        row = {
            'name': model.metadata.name,
            'version': model.metadata.version,
            'stage': model.metadata.stage.value,
            'created_at': model.metadata.created_at,
            'author': model.metadata.author
        }
        
        # Add metrics
        if metrics:
            for metric in metrics:
                row[metric] = model.metadata.metrics.get(metric, None)
        else:
            row.update(model.metadata.metrics)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def export_model_metadata(
    model: ModelVersion,
    format: str = 'json'
) -> str:
    """Export model metadata to string.
    
    Parameters
    ----------
    model : ModelVersion
        Model to export metadata from.
    format : str
        Export format ('json', 'yaml', 'markdown').
        
    Returns
    -------
    str
        Exported metadata string.
    """
    metadata_dict = model.metadata.to_dict()
    metadata_dict['model_id'] = model.model_id
    metadata_dict['parent_version'] = model.parent_version
    
    if format == 'json':
        return json.dumps(metadata_dict, indent=2)
    
    elif format == 'yaml':
        try:
            import yaml
            return yaml.dump(metadata_dict, default_flow_style=False)
        except ImportError:
            raise ImportError("yaml package required for YAML export")
    
    elif format == 'markdown':
        md_lines = [
            f"# Model: {model.metadata.name}",
            f"## Version: {model.metadata.version}",
            "",
            f"**ID:** {model.model_id}",
            f"**Stage:** {model.metadata.stage.value}",
            f"**Author:** {model.metadata.author}",
            f"**Created:** {model.metadata.created_at}",
            f"**Updated:** {model.metadata.updated_at}",
            "",
            "### Description",
            model.metadata.description or "No description provided.",
            "",
            "### Metrics",
        ]
        
        if model.metadata.metrics:
            for metric, value in model.metadata.metrics.items():
                md_lines.append(f"- **{metric}:** {value:.4f}")
        else:
            md_lines.append("No metrics recorded.")
        
        md_lines.extend([
            "",
            "### Parameters",
        ])
        
        if model.metadata.parameters:
            for param, value in model.metadata.parameters.items():
                md_lines.append(f"- **{param}:** {value}")
        else:
            md_lines.append("No parameters recorded.")
        
        md_lines.extend([
            "",
            "### Tags",
        ])
        
        if model.metadata.tags:
            for tag, value in model.metadata.tags.items():
                md_lines.append(f"- **{tag}:** {value}")
        else:
            md_lines.append("No tags.")
        
        return "\n".join(md_lines)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def find_best_model(
    models: List[ModelVersion],
    metric: str,
    higher_is_better: bool = True
) -> Optional[ModelVersion]:
    """Find the best model based on a metric.
    
    Parameters
    ----------
    models : List[ModelVersion]
        List of models to search.
    metric : str
        Metric to optimize.
    higher_is_better : bool
        Whether higher values are better.
        
    Returns
    -------
    Optional[ModelVersion]
        Best model if found, None otherwise.
    """
    if not models:
        return None
    
    # Filter models that have the metric
    models_with_metric = [
        m for m in models
        if metric in m.metadata.metrics
    ]
    
    if not models_with_metric:
        return None
    
    # Find best
    if higher_is_better:
        return max(models_with_metric, key=lambda m: m.metadata.metrics[metric])
    else:
        return min(models_with_metric, key=lambda m: m.metadata.metrics[metric])


def validate_model_name(name: str) -> bool:
    """Validate a model name.
    
    Parameters
    ----------
    name : str
        Model name to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Check if name is not empty
    if not name or not name.strip():
        return False
    
    # Check for invalid characters
    invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in name for char in invalid_chars):
        return False
    
    # Check length
    if len(name) > 255:
        return False
    
    return True


def validate_version_string(version: str) -> bool:
    """Validate a version string.
    
    Parameters
    ----------
    version : str
        Version string to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Check if version is not empty
    if not version or not version.strip():
        return False
    
    # Simple semantic versioning check (e.g., 1.0.0)
    parts = version.split('.')
    if len(parts) == 3:
        try:
            for part in parts:
                int(part)
            return True
        except ValueError:
            pass
    
    # Allow date-based versions (e.g., 20231225.120000)
    if '.' in version and len(version) >= 15:
        parts = version.split('.')
        if len(parts) == 2:
            try:
                # Check if it looks like a date
                datetime.strptime(parts[0], '%Y%m%d')
                return True
            except ValueError:
                pass
    
    # Allow simple numeric or alphanumeric versions
    return version.replace('.', '').replace('-', '').replace('_', '').isalnum()


def create_model_report(
    registry,
    name: str,
    version: Optional[str] = None
) -> Dict[str, Any]:
    """Create a detailed report for a model.
    
    Parameters
    ----------
    registry : ModelRegistry
        The registry to query.
    name : str
        Model name.
    version : Optional[str]
        Specific version. If None, reports on all versions.
        
    Returns
    -------
    Dict[str, Any]
        Detailed model report.
    """
    report = {
        'model_name': name,
        'report_time': datetime.now().isoformat(),
        'versions': []
    }
    
    if version:
        model = registry.get_model(name, version)
        if model:
            report['versions'].append({
                'version': model.metadata.version,
                'id': model.model_id,
                'stage': model.metadata.stage.value,
                'created_at': model.metadata.created_at.isoformat(),
                'author': model.metadata.author,
                'metrics': model.metadata.metrics,
                'parameters': model.metadata.parameters,
                'tags': model.metadata.tags,
                'description': model.metadata.description
            })
    else:
        # Report on all versions
        versions = registry.list_versions(name)
        for ver in versions:
            model = registry.get_model(name, ver)
            if model:
                report['versions'].append({
                    'version': model.metadata.version,
                    'id': model.model_id,
                    'stage': model.metadata.stage.value,
                    'created_at': model.metadata.created_at.isoformat(),
                    'author': model.metadata.author,
                    'metrics': model.metadata.metrics,
                    'parameters': model.metadata.parameters,
                    'tags': model.metadata.tags,
                    'description': model.metadata.description
                })
    
    # Add summary
    if report['versions']:
        report['summary'] = {
            'total_versions': len(report['versions']),
            'latest_version': max(report['versions'], key=lambda x: x['created_at'])['version'],
            'production_version': next(
                (v['version'] for v in report['versions'] if v['stage'] == 'production'),
                None
            )
        }
    
    return report