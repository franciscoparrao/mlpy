"""Utilities for model persistence."""

import os
import json
import shutil
import tempfile
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from datetime import datetime
import hashlib

from .base import save_model, load_model, ModelBundle


class ModelRegistry:
    """Registry for managing saved models.
    
    This class provides a simple way to organize and track
    saved models with metadata and versioning.
    
    Parameters
    ----------
    root_dir : str or Path
        Root directory for the model registry.
    """
    
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root_dir / "registry.json"
        self._load_index()
        
    def _load_index(self):
        """Load the registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "models": {},
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            
    def _save_index(self):
        """Save the registry index."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
            
    def register_model(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        serializer: str = "auto"
    ) -> Path:
        """Register and save a model.
        
        Parameters
        ----------
        model : Any
            The model to save.
        name : str
            Name for the model.
        version : str, optional
            Version string. If None, uses timestamp.
        tags : list of str, optional
            Tags for categorizing the model.
        metadata : dict, optional
            Additional metadata to store.
        serializer : str, default="auto"
            Serializer to use.
            
        Returns
        -------
        Path
            Path where the model was saved.
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Create model directory
        model_dir = self.root_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension based on serializer
        if serializer == "auto":
            ext = ".pkl"
        elif serializer == "joblib":
            ext = ".joblib"
        elif serializer == "onnx":
            ext = ".onnx"
        else:
            ext = ".pkl"
            
        model_path = model_dir / f"model{ext}"
        
        # Prepare metadata
        full_metadata = {
            "name": name,
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "tags": tags or [],
            "serializer": serializer,
            "model_file": model_path.name
        }
        
        if metadata:
            full_metadata.update(metadata)
            
        # Save model
        save_model(model, model_path, serializer=serializer, metadata=full_metadata)
        
        # Update index
        if name not in self.index["models"]:
            self.index["models"][name] = {}
            
        self.index["models"][name][version] = {
            "path": str(model_path.relative_to(self.root_dir)),
            "metadata": full_metadata
        }
        
        self._save_index()
        
        return model_path
        
    def load_model(
        self,
        name: str,
        version: Optional[str] = None,
        return_metadata: bool = False
    ) -> Union[Any, tuple]:
        """Load a registered model.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : str, optional
            Version to load. If None, loads latest.
        return_metadata : bool, default=False
            Whether to return metadata along with model.
            
        Returns
        -------
        model or (model, metadata)
            The loaded model, optionally with metadata.
        """
        if name not in self.index["models"]:
            raise ValueError(f"Model '{name}' not found in registry")
            
        versions = self.index["models"][name]
        
        if version is None:
            # Get latest version
            version = max(versions.keys())
        elif version not in versions:
            available = list(versions.keys())
            raise ValueError(
                f"Version '{version}' not found for model '{name}'. "
                f"Available versions: {available}"
            )
            
        # Load model
        model_info = versions[version]
        model_path = self.root_dir / model_info["path"]
        
        if return_metadata:
            bundle = load_model(model_path, return_bundle=True)
            if isinstance(bundle, ModelBundle):
                return bundle.model, bundle.metadata
            else:
                return bundle, model_info["metadata"]
        else:
            return load_model(model_path)
            
    def list_models(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """List registered models.
        
        Parameters
        ----------
        name : str, optional
            Filter by model name.
            
        Returns
        -------
        dict
            Dictionary mapping model names to list of versions.
        """
        if name:
            if name in self.index["models"]:
                return {name: list(self.index["models"][name].keys())}
            else:
                return {}
        else:
            return {
                name: list(versions.keys())
                for name, versions in self.index["models"].items()
            }
            
    def get_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Get metadata for a model.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : str, optional
            Version. If None, gets latest.
            
        Returns
        -------
        dict
            Model metadata.
        """
        if name not in self.index["models"]:
            raise ValueError(f"Model '{name}' not found")
            
        versions = self.index["models"][name]
        
        if version is None:
            version = max(versions.keys())
        elif version not in versions:
            raise ValueError(f"Version '{version}' not found for model '{name}'")
            
        return versions[version]["metadata"]
        
    def delete_model(self, name: str, version: Optional[str] = None):
        """Delete a model from the registry.
        
        Parameters
        ----------
        name : str
            Name of the model.
        version : str, optional
            Version to delete. If None, deletes all versions.
        """
        if name not in self.index["models"]:
            raise ValueError(f"Model '{name}' not found")
            
        if version:
            # Delete specific version
            if version not in self.index["models"][name]:
                raise ValueError(f"Version '{version}' not found for model '{name}'")
                
            model_info = self.index["models"][name][version]
            model_path = self.root_dir / model_info["path"]
            
            # Delete model directory
            if model_path.parent.exists():
                shutil.rmtree(model_path.parent)
                
            # Update index
            del self.index["models"][name][version]
            
            # Remove model entry if no versions left
            if not self.index["models"][name]:
                del self.index["models"][name]
        else:
            # Delete all versions
            model_dir = self.root_dir / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                
            del self.index["models"][name]
            
        self._save_index()


def export_model_package(
    model: Any,
    output_path: Union[str, Path],
    name: str,
    include_dependencies: bool = True,
    include_examples: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """Export a model as a self-contained package.
    
    This creates a ZIP file containing the model, metadata,
    dependencies list, and optionally example code.
    
    Parameters
    ----------
    model : Any
        The model to export.
    output_path : str or Path
        Where to save the package (ZIP file).
    name : str
        Name for the model.
    include_dependencies : bool, default=True
        Whether to include requirements.txt.
    include_examples : bool, default=True
        Whether to include example usage code.
    metadata : dict, optional
        Additional metadata.
        
    Returns
    -------
    Path
        Path to the created package.
    """
    import zipfile
    import pkg_resources
    
    output_path = Path(output_path)
    
    # Create temporary directory for package contents
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save model
        model_path = tmpdir / "model.pkl"
        full_metadata = {
            "package_name": name,
            "created_at": datetime.now().isoformat(),
            "mlpy_version": _get_mlpy_version()
        }
        if metadata:
            full_metadata.update(metadata)
            
        save_model(model, model_path, metadata=full_metadata)
        
        # Create requirements.txt
        if include_dependencies:
            requirements = []
            
            # Add MLPY
            requirements.append(f"mlpy>={_get_mlpy_version()}")
            
            # Add other detected dependencies
            if hasattr(model, 'packages'):
                for pkg in model.packages:
                    if pkg == 'sklearn':
                        requirements.append('scikit-learn')
                    elif pkg not in ['mlpy', 'numpy']:  # numpy is MLPY dependency
                        requirements.append(pkg)
                        
            # Write requirements
            req_path = tmpdir / "requirements.txt"
            with open(req_path, 'w') as f:
                f.write('\n'.join(requirements))
                
        # Create example script
        if include_examples:
            example_code = f'''"""Example usage of the {name} model."""

from mlpy.persistence import load_model

# Load the model
model = load_model("model.pkl")

# Example prediction (adjust based on your model's requirements)
# X = ...  # Your input data
# predictions = model.predict(X)

print(f"Model loaded successfully: {{type(model).__name__}}")
print(f"Model is trained: {{getattr(model, 'is_trained', 'N/A')}}")
'''
            
            example_path = tmpdir / "example.py"
            with open(example_path, 'w') as f:
                f.write(example_code)
                
        # Create README
        readme_content = f'''# {name} Model Package

This package contains a trained MLPY model.

## Contents

- `model.pkl`: The trained model
- `requirements.txt`: Python dependencies
- `example.py`: Example usage code

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from mlpy.persistence import load_model

model = load_model("model.pkl")
# Use the model for predictions
```

## Metadata

Created: {datetime.now().isoformat()}
MLPY Version: {_get_mlpy_version()}
'''
        
        readme_path = tmpdir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        # Create ZIP package
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(tmpdir)
                    zf.write(file, arcname)
                    
    return output_path


def _get_mlpy_version() -> str:
    """Get MLPY version."""
    try:
        import mlpy
        return getattr(mlpy, '__version__', '0.1.0')
    except:
        return '0.1.0'


def compute_model_hash(model_path: Union[str, Path]) -> str:
    """Compute hash of a saved model file.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the model file.
        
    Returns
    -------
    str
        SHA256 hash of the file.
    """
    sha256_hash = hashlib.sha256()
    
    with open(model_path, "rb") as f:
        # Read in chunks for large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            
    return sha256_hash.hexdigest()


__all__ = [
    "ModelRegistry",
    "export_model_package",
    "compute_model_hash"
]