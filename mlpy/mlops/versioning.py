"""
Model Versioning and Management
================================

Version control for ML models with rollback capabilities.
"""

import os
import json
import shutil
import hashlib
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

from ..learners.base import Learner
from ..measures.base import Measure

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    
    model_id: str
    version: str
    created_at: str
    created_by: str
    description: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    file_path: str
    file_hash: str
    parent_version: Optional[str] = None
    is_active: bool = False
    is_production: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(**data)


class VersionManager:
    """Manages model versions with Git-like semantics."""
    
    def __init__(self, storage_path: str = "./model_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_path / "versions.json"
        self.versions: Dict[str, Dict[str, ModelVersion]] = {}
        self.load_versions()
    
    def load_versions(self):
        """Load version metadata from disk."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.versions[model_id] = {
                        v_id: ModelVersion.from_dict(v_data)
                        for v_id, v_data in versions.items()
                    }
        logger.info(f"Loaded {sum(len(v) for v in self.versions.values())} versions")
    
    def save_versions(self):
        """Save version metadata to disk."""
        data = {}
        for model_id, versions in self.versions.items():
            data[model_id] = {
                v_id: v.to_dict()
                for v_id, v in versions.items()
            }
        
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(
        self,
        model: Learner,
        model_id: str,
        description: str = "",
        created_by: str = "system",
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """Create a new model version."""
        
        # Generate version ID
        version_id = self._generate_version_id(model_id)
        
        # Create model directory
        model_dir = self.storage_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        file_path = model_dir / f"{version_id}.pkl"
        joblib.dump(model, file_path)
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Create version object
        version = ModelVersion(
            model_id=model_id,
            version=version_id,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            description=description or f"Version {version_id} of {model_id}",
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or [],
            file_path=str(file_path),
            file_hash=file_hash,
            parent_version=parent_version,
            is_active=True,
            is_production=False
        )
        
        # Store version
        if model_id not in self.versions:
            self.versions[model_id] = {}
        self.versions[model_id][version_id] = version
        
        # Save metadata
        self.save_versions()
        
        logger.info(f"Created version {version_id} for model {model_id}")
        return version
    
    def get_version(self, model_id: str, version_id: Optional[str] = None) -> Optional[ModelVersion]:
        """Get a specific model version or the latest."""
        if model_id not in self.versions:
            return None
        
        if version_id:
            return self.versions[model_id].get(version_id)
        
        # Return latest version
        versions = self.versions[model_id]
        if versions:
            latest = sorted(versions.values(), key=lambda v: v.created_at, reverse=True)[0]
            return latest
        
        return None
    
    def load_version(self, model_id: str, version_id: Optional[str] = None) -> Optional[Learner]:
        """Load a specific model version."""
        version = self.get_version(model_id, version_id)
        if not version:
            return None
        
        file_path = Path(version.file_path)
        if not file_path.exists():
            logger.error(f"Model file not found: {file_path}")
            return None
        
        # Verify file hash
        current_hash = self._calculate_file_hash(file_path)
        if current_hash != version.file_hash:
            logger.warning(f"File hash mismatch for {model_id}:{version_id}")
        
        # Load model
        model = joblib.load(file_path)
        logger.info(f"Loaded model {model_id}:{version_id}")
        return model
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model."""
        if model_id not in self.versions:
            return []
        
        versions = list(self.versions[model_id].values())
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two model versions."""
        v1 = self.get_version(model_id, version1)
        v2 = self.get_version(model_id, version2)
        
        if not v1 or not v2:
            raise ValueError("Version not found")
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "created_at_diff": (
                datetime.fromisoformat(v2.created_at) - 
                datetime.fromisoformat(v1.created_at)
            ).total_seconds(),
            "metrics_diff": {},
            "parameters_diff": {},
            "tags_added": list(set(v2.tags) - set(v1.tags)),
            "tags_removed": list(set(v1.tags) - set(v2.tags))
        }
        
        # Compare metrics
        all_metrics = set(v1.metrics.keys()) | set(v2.metrics.keys())
        for metric in all_metrics:
            m1 = v1.metrics.get(metric, 0)
            m2 = v2.metrics.get(metric, 0)
            comparison["metrics_diff"][metric] = {
                "v1": m1,
                "v2": m2,
                "diff": m2 - m1,
                "improvement": ((m2 - m1) / m1 * 100) if m1 != 0 else 0
            }
        
        # Compare parameters
        all_params = set(v1.parameters.keys()) | set(v2.parameters.keys())
        for param in all_params:
            p1 = v1.parameters.get(param)
            p2 = v2.parameters.get(param)
            if p1 != p2:
                comparison["parameters_diff"][param] = {
                    "v1": p1,
                    "v2": p2
                }
        
        return comparison
    
    def rollback(self, model_id: str, target_version: str) -> ModelVersion:
        """Rollback to a specific version."""
        target = self.get_version(model_id, target_version)
        if not target:
            raise ValueError(f"Version {target_version} not found")
        
        # Load the target model
        model = self.load_version(model_id, target_version)
        if not model:
            raise ValueError(f"Could not load model {model_id}:{target_version}")
        
        # Create new version as rollback
        new_version = self.create_version(
            model=model,
            model_id=model_id,
            description=f"Rollback to version {target_version}",
            created_by="rollback",
            metrics=target.metrics,
            parameters=target.parameters,
            tags=target.tags + ["rollback"],
            parent_version=target_version
        )
        
        logger.info(f"Rolled back {model_id} to version {target_version}")
        return new_version
    
    def promote_to_production(self, model_id: str, version_id: str) -> ModelVersion:
        """Promote a version to production."""
        version = self.get_version(model_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Demote current production version
        for v in self.versions[model_id].values():
            v.is_production = False
        
        # Promote new version
        version.is_production = True
        version.tags = list(set(version.tags + ["production"]))
        
        # Save changes
        self.save_versions()
        
        logger.info(f"Promoted {model_id}:{version_id} to production")
        return version
    
    def get_production_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the current production version."""
        if model_id not in self.versions:
            return None
        
        for version in self.versions[model_id].values():
            if version.is_production:
                return version
        
        return None
    
    def tag_version(self, model_id: str, version_id: str, tags: List[str]):
        """Add tags to a version."""
        version = self.get_version(model_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        version.tags = list(set(version.tags + tags))
        self.save_versions()
        
        logger.info(f"Tagged {model_id}:{version_id} with {tags}")
    
    def delete_version(self, model_id: str, version_id: str):
        """Delete a specific version."""
        version = self.get_version(model_id, version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        if version.is_production:
            raise ValueError("Cannot delete production version")
        
        # Delete file
        file_path = Path(version.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Remove from versions
        del self.versions[model_id][version_id]
        
        # Clean up empty model entries
        if not self.versions[model_id]:
            del self.versions[model_id]
            model_dir = self.storage_path / model_id
            if model_dir.exists() and not list(model_dir.iterdir()):
                model_dir.rmdir()
        
        # Save changes
        self.save_versions()
        
        logger.info(f"Deleted version {model_id}:{version_id}")
    
    def cleanup_old_versions(self, model_id: str, keep_last: int = 5):
        """Clean up old versions, keeping only the most recent ones."""
        versions = self.list_versions(model_id)
        
        # Keep production and recent versions
        to_delete = []
        kept = 0
        for version in versions:
            if version.is_production:
                continue
            if kept < keep_last:
                kept += 1
                continue
            to_delete.append(version.version)
        
        # Delete old versions
        for version_id in to_delete:
            self.delete_version(model_id, version_id)
        
        logger.info(f"Cleaned up {len(to_delete)} old versions of {model_id}")
    
    def _generate_version_id(self, model_id: str) -> str:
        """Generate a unique version ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Add sequential number if needed
        if model_id in self.versions:
            existing = [v for v in self.versions[model_id].keys() if v.startswith(timestamp)]
            if existing:
                seq = len(existing) + 1
                return f"{timestamp}_{seq:03d}"
        
        return timestamp
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()