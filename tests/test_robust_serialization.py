"""
Tests para el sistema de serialización robusta.

Verifica integridad, múltiples formatos y fallbacks.
"""

import pytest
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import hashlib
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.serialization.robust_serializer import (
    RobustSerializer,
    SerializationError,
    ChecksumMismatchError
)


class TestRobustSerializer:
    """Tests para el serializador robusto."""
    
    @pytest.fixture
    def serializer(self):
        """Crear instancia del serializador."""
        return RobustSerializer()
    
    @pytest.fixture
    def temp_dir(self):
        """Crear directorio temporal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_and_load_simple_object(self, serializer, temp_dir):
        """Test guardar y cargar objeto simple."""
        obj = {'key': 'value', 'number': 42}
        path = temp_dir / 'test.pkl'
        
        # Guardar
        result = serializer.save(obj, path)
        assert result['success'] == True
        assert 'checksum' in result
        assert path.exists()
        
        # Cargar
        loaded = serializer.load(path)
        assert loaded == obj
    
    def test_save_with_metadata(self, serializer, temp_dir):
        """Test guardar con metadata."""
        obj = [1, 2, 3]
        metadata = {
            'version': '1.0',
            'author': 'test',
            'description': 'test data'
        }
        path = temp_dir / 'with_metadata.pkl'
        
        result = serializer.save(obj, path, metadata=metadata)
        assert result['success'] == True
        
        # Verificar que metadata se guardó
        meta_path = path.with_suffix('.meta.json')
        assert meta_path.exists()
        
        with open(meta_path, 'r') as f:
            saved_meta = json.load(f)
        assert saved_meta['version'] == '1.0'
        assert saved_meta['author'] == 'test'
    
    def test_checksum_validation(self, serializer, temp_dir):
        """Test validación de checksum."""
        obj = {'data': np.array([1, 2, 3])}
        path = temp_dir / 'checksum_test.pkl'
        
        # Guardar con checksum
        save_result = serializer.save(obj, path)
        original_checksum = save_result['checksum']
        
        # Cargar con validación
        loaded = serializer.load(path, validate_checksum=True)
        assert np.array_equal(loaded['data'], obj['data'])
        
        # Corromper archivo
        with open(path, 'ab') as f:
            f.write(b'corrupted')
        
        # Debe fallar validación
        with pytest.raises(ChecksumMismatchError):
            serializer.load(path, validate_checksum=True)
    
    def test_fallback_formats(self, serializer, temp_dir):
        """Test fallback entre formatos."""
        # Objeto que podría fallar con pickle estándar
        import sklearn.ensemble
        model = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        X = np.random.randn(50, 5)
        y = np.random.choice([0, 1], 50)
        model.fit(X, y)
        
        path = temp_dir / 'model.pkl'
        
        # Debe intentar múltiples formatos
        result = serializer.save(model, path)
        assert result['success'] == True
        assert result['format'] in ['pickle', 'cloudpickle', 'joblib']
        
        # Cargar
        loaded_model = serializer.load(path)
        assert hasattr(loaded_model, 'predict')
        
        # Verificar que predice correctamente
        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_compression(self, serializer, temp_dir):
        """Test compresión de archivos."""
        # Objeto grande
        large_obj = {
            'data': np.random.randn(1000, 100),
            'labels': ['label'] * 1000
        }
        
        path_uncompressed = temp_dir / 'uncompressed.pkl'
        path_compressed = temp_dir / 'compressed.pkl'
        
        # Sin compresión
        serializer.save(large_obj, path_uncompressed, compress=False)
        
        # Con compresión
        serializer.save(large_obj, path_compressed, compress=True)
        
        # Archivo comprimido debe ser más pequeño
        size_uncompressed = path_uncompressed.stat().st_size
        size_compressed = path_compressed.stat().st_size
        assert size_compressed < size_uncompressed
        
        # Ambos deben cargar correctamente
        loaded_uncompressed = serializer.load(path_uncompressed)
        loaded_compressed = serializer.load(path_compressed)
        
        assert np.array_equal(
            loaded_uncompressed['data'],
            loaded_compressed['data']
        )
    
    def test_numpy_array_serialization(self, serializer, temp_dir):
        """Test serialización de arrays NumPy."""
        arrays = {
            'float64': np.random.randn(10, 10),
            'int32': np.random.randint(0, 100, (5, 5)).astype(np.int32),
            'bool': np.random.choice([True, False], (3, 3))
        }
        
        for dtype, arr in arrays.items():
            path = temp_dir / f'array_{dtype}.pkl'
            
            result = serializer.save(arr, path)
            assert result['success'] == True
            
            loaded = serializer.load(path)
            assert np.array_equal(loaded, arr)
            assert loaded.dtype == arr.dtype
    
    def test_pandas_dataframe_serialization(self, serializer, temp_dir):
        """Test serialización de DataFrames."""
        df = pd.DataFrame({
            'numeric': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'datetime': pd.date_range('2024-01-01', periods=100),
            'boolean': np.random.choice([True, False], 100)
        })
        
        path = temp_dir / 'dataframe.pkl'
        
        result = serializer.save(df, path)
        assert result['success'] == True
        
        loaded_df = serializer.load(path)
        pd.testing.assert_frame_equal(loaded_df, df)
    
    def test_error_handling(self, serializer, temp_dir):
        """Test manejo de errores."""
        # Objeto no serializable
        class NonSerializable:
            def __reduce__(self):
                raise TypeError("Cannot serialize")
        
        obj = NonSerializable()
        path = temp_dir / 'error.pkl'
        
        with pytest.raises(SerializationError):
            serializer.save(obj, path)
        
        # Path inválido para carga
        with pytest.raises(FileNotFoundError):
            serializer.load('nonexistent.pkl')
    
    def test_versioning(self, serializer, temp_dir):
        """Test versionado de objetos."""
        obj_v1 = {'version': 1, 'data': [1, 2, 3]}
        obj_v2 = {'version': 2, 'data': [1, 2, 3, 4]}
        
        path_v1 = temp_dir / 'obj_v1.pkl'
        path_v2 = temp_dir / 'obj_v2.pkl'
        
        # Guardar con metadata de versión
        serializer.save(obj_v1, path_v1, metadata={'version': '1.0.0'})
        serializer.save(obj_v2, path_v2, metadata={'version': '2.0.0'})
        
        # Verificar metadata
        meta_v1 = json.loads((path_v1.with_suffix('.meta.json')).read_text())
        meta_v2 = json.loads((path_v2.with_suffix('.meta.json')).read_text())
        
        assert meta_v1['version'] == '1.0.0'
        assert meta_v2['version'] == '2.0.0'
    
    def test_batch_serialization(self, serializer, temp_dir):
        """Test serialización de múltiples objetos."""
        objects = {
            'model': {'type': 'model', 'params': [1, 2, 3]},
            'data': np.random.randn(50, 10),
            'config': {'learning_rate': 0.01, 'epochs': 100}
        }
        
        # Guardar batch
        for name, obj in objects.items():
            path = temp_dir / f'{name}.pkl'
            result = serializer.save(obj, path)
            assert result['success'] == True
        
        # Cargar batch
        loaded_objects = {}
        for name in objects.keys():
            path = temp_dir / f'{name}.pkl'
            loaded_objects[name] = serializer.load(path)
        
        # Verificar
        assert loaded_objects['model'] == objects['model']
        assert np.array_equal(loaded_objects['data'], objects['data'])
        assert loaded_objects['config'] == objects['config']


class TestSerializationIntegration:
    """Tests de integración con modelos ML reales."""
    
    @pytest.fixture
    def serializer(self):
        return RobustSerializer()
    
    def test_sklearn_model_serialization(self, serializer, temp_dir):
        """Test con modelo scikit-learn."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Crear pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=5))
        ])
        
        # Entrenar
        X = np.random.randn(100, 10)
        y = np.random.choice([0, 1], 100)
        pipeline.fit(X, y)
        
        # Serializar
        path = temp_dir / 'sklearn_pipeline.pkl'
        metadata = {
            'model_type': 'RandomForest',
            'n_features': 10,
            'accuracy': 0.85
        }
        
        result = serializer.save(pipeline, path, metadata=metadata)
        assert result['success'] == True
        
        # Cargar y verificar
        loaded_pipeline = serializer.load(path, validate_checksum=True)
        
        # Debe predecir correctamente
        predictions = loaded_pipeline.predict(X[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_complex_nested_object(self, serializer, temp_dir):
        """Test con objeto complejo anidado."""
        complex_obj = {
            'models': [
                {'name': 'model1', 'params': np.array([1, 2, 3])},
                {'name': 'model2', 'params': np.array([4, 5, 6])}
            ],
            'data': {
                'train': pd.DataFrame({'x': [1, 2], 'y': [3, 4]}),
                'test': pd.DataFrame({'x': [5, 6], 'y': [7, 8]})
            },
            'metadata': {
                'created': '2024-01-01',
                'version': 1.0,
                'tags': ['ml', 'production']
            }
        }
        
        path = temp_dir / 'complex.pkl'
        
        result = serializer.save(complex_obj, path)
        assert result['success'] == True
        
        loaded = serializer.load(path)
        
        # Verificar estructura
        assert len(loaded['models']) == 2
        assert np.array_equal(loaded['models'][0]['params'], np.array([1, 2, 3]))
        pd.testing.assert_frame_equal(loaded['data']['train'], complex_obj['data']['train'])
        assert loaded['metadata']['version'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])