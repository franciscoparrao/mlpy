"""
Tests para el sistema de validación con Pydantic.

Verifica que el sistema proporciona errores educativos y 
previene problemas comunes en MLPY.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.validation import (
    validate_task_data,
    ValidatedTask,
    MLPYValidationError,
    TaskValidationError
)


class TestDataValidation:
    """Tests para validación de datos."""
    
    def test_valid_dataframe(self):
        """Test con DataFrame válido."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        result = validate_task_data(df, target='target')
        assert result['valid'] == True
        assert len(result['errors']) == 0
    
    def test_missing_target_column(self):
        """Test cuando falta la columna target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3]
        })
        
        result = validate_task_data(df, target='missing_column')
        assert result['valid'] == False
        assert any('not found' in str(e).lower() for e in result['errors'])
    
    def test_empty_dataframe(self):
        """Test con DataFrame vacío."""
        df = pd.DataFrame()
        
        result = validate_task_data(df, target='target')
        assert result['valid'] == False
        assert any('empty' in str(e).lower() for e in result['errors'])
    
    def test_nan_values_warning(self):
        """Test que detecta valores NaN."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'target': [0, 1, 0, 1]
        })
        
        result = validate_task_data(df, target='target')
        assert result['valid'] == True  # NaN genera warning, no error
        assert len(result['warnings']) > 0
        assert any('nan' in str(w).lower() or 'missing' in str(w).lower() 
                  for w in result['warnings'])
    
    def test_insufficient_samples(self):
        """Test con muy pocas muestras."""
        df = pd.DataFrame({
            'feature1': [1],
            'target': [0]
        })
        
        result = validate_task_data(df, target='target')
        assert result['valid'] == False
        assert any('sample' in str(e).lower() for e in result['errors'])
    
    def test_duplicate_columns(self):
        """Test con columnas duplicadas."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [0.1, 0.2, 0.3],
            'target': [0, 1, 0]
        })
        # Simular columnas duplicadas
        df['feature1_duplicate'] = df['feature1']
        
        result = validate_task_data(df, target='target')
        # Debería pasar pero con warnings sobre posible redundancia
        assert result['valid'] == True
    
    def test_constant_feature_warning(self):
        """Test que detecta features constantes."""
        df = pd.DataFrame({
            'feature1': [1, 1, 1, 1],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'target': [0, 1, 0, 1]
        })
        
        result = validate_task_data(df, target='target')
        assert result['valid'] == True
        assert len(result['warnings']) > 0
        assert any('constant' in str(w).lower() or 'variance' in str(w).lower() 
                  for w in result['warnings'])


class TestValidatedTask:
    """Tests para la clase ValidatedTask."""
    
    def test_create_valid_classification_task(self):
        """Test creación de tarea de clasificación válida."""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        task = ValidatedTask(
            data=df,
            target='target',
            task_type='classification'
        )
        
        assert task.task is not None
        assert task.task.task_type == 'classification'
        assert task.task.n_obs == 100
    
    def test_create_valid_regression_task(self):
        """Test creación de tarea de regresión válida."""
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randn(50)
        })
        
        task = ValidatedTask(
            data=df,
            target='target',
            task_type='regression'
        )
        
        assert task.task is not None
        assert task.task.task_type == 'regression'
    
    def test_invalid_task_raises_helpful_error(self):
        """Test que errores en tareas generan mensajes útiles."""
        df = pd.DataFrame({
            'feature1': [1, 2],
            'target': [0, 1]
        })
        
        with pytest.raises(MLPYValidationError) as exc_info:
            ValidatedTask(
                data=df,
                target='missing_column',
                task_type='classification'
            )
        
        error_msg = str(exc_info.value)
        assert 'suggestion' in error_msg.lower() or 'try' in error_msg.lower()
    
    def test_automatic_task_type_detection(self):
        """Test detección automática del tipo de tarea."""
        # Clasificación binaria
        df_binary = pd.DataFrame({
            'feature1': np.random.randn(50),
            'target': np.random.choice([0, 1], 50)
        })
        
        task_binary = ValidatedTask(data=df_binary, target='target')
        assert task_binary.task.task_type == 'classification'
        
        # Regresión
        df_regression = pd.DataFrame({
            'feature1': np.random.randn(50),
            'target': np.random.randn(50)
        })
        
        task_regression = ValidatedTask(data=df_regression, target='target')
        assert task_regression.task.task_type == 'regression'
    
    def test_task_with_categorical_features(self):
        """Test con features categóricas."""
        df = pd.DataFrame({
            'numeric': np.random.randn(50),
            'categorical': np.random.choice(['A', 'B', 'C'], 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        task = ValidatedTask(data=df, target='target')
        assert task.task is not None
        # El sistema debería manejar categóricas automáticamente


class TestErrorMessages:
    """Tests para verificar que los mensajes de error son educativos."""
    
    def test_error_message_formatting(self):
        """Test formato de mensajes de error."""
        error = MLPYValidationError(
            "El DataFrame no tiene columna target",
            field='target',
            suggestions=[
                "Verificar el nombre de la columna",
                "Usar df.columns para ver columnas disponibles"
            ]
        )
        
        msg = error.format_message()
        assert "ERROR" in msg
        assert "SUGGESTION" in msg
        assert "Verificar" in msg
    
    def test_data_error_suggestions(self):
        """Test sugerencias en errores de datos."""
        error = TaskValidationError(
            "Datos insuficientes para entrenar",
            field='data',
            suggestions=['Aumentar el número de muestras', 'Usar validación cruzada']
        )
        
        msg = str(error)
        assert "insuficientes" in msg.lower()
        assert "data" in msg.lower()
    
    def test_task_creation_error_context(self):
        """Test contexto en errores de creación de tareas."""
        error = TaskValidationError(
            "No se pudo crear la tarea de clasificación: Target tiene valores continuos",
            field='task_type',
            suggestions=['Usar regresión en lugar de clasificación', 'Discretizar el target']
        )
        
        msg = str(error)
        assert "clasificación" in msg.lower() or "classification" in msg.lower()
        assert "continuos" in msg.lower() or "continuous" in msg.lower()


class TestValidationIntegration:
    """Tests de integración con el flujo completo."""
    
    def test_full_validation_workflow(self):
        """Test flujo completo de validación."""
        # 1. Datos crudos con problemas
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'constant': [1, 1, 1, 1, 1],
            'target': [0, 1, 0, 1, 0]
        })
        
        # 2. Validar datos
        validation = validate_task_data(df, target='target')
        assert validation['valid'] == True
        assert len(validation['warnings']) > 0  # NaN y constante
        
        # 3. Crear tarea con validación
        task = ValidatedTask(
            data=df.dropna(),  # Limpiar NaN basado en warning
            target='target',
            task_type='classification'
        )
        
        assert task.task is not None
        assert task.task.n_obs == 4  # 5 - 1 NaN
    
    def test_validation_prevents_common_mistakes(self):
        """Test que la validación previene errores comunes."""
        # Error 1: Olvidar especificar target
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [0, 1, 0]
        })
        
        result = validate_task_data(df, target=None)
        assert result['valid'] == False
        
        # Error 2: Target como string en clasificación
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4],
            'target': ['yes', 'no', 'yes', 'no']
        })
        
        result = validate_task_data(df, target='target')
        # Debería detectar y sugerir encoding
        assert len(result['warnings']) > 0 or len(result['errors']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])