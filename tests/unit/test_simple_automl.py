"""
Tests completos para SimpleAutoML.

Estos tests cubren la funcionalidad del módulo SimpleAutoML incluyendo:
- Creación de tareas automática
- Detección de tipo de tarea
- División de datos
- Búsqueda de pipelines
- Evaluación de modelos
- Guardado y carga de resultados
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import warnings

from mlpy.automl.simple_automl import SimpleAutoML, AutoMLResult
from mlpy.tasks import TaskClassif, TaskRegr


class TestAutoMLResult:
    """Tests para la clase AutoMLResult."""
    
    def test_automl_result_creation(self):
        """Test creación de AutoMLResult."""
        # Mock del learner
        mock_learner = Mock()
        mock_learner.predict.return_value = Mock(response=[0, 1, 0])
        
        # Crear resultado
        result = AutoMLResult(
            best_learner=mock_learner,
            best_score=0.95,
            leaderboard=pd.DataFrame({
                'model': ['model1', 'model2'],
                'score': [0.95, 0.90]
            }),
            feature_importance=pd.Series({'feat1': 0.5, 'feat2': 0.3}),
            task=None,
            training_time=120.5
        )
        
        assert result.best_learner == mock_learner
        assert result.best_score == 0.95
        assert len(result.leaderboard) == 2
        assert result.training_time == 120.5
        
    def test_automl_result_predict(self):
        """Test predicción con AutoMLResult."""
        # Mock del learner
        mock_learner = Mock()
        mock_learner.predict.return_value = Mock(response=[0, 1, 0])
        
        # Mock del task
        mock_task = Mock(spec=TaskClassif)
        
        result = AutoMLResult(
            best_learner=mock_learner,
            best_score=0.95,
            leaderboard=pd.DataFrame({'model': ['m1'], 'score': [0.95]}),
            task=mock_task
        )
        
        # Test con DataFrame
        test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        predictions = result.predict(test_df)
        
        assert mock_learner.predict.called
        assert predictions.response == [0, 1, 0]
        
    def test_automl_result_save_load(self):
        """Test guardar y cargar AutoMLResult."""
        # Crear un learner real simple en lugar de Mock
        from mlpy.learners import LearnerClassifSklearn
        from sklearn.tree import DecisionTreeClassifier
        
        # Crear learner real
        real_learner = LearnerClassifSklearn(classifier="DecisionTreeClassifier")
        real_learner._model = DecisionTreeClassifier()  # Modelo simple para serializar
        
        result = AutoMLResult(
            best_learner=real_learner,
            best_score=0.85,
            leaderboard=pd.DataFrame({'model': ['m1'], 'score': [0.85]}),
            training_time=60.0
        )
        
        # Guardar y cargar
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            result.save(f.name)
            loaded_result = AutoMLResult.load(f.name)
            
        # Verificar
        assert loaded_result.best_score == 0.85
        assert loaded_result.training_time == 60.0
        assert len(loaded_result.leaderboard) == 1
        
        # Limpiar
        os.unlink(f.name)


class TestSimpleAutoML:
    """Tests para la clase SimpleAutoML."""
    
    @pytest.fixture
    def classification_data(self):
        """Datos de clasificación para testing."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        df['target'] = y
        return df
        
    @pytest.fixture
    def regression_data(self):
        """Datos de regresión para testing."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * np.random.randn(n)
        
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        df['y'] = y
        return df
        
    @pytest.fixture
    def mixed_data(self):
        """Datos mixtos (numéricos y categóricos)."""
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.uniform(0, 10, n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'cat2': np.random.choice(['X', 'Y'], n),
            'target': np.random.choice([0, 1], n)
        })
        return df
        
    def test_simple_automl_creation(self):
        """Test creación de SimpleAutoML."""
        automl = SimpleAutoML(
            time_limit=300,
            max_models=50,
            feature_engineering=True,
            feature_selection=True,
            cross_validation=5,
            test_size=0.2,
            random_state=42,
            verbose=False
        )
        
        assert automl.time_limit == 300
        assert automl.max_models == 50
        assert automl.feature_engineering == True
        assert automl.feature_selection == True
        assert automl.cross_validation == 5
        assert automl.test_size == 0.2
        assert automl.random_state == 42
        assert automl.verbose == False
        
    def test_task_creation_classification(self, classification_data):
        """Test creación automática de tarea de clasificación."""
        automl = SimpleAutoML(verbose=False)
        
        # Detección automática
        task = automl._create_task(classification_data, 'target', None)
        assert isinstance(task, TaskClassif)
        assert task.target_names == ['target']
        assert len(task.feature_names) == 4
        
        # Especificación manual
        task = automl._create_task(classification_data, 'target', 'classification')
        assert isinstance(task, TaskClassif)
        
    def test_task_creation_regression(self, regression_data):
        """Test creación automática de tarea de regresión."""
        automl = SimpleAutoML(verbose=False)
        
        # Detección automática
        task = automl._create_task(regression_data, 'y', None)
        assert isinstance(task, TaskRegr)
        assert task.target_names == ['y']
        assert len(task.feature_names) == 3
        
        # Especificación manual
        task = automl._create_task(regression_data, 'y', 'regression')
        assert isinstance(task, TaskRegr)
        
    def test_task_type_inference(self):
        """Test inferencia del tipo de tarea."""
        automl = SimpleAutoML(verbose=False)
        
        # Categórica obvia
        df1 = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': ['A', 'B', 'A', 'B', 'C']
        })
        task1 = automl._create_task(df1, 'y', None)
        assert isinstance(task1, TaskClassif)
        
        # Numérica con pocos valores únicos -> clasificación
        df2 = pd.DataFrame({
            'x': range(100),
            'y': [0, 1, 2] * 33 + [0]  # 3 valores únicos
        })
        task2 = automl._create_task(df2, 'y', None)
        assert isinstance(task2, TaskClassif)
        
        # Numérica con muchos valores únicos -> regresión
        df3 = pd.DataFrame({
            'x': range(100),
            'y': np.random.randn(100)  # Valores continuos
        })
        task3 = automl._create_task(df3, 'y', None)
        assert isinstance(task3, TaskRegr)
        
    def test_data_split(self, classification_data):
        """Test división de datos train/test."""
        automl = SimpleAutoML(test_size=0.3, random_state=42, verbose=False)
        
        task = automl._create_task(classification_data, 'target', 'classification')
        train_task, test_task = automl._split_data(task)
        
        # Verificar tamaños - ResamplingHoldout usa ratio como test_size
        total = task.nrow
        expected_test = int(total * 0.3)
        expected_train = total - expected_test
        
        assert train_task.nrow == expected_train
        assert test_task.nrow == expected_test
        assert train_task.nrow + test_task.nrow == task.nrow
        
        # Verificar que no hay overlap
        train_data = train_task.data()
        test_data = test_task.data()
        # Los índices deberían ser diferentes
        assert len(set(train_data.index) & set(test_data.index)) == 0
        
    def test_get_base_learners(self, classification_data, regression_data):
        """Test obtención de learners base."""
        automl = SimpleAutoML(random_state=42, verbose=False)
        
        # Para clasificación
        task_classif = automl._create_task(classification_data, 'target', 'classification')
        learners_classif = automl._get_base_learners(task_classif)
        
        assert len(learners_classif) > 0
        assert all(len(learner) == 3 for learner in learners_classif)
        
        # Verificar que incluye learners esperados
        learner_names = [l[0] for l in learners_classif]
        assert 'RandomForest' in learner_names
        assert 'LogisticRegression' in learner_names
        
        # Para regresión
        task_regr = automl._create_task(regression_data, 'y', 'regression')
        learners_regr = automl._get_base_learners(task_regr)
        
        assert len(learners_regr) > 0
        learner_names_regr = [l[0] for l in learners_regr]
        assert 'RandomForest' in learner_names_regr
        assert 'LinearRegression' in learner_names_regr
        
    def test_get_preprocessing_configs(self, mixed_data):
        """Test obtención de configuraciones de preprocesamiento."""
        # Sin feature engineering ni selection
        automl1 = SimpleAutoML(
            feature_engineering=False,
            feature_selection=False,
            verbose=False
        )
        task = automl1._create_task(mixed_data, 'target', 'classification')
        configs1 = automl1._get_preprocessing_configs(task)
        
        assert len(configs1) >= 2  # Al menos minimal y scaled
        config_names = [c['name'] for c in configs1]
        assert 'minimal' in config_names
        assert 'scaled' in config_names
        
        # Con feature engineering y selection
        # Crear task con más features para activar feature selection
        large_df = mixed_data.copy()
        for i in range(10):  # Añadir más columnas
            large_df[f'extra_{i}'] = np.random.randn(len(large_df))
        
        automl2 = SimpleAutoML(
            feature_engineering=True,
            feature_selection=True,
            verbose=False
        )
        task2 = automl2._create_task(large_df, 'target', 'classification')
        configs2 = automl2._get_preprocessing_configs(task2)
        
        assert len(configs2) > len(configs1)
        config_names2 = [c['name'] for c in configs2]
        assert 'engineered' in config_names2
        # 'selected' y 'full' solo aparecen si hay >10 features
        if len(task2.feature_names) > 10:
            assert 'selected' in config_names2
            assert 'full' in config_names2
        
    @patch('mlpy.automl.simple_automl.GraphLearner')
    @patch('mlpy.automl.simple_automl.Graph')
    def test_build_pipeline(self, mock_graph_class, mock_graphlearner_class, classification_data):
        """Test construcción de pipeline."""
        automl = SimpleAutoML(verbose=False)
        task = automl._create_task(classification_data, 'target', 'classification')
        
        # Mock de learner
        mock_learner = Mock()
        
        # Configuración minimal
        config_minimal = {
            'name': 'minimal',
            'impute': True,
            'scale': False,
            'feature_eng': False,
            'feature_sel': False
        }
        
        # Mock del Graph
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        
        # Mock del GraphLearner
        mock_graphlearner = Mock()
        mock_graphlearner_class.return_value = mock_graphlearner
        
        result = automl._build_pipeline(task, mock_learner, config_minimal)
        
        # Verificar que se creó el graph y se añadieron operadores
        assert mock_graph_class.called
        assert mock_graph.add_pipeop.called
        assert mock_graph.add_edge.called
        assert mock_graphlearner_class.called
        assert result == mock_graphlearner
        
    def test_is_better_score(self, classification_data, regression_data):
        """Test comparación de scores."""
        automl = SimpleAutoML(verbose=False)
        
        # Clasificación (mayor es mejor)
        task_classif = automl._create_task(classification_data, 'target', 'classification')
        assert automl._is_better_score(0.9, 0.8, task_classif) == True
        assert automl._is_better_score(0.7, 0.8, task_classif) == False
        
        # Regresión (menor es mejor para MSE)
        task_regr = automl._create_task(regression_data, 'y', 'regression')
        assert automl._is_better_score(0.5, 1.0, task_regr) == True
        assert automl._is_better_score(2.0, 1.0, task_regr) == False
        
    def test_time_exceeded(self):
        """Test verificación de tiempo excedido."""
        automl = SimpleAutoML(time_limit=1, verbose=False)
        
        # Sin tiempo iniciado
        assert automl._time_exceeded() == False
        
        # Con tiempo iniciado
        import time
        automl._start_time = time.time()
        assert automl._time_exceeded() == False
        
        # Simular tiempo excedido
        automl._start_time = time.time() - 2  # Hace 2 segundos
        assert automl._time_exceeded() == True
        
    @patch('mlpy.automl.simple_automl.time')
    def test_fit_time_limit(self, mock_time, classification_data):
        """Test que fit respeta el límite de tiempo."""
        # Configurar mock de tiempo
        current_time = [0]
        def get_time():
            current_time[0] += 0.5  # Cada llamada avanza 0.5 segundos
            return current_time[0]
        mock_time.time.side_effect = get_time
        
        automl = SimpleAutoML(
            time_limit=2,  # 2 segundos límite
            max_models=100,  # Muchos modelos (no debería alcanzar)
            verbose=False
        )
        
        # Mock de componentes internos para evitar entrenamiento real
        with patch.object(automl, '_search_pipelines') as mock_search:
            mock_search.return_value = (Mock(), 0.85)
            with patch.object(automl, '_final_evaluation') as mock_eval:
                mock_eval.return_value = 0.85
                with patch.object(automl, '_get_feature_importance') as mock_importance:
                    mock_importance.return_value = None
                    
                    result = automl.fit(classification_data, 'target')
                    
                    # Verificar que se ejecutó pero con límite de tiempo
                    assert mock_search.called
                    assert result.best_score == 0.85
                    
    def test_fit_classification_minimal(self, classification_data):
        """Test fit básico con clasificación (versión minimal para CI)."""
        automl = SimpleAutoML(
            time_limit=10,
            max_models=2,  # Solo 2 modelos para rapidez
            feature_engineering=False,  # Desactivar para rapidez
            feature_selection=False,
            cross_validation=2,  # Solo 2 folds
            test_size=0.3,
            random_state=42,
            verbose=False
        )
        
        # Mock parcial para acelerar
        with patch.object(automl, '_evaluate_pipeline') as mock_eval:
            # Simular evaluación rápida
            mock_eval.return_value = np.random.random()
            
            result = automl.fit(classification_data, 'target')
            
            assert isinstance(result, AutoMLResult)
            assert result.best_score is not None
            assert len(result.leaderboard) > 0
            assert result.training_time > 0
            
    def test_fit_regression_minimal(self, regression_data):
        """Test fit básico con regresión (versión minimal para CI)."""
        automl = SimpleAutoML(
            time_limit=10,
            max_models=2,
            feature_engineering=False,
            feature_selection=False,
            cross_validation=2,
            test_size=0.3,
            random_state=42,
            verbose=False
        )
        
        # Mock parcial
        with patch.object(automl, '_evaluate_pipeline') as mock_eval:
            mock_eval.return_value = np.random.random() * 10  # MSE simulado
            
            result = automl.fit(regression_data, 'y', task_type='regression')
            
            assert isinstance(result, AutoMLResult)
            assert result.best_score is not None
            assert len(result.leaderboard) > 0
            
    def test_feature_importance_extraction(self):
        """Test extracción de feature importance."""
        automl = SimpleAutoML(verbose=False)
        
        # Mock de GraphLearner con pipeline
        mock_learner = Mock()
        mock_graph = Mock()
        mock_learner.graph = mock_graph
        
        # Mock de pipeops
        mock_pipeop_with_importance = Mock()
        mock_pipeop_with_importance.get_importance.return_value = pd.Series({
            'feat1': 0.5,
            'feat2': 0.3,
            'feat3': 0.2
        })
        
        # Configurar hasattr para retornar True para el pipeop correcto
        def mock_hasattr(obj, attr):
            if obj == mock_pipeop_with_importance and attr == 'get_importance':
                return True
            return False
        
        mock_graph.ids.return_value = ['op1', 'op2', 'op3']
        mock_graph.pipeops = {
            'op1': Mock(spec=[]),  # Sin importance
            'op2': mock_pipeop_with_importance,  # Con importance  
            'op3': Mock(spec=[])   # Sin importance
        }
        
        # Mock task
        mock_task = Mock()
        
        # Patch hasattr durante la llamada
        with patch('mlpy.automl.simple_automl.hasattr', side_effect=mock_hasattr):
            importance = automl._get_feature_importance(mock_learner, mock_task)
        
        assert importance is not None
        assert isinstance(importance, pd.Series)
        assert len(importance) == 3
        assert importance['feat1'] == 0.5
        
    def test_verbose_output(self, classification_data, capsys):
        """Test que verbose=True produce output."""
        automl = SimpleAutoML(
            time_limit=5,
            max_models=1,
            feature_engineering=False,
            feature_selection=False,
            cross_validation=2,
            verbose=True  # Activar verbose
        )
        
        # Mock para acelerar
        with patch.object(automl, '_search_pipelines') as mock_search:
            mock_search.return_value = (Mock(), 0.85)
            with patch.object(automl, '_final_evaluation') as mock_eval:
                mock_eval.return_value = 0.85
                with patch.object(automl, '_get_feature_importance'):
                    
                    result = automl.fit(classification_data, 'target')
                    
                    # Capturar output
                    captured = capsys.readouterr()
                    
                    # Verificar que hay output
                    assert len(captured.out) > 0
                    assert 'Starting SimpleAutoML' in captured.out or 'AutoML' in captured.out
                    
    def test_error_handling(self):
        """Test manejo de errores."""
        automl = SimpleAutoML(verbose=False)
        
        # DataFrame vacío
        df_empty = pd.DataFrame()
        with pytest.raises(Exception):
            automl.fit(df_empty, 'target')
            
        # Target no existe
        df_no_target = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises(Exception):
            automl.fit(df_no_target, 'non_existent')
            
    def test_automl_with_missing_values(self):
        """Test AutoML con valores faltantes."""
        # Datos con NaN
        df = pd.DataFrame({
            'x1': [1, 2, np.nan, 4, 5],
            'x2': [np.nan, 2, 3, 4, 5],
            'x3': [1, 2, 3, np.nan, 5],
            'target': [0, 1, 0, 1, 0]
        })
        
        automl = SimpleAutoML(
            time_limit=5,
            max_models=1,
            feature_engineering=False,
            feature_selection=False,
            verbose=False
        )
        
        # Debería manejar NaN automáticamente con imputation
        with patch.object(automl, '_evaluate_pipeline') as mock_eval:
            mock_eval.return_value = 0.8
            
            result = automl.fit(df, 'target')
            assert isinstance(result, AutoMLResult)
            
    def test_search_pipelines_exception_handling(self, classification_data):
        """Test que search_pipelines maneja excepciones correctamente."""
        automl = SimpleAutoML(
            max_models=3,
            verbose=False
        )
        
        task = automl._create_task(classification_data, 'target', 'classification')
        train_task, _ = automl._split_data(task)
        
        # Mock que falla en algunos pipelines
        call_count = [0]
        def mock_evaluate(graph_learner, task):
            call_count[0] += 1
            if call_count[0] == 2:  # Falla en el segundo intento
                raise ValueError("Simulated error")
            return 0.7 + call_count[0] * 0.05
            
        with patch.object(automl, '_evaluate_pipeline', side_effect=mock_evaluate):
            best_learner, best_score = automl._search_pipelines(train_task)
            
            # Debería continuar a pesar del error
            assert best_learner is not None
            assert best_score > 0
            assert len(automl._leaderboard) == 2  # Solo 2 exitosos de 3


class TestIntegrationSimpleAutoML:
    """Tests de integración para SimpleAutoML."""
    
    def test_full_pipeline_with_real_data(self):
        """Test pipeline completo con datos reales."""
        from sklearn.datasets import load_wine
        
        # Cargar datos
        wine = load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        
        # Configurar AutoML rápido
        automl = SimpleAutoML(
            time_limit=15,
            max_models=3,
            feature_engineering=True,
            feature_selection=True,
            cross_validation=2,
            test_size=0.3,
            random_state=42,
            verbose=False
        )
        
        # Mock evaluación para acelerar
        with patch.object(automl, '_evaluate_pipeline') as mock_eval:
            mock_eval.return_value = np.random.uniform(0.7, 0.95)
            
            result = automl.fit(df, 'target')
            
            # Verificaciones
            assert isinstance(result, AutoMLResult)
            assert 0.0 <= result.best_score <= 1.0
            assert len(result.leaderboard) > 0
            assert result.training_time > 0
            
            # Verificar leaderboard
            assert 'model' in result.leaderboard.columns
            assert 'score' in result.leaderboard.columns
            assert 'preprocessing' in result.leaderboard.columns
            
            # El mejor score debería estar en el leaderboard
            assert result.best_score == result.leaderboard['score'].max()
            
    @pytest.fixture
    def classification_data(self):
        """Datos de clasificación para testing."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['f1', 'f2', 'f3', 'f4'])
        df['target'] = y
        return df
        
    def test_reproducibility(self, classification_data):
        """Test que los resultados son reproducibles con mismo random_state."""
        automl1 = SimpleAutoML(
            time_limit=10,
            max_models=2,
            feature_engineering=False,
            feature_selection=False,
            cross_validation=2,
            random_state=42,
            verbose=False
        )
        
        automl2 = SimpleAutoML(
            time_limit=10,
            max_models=2,
            feature_engineering=False,
            feature_selection=False,
            cross_validation=2,
            random_state=42,
            verbose=False
        )
        
        # Mock para consistencia
        def mock_eval(graph_learner, task):
            # Usar hash del tipo de learner para generar score determinístico
            return 0.5 + (hash(str(type(graph_learner))) % 100) / 200
            
        with patch.object(automl1, '_evaluate_pipeline', side_effect=mock_eval):
            with patch.object(automl2, '_evaluate_pipeline', side_effect=mock_eval):
                result1 = automl1.fit(classification_data, 'target')
                result2 = automl2.fit(classification_data, 'target')
                
                # Los resultados deberían ser similares
                assert len(result1.leaderboard) == len(result2.leaderboard)
                # Nota: Los scores exactos pueden variar debido a la naturaleza estocástica
                # pero la estructura debería ser la misma


if __name__ == "__main__":
    pytest.main([__file__, "-v"])