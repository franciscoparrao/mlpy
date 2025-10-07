# Plan de Trabajo - MLPY (MLR3 en Python)

## Objetivo del Proyecto
Crear un framework de Machine Learning en Python que replique la arquitectura y filosofía de diseño de mlr3 (R), proporcionando una API unificada, modular y extensible para tareas de ML.

## Arquitectura Core a Implementar

### 1. Fundamentos del Framework (Fase 1)
#### 1.1 Estructura Base del Proyecto
- **Setup inicial del proyecto**
  - Configurar estructura de directorios (`mlpy/`, `tests/`, `docs/`, `examples/`)
  - Configurar `pyproject.toml` con Poetry o `setup.py`
  - Configurar pre-commit hooks (black, flake8, mypy)
  - Configurar GitHub Actions para CI/CD
  - Configurar documentación con Sphinx

#### 1.2 Sistema de Reflections y Registry
- **Implementar sistema de registros globales**
  - `mlpy_tasks`: Registro de tareas disponibles
  - `mlpy_learners`: Registro de learners
  - `mlpy_measures`: Registro de métricas
  - `mlpy_resamplings`: Registro de estrategias de resampling
  - Sistema de auto-registro mediante decoradores

#### 1.3 Clases Base y Utilidades
- **Implementar clase base abstracta común**
  - Sistema de hashing único para objetos
  - Sistema de parámetros (param_set)
  - Sistema de validación y assertions
  - Métodos de clonación profunda
  - Sistema de logging integrado

### 2. Sistema de Datos (Fase 2)
#### 2.1 DataBackend
- **Implementar clase abstracta DataBackend**
  - Métodos abstractos: `data()`, `head()`, `nrow()`, `ncol()`, `colnames()`
  - Sistema de tipos de columnas
  - Sistema de missing values

- **Implementaciones concretas**
  - `DataBackendPandas`: Backend con pandas DataFrame
  - `DataBackendNumPy`: Backend con arrays NumPy
  - `DataBackendPolars`: Backend con Polars (opcional)
  - `DataBackendArrow`: Backend con PyArrow (opcional)

#### 2.2 Task
- **Implementar clase abstracta Task**
  - Constructor con validación de datos
  - Sistema de roles de columnas (feature, target, weight, etc.)
  - Métodos de filtrado y selección
  - Sistema de metadatos (descripción, fuente, etc.)
  - Propiedades calculadas (nrow, ncol, feature_names, etc.)

- **Implementaciones concretas**
  - `TaskClassif`: Tareas de clasificación
    - Manejo de clases (binarias/multiclase)
    - Balance de clases
    - Estratificación
  - `TaskRegr`: Tareas de regresión
  - `TaskUnsupervised`: Para clustering/reducción dimensionalidad

#### 2.3 TaskGenerator
- **Sistema de generación sintética de datos**
  - `TaskGenerator2DNormals`
  - `TaskGeneratorMoons`
  - `TaskGeneratorSpirals`
  - `TaskGeneratorFriedman1`

### 3. Sistema de Modelos (Fase 3)
#### 3.1 Learner
- **Implementar clase abstracta Learner**
  - Métodos abstractos: `_train()`, `_predict()`
  - Sistema de tipos de predicción (response, prob, etc.)
  - Sistema de propiedades (soporta weights, missing values, etc.)
  - Sistema de fallback para errores
  - Encapsulación de errores y timeouts

- **Learners básicos**
  - `LearnerClassifBaseline`: Predictor baseline (mayoría, aleatorio)
  - `LearnerRegrBaseline`: Predictor baseline (media, mediana)
  - `LearnerClassifDebug`: Para testing
  - `LearnerRegrDebug`: Para testing

#### 3.2 Integración con scikit-learn
- **Wrapper genérico para scikit-learn**
  - `LearnerClassifSklearn`: Wrapper para clasificadores sklearn
  - `LearnerRegrSklearn`: Wrapper para regresores sklearn
  - Auto-detección de propiedades
  - Conversión de parámetros

#### 3.3 Sistema de Predicciones
- **Implementar clases de Prediction**
  - `PredictionClassif`: Predicciones de clasificación
    - Probabilidades
    - Clases predichas
    - Matriz de confusión
  - `PredictionRegr`: Predicciones de regresión
    - Valores predichos
    - Intervalos de confianza (opcional)

### 4. Sistema de Evaluación (Fase 4)
#### 4.1 Measure
- **Implementar clase abstracta Measure**
  - Método abstracto: `_score()`
  - Sistema de agregación
  - Propiedades (minimizar/maximizar, rango, etc.)

- **Métricas de clasificación**
  - Accuracy, Precision, Recall, F1
  - AUC, LogLoss, Brier Score
  - MCC, Cohen's Kappa
  - Métricas multiclase (macro/micro/weighted)

- **Métricas de regresión**
  - MSE, RMSE, MAE, MAPE
  - R², R² ajustado
  - Pinball loss (cuantiles)

#### 4.2 Resampling
- **Implementar clase abstracta Resampling**
  - Método `instantiate()` para fijar particiones
  - Métodos `train_set()`, `test_set()`
  - Soporte para estratificación

- **Estrategias de resampling**
  - `ResamplingCV`: Cross-validation
  - `ResamplingRepeatedCV`: CV repetido
  - `ResamplingHoldout`: Train/test split
  - `ResamplingBootstrap`: Bootstrap sampling
  - `ResamplingLOO`: Leave-one-out
  - `ResamplingCustom`: Particiones personalizadas

### 5. Sistema de Ejecución (Fase 5)
#### 5.1 Funciones principales
- **resample()**
  - Ejecutar learner con resampling strategy
  - Recolectar predicciones y métricas
  - Manejo de errores y fallbacks
  - Callbacks para progreso

- **benchmark()**
  - Evaluar múltiples learners en múltiples tasks
  - Grid de experimentos
  - Paralelización
  - Almacenamiento de resultados

#### 5.2 ResampleResult y BenchmarkResult
- **ResampleResult**
  - Almacenar resultados de un resample
  - Métodos de agregación
  - Visualizaciones

- **BenchmarkResult**
  - Almacenar resultados de benchmark
  - Comparaciones estadísticas
  - Rankings
  - Visualizaciones

### 6. Sistemas Avanzados (Fase 6)
#### 6.1 Pipelines y Graphs
- **PipeOp (Pipeline Operators)**
  - Operadores básicos (scale, impute, encode, etc.)
  - Sistema de composición

- **GraphLearner**
  - DAG de operaciones
  - Optimización de hiperparámetros en pipeline

#### 6.2 AutoML básico
- **Tuning**
  - Grid search
  - Random search
  - Bayesian optimization (opcional)

- **Feature Engineering automático**
  - Selección de características
  - Generación de características

#### 6.3 Paralelización
- **Backends de paralelización**
  - Threading
  - Multiprocessing
  - Joblib
  - Ray (opcional)
  - Dask (opcional)

### 7. Extensiones y Extras (Fase 7)
#### 7.1 Callbacks y Hooks
- Sistema de callbacks para eventos
- Logging personalizado
- Early stopping
- Checkpointing

#### 7.2 Visualización
- Plots de rendimiento
- Curvas ROC/PR
- Importancia de características
- Diagnósticos de modelo

#### 7.3 Interpretabilidad
- SHAP integration
- LIME integration
- Permutation importance

### 8. Testing y Documentación (Continuo)
#### 8.1 Testing
- Unit tests para cada componente
- Integration tests
- Property-based testing con Hypothesis
- Benchmarks de rendimiento

#### 8.2 Documentación
- Docstrings completos (formato Google/NumPy)
- Guía de usuario
- Tutoriales
- Ejemplos de uso
- API reference

#### 8.3 Ejemplos y Demos
- Jupyter notebooks con casos de uso
- Scripts de ejemplo
- Comparaciones con scikit-learn
- Casos de uso avanzados

## Principios de Diseño

1. **API Consistente**: Todos los objetos siguen patrones similares
2. **Composabilidad**: Los componentes se pueden combinar fácilmente
3. **Extensibilidad**: Fácil agregar nuevos learners, measures, etc.
4. **Type Safety**: Uso de type hints y validación en runtime
5. **Inmutabilidad**: Operaciones no modifican objetos in-place
6. **Lazy Evaluation**: Cálculos se realizan solo cuando es necesario
7. **Error Handling**: Manejo robusto de errores con fallbacks

## Tecnologías y Dependencias

### Core
- Python >= 3.8
- NumPy
- Pandas
- scikit-learn (para integración)

### Opcionales
- Polars (backend alternativo)
- PyArrow (backend alternativo)
- Joblib (paralelización)
- Ray/Dask (paralelización distribuida)
- Matplotlib/Seaborn (visualización)
- SHAP/LIME (interpretabilidad)

### Desarrollo
- pytest (testing)
- black (formateo)
- mypy (type checking)
- Sphinx (documentación)
- pre-commit (hooks)

## Cronograma Estimado

- **Fase 1**: 2-3 semanas - Fundamentos
- **Fase 2**: 3-4 semanas - Sistema de Datos
- **Fase 3**: 3-4 semanas - Sistema de Modelos
- **Fase 4**: 2-3 semanas - Sistema de Evaluación
- **Fase 5**: 2-3 semanas - Sistema de Ejecución
- **Fase 6**: 4-6 semanas - Sistemas Avanzados
- **Fase 7**: 2-3 semanas - Extensiones

**Total estimado**: 4-6 meses para versión 1.0

## Criterios de Éxito

1. API pythonica y fácil de usar
2. Performance comparable a scikit-learn
3. 100% de cobertura de tests en core
4. Documentación completa
5. Comunidad activa de usuarios
6. Integración fluida con ecosistema Python ML