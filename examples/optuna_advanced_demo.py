"""
Demo avanzado de Optuna con MLPY.

Muestra características avanzadas como:
- Optimización multiobjetivo
- Pruning (early stopping)
- Callbacks personalizados
- Integración con pipelines
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import time

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.learners import learner_sklearn
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC
from mlpy.resamplings import ResamplingCV
from mlpy.pipelines import PipeOpScale, PipeOpSelect, PipeOpLearner, linear_pipeline, GraphLearner

print("="*60)
print("DEMO AVANZADO DE OPTUNA CON MLPY")
print("="*60)

# Crear dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=50, 
    n_informative=20,
    n_redundant=10,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['target'] = y

task = TaskClassif(data=df, target='target')
print(f"\nDataset: {task.nrow} muestras, {len(task.feature_names)} características, {task.n_classes} clases")

try:
    import optuna
    from mlpy.tuning import OptunaTuner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("\n⚠️ Optuna no está instalado. Instálalo con: pip install optuna")

if OPTUNA_AVAILABLE:
    # Ejemplo 1: Optimización con Pruning
    print("\n" + "="*60)
    print("EJEMPLO 1: Optimización con Pruning")
    print("="*60)
    
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Configurar pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1
    )
    
    # Espacio de búsqueda para GBM
    gbm_space = {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0}
    }
    
    gbm_learner = learner_sklearn(GradientBoostingClassifier(random_state=42))
    
    print("\nOptimizando con pruning para detener trials no prometedores...")
    tuner_pruning = OptunaTuner(
        learner=gbm_learner,
        search_space=gbm_space,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        n_trials=30,
        pruner=pruner,
        show_progress_bar=True
    )
    
    start_time = time.time()
    tuner_pruning.tune(task)
    elapsed = time.time() - start_time
    
    print(f"\nTiempo de optimización: {elapsed:.2f} segundos")
    print(f"Trials completados: {len(tuner_pruning.study.trials)}")
    pruned = sum(1 for t in tuner_pruning.study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"Trials podados (pruned): {pruned}")
    print(f"Mejor accuracy: {tuner_pruning.best_score_:.4f}")
    
    # Ejemplo 2: Optimización de Pipeline Completo
    print("\n" + "="*60)
    print("EJEMPLO 2: Optimización de Pipeline ML")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    def pipeline_search_space(trial):
        """Espacio de búsqueda para pipeline completo."""
        # Decisiones de pipeline
        use_scaling = trial.suggest_categorical('use_scaling', [True, False])
        feature_selection = trial.suggest_categorical('feature_selection', ['none', 'k_best', 'percentile'])
        classifier = trial.suggest_categorical('classifier', ['rf', 'svm'])
        
        params = {}
        
        # Parámetros de selección de características
        if feature_selection == 'k_best':
            params['k_features'] = trial.suggest_int('k_features', 5, 30)
        elif feature_selection == 'percentile':
            params['percentile'] = trial.suggest_int('percentile', 10, 90)
            
        # Parámetros del clasificador
        if classifier == 'rf':
            params['n_estimators'] = trial.suggest_int('rf_n_estimators', 50, 200)
            params['max_depth'] = trial.suggest_int('rf_max_depth', 5, 20)
        else:  # svm
            params['C'] = trial.suggest_float('svm_C', 0.01, 100, log=True)
            params['gamma'] = trial.suggest_float('svm_gamma', 0.001, 1, log=True)
            
        # Construir pipeline basado en decisiones
        ops = []
        
        if use_scaling:
            ops.append(PipeOpScale(method='standard'))
            
        if feature_selection == 'k_best':
            ops.append(PipeOpSelect(k=params.get('k_features', 10), method='f_classif'))
        elif feature_selection == 'percentile':
            k = int(task.n_features * params.get('percentile', 50) / 100)
            ops.append(PipeOpSelect(k=k, method='f_classif'))
            
        if classifier == 'rf':
            learner = learner_sklearn(RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42
            ))
        else:
            learner = learner_sklearn(SVC(
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            ))
            
        ops.append(PipeOpLearner(learner))
        
        # Crear pipeline
        graph = linear_pipeline(*ops)
        pipeline = GraphLearner(graph, id=f"pipeline_trial_{trial.number}")
        
        return pipeline
    
    # Crear tuner especial para pipelines
    class PipelineOptunaTuner(OptunaTuner):
        """Tuner especial para optimizar pipelines completos."""
        
        def _objective(self, trial):
            # Obtener pipeline para este trial
            pipeline = self.search_space(trial)
            
            # Evaluar
            try:
                from mlpy import resample
                result = resample(
                    task=self._task,
                    learner=pipeline,
                    resampling=self.resampling,
                    measures=self.measure
                )
                
                scores = result.score(measures=self.measure)
                score = scores.mean()
                
                # Guardar configuración
                self.results_.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'score': score,
                    'pipeline': str(pipeline.graph)
                })
                
                return score
                
            except Exception as e:
                print(f"Trial {trial.number} failed: {e}")
                return float('inf') if self.direction == 'minimize' else float('-inf')
    
    print("\nOptimizando pipeline completo...")
    
    # Usar tuner modificado
    tuner_pipeline = PipelineOptunaTuner(
        learner=None,  # No se usa, pipeline se crea en search_space
        search_space=pipeline_search_space,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        n_trials=20,
        show_progress_bar=True
    )
    
    tuner_pipeline.tune(task)
    
    print(f"\nMejor configuración de pipeline:")
    for param, value in tuner_pipeline.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Mejor accuracy: {tuner_pipeline.best_score_:.4f}")
    
    # Ejemplo 3: Callbacks y Monitoreo
    print("\n" + "="*60)
    print("EJEMPLO 3: Callbacks Personalizados")
    print("="*60)
    
    # Crear callback para monitorear progreso
    best_value_history = []
    
    def track_best_value(study, trial):
        """Callback para rastrear mejor valor."""
        best_value_history.append(study.best_value)
        if trial.number % 5 == 0:
            print(f"  Trial {trial.number}: Mejor valor actual = {study.best_value:.4f}")
    
    # Optimización con callback
    from sklearn.neural_network import MLPClassifier
    
    mlp_space = {
        'hidden_layer_sizes': {
            'type': 'categorical',
            'choices': [(50,), (100,), (50, 50), (100, 50)]
        },
        'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
        'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True}
    }
    
    mlp_learner = learner_sklearn(MLPClassifier(max_iter=500, random_state=42))
    
    print("\nOptimizando con monitoreo de progreso...")
    tuner_callback = OptunaTuner(
        learner=mlp_learner,
        search_space=mlp_space,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAUC(),
        n_trials=20,
        show_progress_bar=False  # Desactivar para ver callbacks
    )
    
    # Añadir callback al study
    tuner_callback.study.optimize(
        tuner_callback._objective,
        n_trials=20,
        callbacks=[track_best_value]
    )
    
    # Ejemplo 4: Optimización Multiobjetivo
    print("\n" + "="*60)
    print("EJEMPLO 4: Optimización Multiobjetivo")
    print("="*60)
    
    print("\n⚠️ Optimización multiobjetivo requiere modificación del tuner base.")
    print("Concepto: Optimizar accuracy Y tiempo de entrenamiento simultáneamente")
    
    print("""
# Código conceptual:
study = optuna.create_study(
    directions=['maximize', 'minimize']  # accuracy, tiempo
)

def multi_objective(trial):
    # Configurar modelo
    params = {...}
    
    # Medir accuracy
    accuracy = evaluate_model(params)
    
    # Medir tiempo
    start = time.time()
    train_model(params)
    train_time = time.time() - start
    
    return accuracy, train_time

study.optimize(multi_objective, n_trials=100)

# Obtener frente de Pareto
pareto_front = study.best_trials
""")
    
    # Visualización de resultados
    print("\n" + "="*60)
    print("ANÁLISIS Y VISUALIZACIÓN")
    print("="*60)
    
    # Historia de optimización
    if best_value_history:
        print("\nProgreso de optimización (mejores valores):")
        for i in range(0, len(best_value_history), 5):
            print(f"  Trial {i}: {best_value_history[i]:.4f}")
    
    # Importancia de hiperparámetros
    try:
        # Obtener importancias del primer estudio
        importance = optuna.importance.get_param_importances(tuner_pruning.study)
        print("\nImportancia de hiperparámetros (GBM):")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.3f}")
    except:
        print("\nNo se pudo calcular importancia de parámetros")

else:
    # Sin Optuna
    print("\nEjemplos de características avanzadas (requiere Optuna):")
    print("- Pruning para detener trials no prometedores")
    print("- Optimización de pipelines completos")
    print("- Callbacks para monitoreo personalizado")
    print("- Optimización multiobjetivo")
    print("- Análisis de importancia de parámetros")

# Resumen
print("\n" + "="*60)
print("CARACTERÍSTICAS AVANZADAS DE OPTUNA + MLPY")
print("="*60)
print("""
1. PRUNING
   - Detiene trials no prometedores temprano
   - Ahorra tiempo computacional
   - Útil para modelos costosos

2. OPTIMIZACIÓN DE PIPELINES
   - Optimiza arquitectura + hiperparámetros
   - Decisiones condicionales
   - Búsqueda en espacios complejos

3. CALLBACKS
   - Monitoreo personalizado
   - Logging avanzado
   - Integración con herramientas externas

4. MULTIOBJETIVO
   - Optimizar múltiples métricas
   - Trade-offs (accuracy vs velocidad)
   - Frente de Pareto

5. ANÁLISIS
   - Importancia de parámetros
   - Visualizaciones interactivas
   - Exportación de resultados
""")