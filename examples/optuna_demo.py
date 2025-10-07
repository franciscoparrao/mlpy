"""
Demo de integración de Optuna con MLPY.

Muestra cómo usar Optuna para optimización de hiperparámetros
con learners de MLPY.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Path para MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.tasks import TaskClassif
from mlpy.learners import learner_sklearn
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC
from mlpy.resamplings import ResamplingCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("="*60)
print("DEMO DE OPTUNA CON MLPY")
print("="*60)

# Crear dataset
X, y = make_classification(
    n_samples=500, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['target'] = y

task = TaskClassif(data=df, target='target')
print(f"\nDataset: {task.nrow} muestras, {len(task.feature_names)} características")

# Verificar si Optuna está instalado
try:
    from mlpy.tuning import OptunaTuner, tune_learner
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("\n⚠️ Optuna no está instalado. Instálalo con: pip install optuna")

if OPTUNA_AVAILABLE:
    # Ejemplo 1: Optimizar Random Forest
    print("\n" + "="*60)
    print("EJEMPLO 1: Optimizar Random Forest")
    print("="*60)
    
    # Definir espacio de búsqueda
    rf_search_space = {
        'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
        'max_depth': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
        'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Crear learner base
    rf_learner = learner_sklearn(RandomForestClassifier(random_state=42))
    
    # Configurar tuner
    print("\nConfigurando Optuna tuner...")
    tuner = OptunaTuner(
        learner=rf_learner,
        search_space=rf_search_space,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        n_trials=20,  # Pocas pruebas para el demo
        show_progress_bar=True
    )
    
    # Ejecutar optimización
    print("\nOptimizando hiperparámetros...")
    tuner.tune(task)
    
    # Resultados
    print(f"\nMejores parámetros encontrados:")
    for param, value in tuner.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor accuracy: {tuner.best_score_:.4f}")
    
    # Obtener mejor learner
    best_rf = tuner.get_best_learner()
    
    # Ejemplo 2: Usar función de conveniencia
    print("\n" + "="*60)
    print("EJEMPLO 2: Función tune_learner() simplificada")
    print("="*60)
    
    # Optimizar SVM
    svm_search_space = {
        'C': {'type': 'float', 'low': 0.01, 'high': 100, 'log': True},
        'gamma': {'type': 'float', 'low': 0.001, 'high': 1, 'log': True},
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    svm_learner = learner_sklearn(SVC(probability=True, random_state=42))
    
    print("\nOptimizando SVM con tune_learner()...")
    best_svm, best_params, best_score = tune_learner(
        learner=svm_learner,
        task=task,
        search_space=svm_search_space,
        measure=MeasureClassifAUC(),
        n_trials=15
    )
    
    print(f"\nMejores parámetros SVM:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"\nMejor AUC: {best_score:.4f}")
    
    # Ejemplo 3: Espacio de búsqueda con función
    print("\n" + "="*60)
    print("EJEMPLO 3: Espacio de búsqueda dinámico")
    print("="*60)
    
    def dynamic_search_space(trial):
        """Espacio de búsqueda con dependencias."""
        # Primero elegir el tipo de modelo
        use_deep_trees = trial.suggest_categorical('use_deep_trees', [True, False])
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300)
        }
        
        if use_deep_trees:
            # Árboles profundos con menos muestras por hoja
            params['max_depth'] = trial.suggest_int('max_depth', 10, 30)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 5)
        else:
            # Árboles poco profundos con más muestras
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 5, 20)
            
        return params
    
    rf_dynamic = learner_sklearn(RandomForestClassifier(random_state=42))
    
    tuner_dynamic = OptunaTuner(
        learner=rf_dynamic,
        search_space=dynamic_search_space,
        resampling=ResamplingCV(folds=3),
        measure=MeasureClassifAccuracy(),
        n_trials=15
    )
    
    print("\nOptimizando con espacio dinámico...")
    tuner_dynamic.tune(task)
    
    print(f"\nMejores parámetros (dinámico):")
    for param, value in tuner_dynamic.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nMejor accuracy: {tuner_dynamic.best_score_:.4f}")
    
    # Mostrar resultados completos
    print("\n" + "="*60)
    print("ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    # DataFrame con todos los resultados
    results_df = tuner.get_results_df()
    print("\nTop 5 configuraciones de Random Forest:")
    print(results_df.head())
    
    # Visualizaciones (si plotly está instalado)
    try:
        print("\nGenerando visualizaciones...")
        
        # Historia de optimización
        fig1 = tuner.plot_optimization_history()
        if fig1:
            print("  - Historia de optimización generada")
            
        # Importancia de parámetros
        fig2 = tuner.plot_param_importances()
        if fig2:
            print("  - Importancia de parámetros generada")
            
    except Exception as e:
        print(f"  No se pudieron generar visualizaciones: {e}")

else:
    # Código de ejemplo sin Optuna
    print("\n" + "="*60)
    print("CÓDIGO DE EJEMPLO (sin Optuna instalado)")
    print("="*60)
    
    print("""
# Instalar Optuna:
pip install optuna

# Uso básico:
from mlpy.tuning import OptunaTuner, tune_learner

# 1. Definir espacio de búsqueda
search_space = {
    'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
    'max_depth': {'type': 'int', 'low': 2, 'high': 20}
}

# 2. Crear tuner
tuner = OptunaTuner(
    learner=learner,
    search_space=search_space,
    n_trials=50
)

# 3. Optimizar
tuner.tune(task)

# 4. Obtener mejor modelo
best_learner = tuner.get_best_learner()
print(f"Mejores parámetros: {tuner.best_params_}")
""")

# Comparación con otros métodos
print("\n" + "="*60)
print("COMPARACIÓN: Grid Search vs Random Search vs Optuna")
print("="*60)

from mlpy.tuning import TunerGridSearch, TunerRandomSearch

# Espacio pequeño para comparación rápida
small_space = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15]
}

print("\n1. Grid Search:")
print("   - Prueba TODAS las combinaciones (3 x 3 = 9)")
print("   - Garantiza encontrar el mejor en el grid")
print("   - Costoso para espacios grandes")

print("\n2. Random Search:")
print("   - Prueba combinaciones aleatorias") 
print("   - Más eficiente que grid para espacios grandes")
print("   - No garantiza el óptimo global")

print("\n3. Optuna (Bayesian Optimization):")
print("   - Aprende de pruebas anteriores")
print("   - Enfoca búsqueda en regiones prometedoras")
print("   - Más eficiente para espacios complejos")
print("   - Soporta pruning (early stopping)")

# Resumen
print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print("""
Optuna + MLPY permite:
✓ Optimización eficiente de hiperparámetros
✓ Integración transparente con learners MLPY
✓ Soporte para sklearn, XGBoost, H2O, etc.
✓ Visualizaciones interactivas
✓ Optimización distribuida
✓ Pruning para ahorrar tiempo

Casos de uso:
- Encontrar mejores hiperparámetros
- Comparar diferentes configuraciones
- Optimización con recursos limitados
- Búsqueda en espacios complejos
""")