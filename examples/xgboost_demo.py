"""
Demo de XGBoost con MLPY.

Muestra las dos formas de usar XGBoost:
1. A traves del wrapper sklearn (mas simple)
2. Con el wrapper nativo de XGBoost (mas caracteristicas)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Agregar el path para importar MLPY
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MLPY imports
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners import learner_sklearn
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy, MeasureClassifAUC, MeasureRegrRMSE
from mlpy import resample, benchmark

print("=" * 60)
print("DEMO DE XGBOOST CON MLPY")
print("=" * 60)

# Crear datos de ejemplo
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
df['target'] = y

task = TaskClassif(data=df, target='target')
print(f"\nTarea creada: {task.nrow} muestras, {len(task.feature_names)} características")

# Método 1: Usar XGBoost a través del wrapper sklearn
print("\n" + "="*60)
print("MÉTODO 1: XGBoost via sklearn wrapper")
print("="*60)

try:
    from xgboost import XGBClassifier
    
    # Crear learner con sklearn wrapper
    xgb_sklearn = learner_sklearn(
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.3,
            random_state=42
        ),
        id="xgb_sklearn"
    )
    
    # Evaluar con cross-validation
    print("\n Evaluando XGBoost (sklearn wrapper)...")
    result = resample(
        task=task,
        learner=xgb_sklearn,
        resampling=ResamplingCV(folds=5),
        measures=[MeasureClassifAccuracy(), MeasureClassifAUC()]
    )
    
    scores = result.aggregate()
    print("\n Resultados:")
    print(f"Accuracy: {scores['classif.acc'].mean():.3f} ± {scores['classif.acc'].std():.3f}")
    print(f"AUC: {scores['classif.auc'].mean():.3f} ± {scores['classif.auc'].std():.3f}")
    
except ImportError:
    print("\nXGBoost no esta instalado. Instalalo con: pip install xgboost")
    print("\nEjemplo de código que funcionaría:")
    print("""
    from xgboost import XGBClassifier
    xgb_learner = learner_sklearn(XGBClassifier(n_estimators=100))
    xgb_learner.train(task)
    predictions = xgb_learner.predict(task)
    """)

# Método 2: Usar wrapper nativo de XGBoost
print("\n" + "="*60)
print("MÉTODO 2: XGBoost wrapper nativo (más características)")
print("="*60)

try:
    from mlpy.learners.xgboost_wrapper import learner_xgboost
    
    # Crear learner con wrapper nativo
    xgb_native = learner_xgboost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("\n Entrenando XGBoost nativo...")
    xgb_native.train(task)
    
    # Hacer predicciones
    pred = xgb_native.predict(task)
    accuracy = np.mean(pred.response == pred.truth)
    print(f"\n Accuracy en entrenamiento: {accuracy:.3f}")
    
    # Obtener feature importances
    if xgb_native.feature_importances:
        print("\n Top 5 características más importantes:")
        importance = xgb_native.feature_importances
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.3f}")
    
    # Obtener SHAP values
    print("\n Calculando SHAP values...")
    shap_values = xgb_native.get_shap_values(task, row_ids=[0, 1, 2, 3, 4])
    print(f"SHAP values shape: {shap_values.shape}")
    
except ImportError:
    print("\n️ El wrapper nativo requiere XGBoost instalado")
    print("Pero las características adicionales incluirían:")
    print("- Feature importances")
    print("- SHAP values nativos")
    print("- Plot de árboles")
    print("- Early stopping")

# Comparación con otros modelos
print("\n" + "="*60)
print("COMPARACIÓN: XGBoost vs Otros Modelos")
print("="*60)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

learners = [
    learner_sklearn(LogisticRegression(max_iter=1000), id="logistic"),
    learner_sklearn(RandomForestClassifier(n_estimators=100, random_state=42), id="random_forest"),
    learner_sklearn(GradientBoostingClassifier(n_estimators=100, random_state=42), id="gradient_boost"),
]

# Añadir XGBoost si está disponible
try:
    from xgboost import XGBClassifier
    learners.append(
        learner_sklearn(XGBClassifier(n_estimators=100, random_state=42), id="xgboost")
    )
except ImportError:
    pass

print("\n Ejecutando benchmark...")
bench_result = benchmark(
    tasks=[task],
    learners=learners,
    resampling=ResamplingCV(folds=3),
    measures=MeasureClassifAccuracy()
)

print("\n Ranking de modelos:")
print(bench_result.aggregate())

# Ejemplo con regresión
print("\n" + "="*60)
print("EJEMPLO DE REGRESIÓN CON XGBOOST")
print("="*60)

# Crear datos de regresión
X_reg, y_reg = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
df_reg = pd.DataFrame(X_reg, columns=[f"x{i}" for i in range(X_reg.shape[1])])
df_reg['y'] = y_reg

task_reg = TaskRegr(data=df_reg, target='y')

try:
    from xgboost import XGBRegressor
    
    xgb_reg = learner_sklearn(
        XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    )
    
    result_reg = resample(
        task=task_reg,
        learner=xgb_reg,
        resampling=ResamplingCV(folds=3),
        measures=MeasureRegrRMSE()
    )
    
    scores_reg = result_reg.aggregate()
    print(f"\nRMSE: {scores_reg['regr.rmse'].mean():.3f} ± {scores_reg['regr.rmse'].std():.3f}")
    
except ImportError:
    print("\n️ XGBoost no instalado para ejemplo de regresión")

# Uso en pipelines
print("\n" + "="*60)
print("XGBOOST EN PIPELINES MLPY")
print("="*60)

from mlpy.pipelines import PipeOpScale, PipeOpSelect, PipeOpLearner, linear_pipeline, GraphLearner

try:
    from xgboost import XGBClassifier
    
    # Crear pipeline: Escalar -> Seleccionar features -> XGBoost
    xgb_model = learner_sklearn(XGBClassifier(n_estimators=50))
    
    graph = linear_pipeline(
        PipeOpScale(method="standard"),
        PipeOpSelect(k=10, method="f_classif"),
        PipeOpLearner(xgb_model)
    )
    
    pipeline = GraphLearner(graph, id="xgb_pipeline")
    
    print("\n Pipeline creado: Scale -> Select(10) -> XGBoost")
    print(" Evaluando pipeline...")
    
    pipe_result = resample(
        task=task,
        learner=pipeline,
        resampling=ResamplingCV(folds=3),
        measures=MeasureClassifAccuracy()
    )
    
    pipe_scores = pipe_result.aggregate()
    print(f"\nAccuracy del pipeline: {pipe_scores['classif.acc'].mean():.3f}")
    
except ImportError:
    print("\n️ XGBoost no instalado para ejemplo de pipeline")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print("""
 XGBoost se puede usar con MLPY de dos formas:

1. **Via sklearn wrapper** (más simple):
   - from mlpy.learners import learner_sklearn
   - from xgboost import XGBClassifier, XGBRegressor
   - learner = learner_sklearn(XGBClassifier())

2. **Via wrapper nativo** (más características):
   - from mlpy.learners.xgboost_wrapper import learner_xgboost
   - learner = learner_xgboost(n_estimators=100)
   - Acceso a SHAP values, feature importance, etc.

Ambos métodos se integran perfectamente con:
- Resampling (CV, Bootstrap, etc.)
- Benchmarking
- Pipelines
- Medidas de evaluación
- Persistencia
""")