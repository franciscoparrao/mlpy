"""
Demo simple de XGBoost con MLPY
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
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample, benchmark

# Importar XGBoost
from xgboost import XGBClassifier

print("="*60)
print("DEMO SIMPLE: XGBoost con MLPY") 
print("="*60)

# 1. Crear datos
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
df['target'] = y

# 2. Crear tarea MLPY
task = TaskClassif(data=df, target='target')
print(f"\nTarea: {task.nrow} muestras, {len(task.feature_names)} caracteristicas")

# 3. Crear learner XGBoost
xgb_learner = learner_sklearn(
    XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    ),
    id="xgboost"
)

print("\n--- ENTRENAMIENTO Y PREDICCION ---")
# 4. Entrenar
xgb_learner.train(task)
print("Modelo entrenado!")

# 5. Predecir
pred = xgb_learner.predict(task)
accuracy = np.mean(pred.response == pred.truth)
print(f"Accuracy en datos de entrenamiento: {accuracy:.3f}")

print("\n--- EVALUACION CON CROSS-VALIDATION ---")
# 6. Evaluar con CV
result = resample(
    task=task,
    learner=xgb_learner,
    resampling=ResamplingCV(folds=5),
    measures=MeasureClassifAccuracy()
)

scores = result.aggregate()
print(f"\nResultados CV (5 folds):")
print(f"Accuracy promedio: {scores['mean'].iloc[0]:.3f}")
print(f"Desviacion estandar: {scores['std'].iloc[0]:.3f}")

print("\n--- COMPARACION CON OTROS MODELOS ---")
# 7. Comparar con otros modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

learners = [
    learner_sklearn(XGBClassifier(n_estimators=100, random_state=42), id="XGBoost"),
    learner_sklearn(RandomForestClassifier(n_estimators=100, random_state=42), id="RandomForest"),
    learner_sklearn(LogisticRegression(max_iter=1000), id="LogisticReg")
]

bench = benchmark(
    tasks=[task],
    learners=learners,
    resampling=ResamplingCV(folds=3),
    measures=MeasureClassifAccuracy()
)

print("\nBenchmark resultados:")
print(bench.aggregate("classif.acc").to_string())

print("\n--- USO EN PIPELINE ---")
# 8. XGBoost en pipeline
from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline, GraphLearner

# Pipeline: Escalar -> XGBoost
xgb_model = learner_sklearn(XGBClassifier(n_estimators=50, random_state=42))
graph = linear_pipeline(
    PipeOpScale(method="standard"),
    PipeOpLearner(xgb_model)
)
pipeline = GraphLearner(graph, id="xgb_pipeline")

# Evaluar pipeline
pipe_result = resample(
    task=task,
    learner=pipeline,
    resampling=ResamplingCV(folds=3),
    measures=MeasureClassifAccuracy()
)

pipe_scores = pipe_result.aggregate()
print(f"\nPipeline (Scale + XGBoost) accuracy: {pipe_scores['mean'].iloc[0]:.3f}")

print("\n" + "="*60)
print("RESUMEN: XGBoost funciona perfectamente con MLPY!")
print("="*60)