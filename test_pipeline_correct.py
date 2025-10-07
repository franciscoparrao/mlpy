"""Test correct pipeline usage"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Crear datos
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
df['target'] = y

# Importar componentes
from mlpy.tasks import TaskClassif
from mlpy.learners import learner_sklearn
from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline, GraphLearner
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample
from mlpy.resamplings import ResamplingCV

# Crear task
task = TaskClassif(data=df, target='target')

# Crear learner
rf = learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf')

# Opción 1: Pipeline como Graph
graph = linear_pipeline(
    PipeOpScale(id='scale'),
    PipeOpLearner(rf, id='learner')
)
print("Tipo de linear_pipeline():", type(graph))

# Opción 2: Crear GraphLearner del Graph
pipeline_learner = GraphLearner(graph, id='pipeline')
print("Tipo de GraphLearner:", type(pipeline_learner))
print("pipeline_learner.id:", pipeline_learner.id)

# Ahora sí debería funcionar resample
result = resample(
    task=task,
    learner=pipeline_learner,
    resampling=ResamplingCV(folds=3),
    measures=MeasureClassifAccuracy()
)

print("\nResample exitoso!")
scores = result.aggregate()
print("Resultados:", scores)