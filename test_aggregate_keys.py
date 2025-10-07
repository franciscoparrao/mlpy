"""Test para ver qué keys devuelve aggregate()"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Crear datos
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(5)])
df['target'] = y

# Crear componentes
from mlpy.tasks import TaskClassif
from mlpy.learners import learner_sklearn
from mlpy.measures import MeasureClassifAccuracy
from mlpy import resample
from mlpy.resamplings import ResamplingCV

task = TaskClassif(data=df, target='target')
learner = learner_sklearn(RandomForestClassifier(n_estimators=10), id='rf')
measure = MeasureClassifAccuracy()

# Resample
result = resample(
    task=task,
    learner=learner,
    resampling=ResamplingCV(folds=3),
    measures=measure
)

# Ver resultado
print("Tipo de result:", type(result))
print("\nAtributos de result:")
for attr in dir(result):
    if not attr.startswith('_'):
        print(f"  - {attr}")

print("\nContenido de result.scores:")
print(result.scores)

print("\nResultado de aggregate():")
agg = result.aggregate()
print("Tipo:", type(agg))
print("Keys:", list(agg.keys()) if hasattr(agg, 'keys') else 'No keys')
print("Contenido:", agg)