# Tutorial 1: Tu Primer Modelo en 5 Minutos ⚡

## 🎯 Objetivo
Entrenar un clasificador que prediga especies de flores iris usando MLPY.

**Tiempo:** 5 minutos  
**Nivel:** 🟢 Principiante  
**Lo que aprenderás:** Tasks, Learners, validación básica

---

## 🚀 Setup (30 segundos)

```python
# Instalar si no lo tienes
# pip install mlpy-framework[full]

import mlpy
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print(f"MLPY Version: {mlpy.__version__}")
```

---

## 📊 Paso 1: Cargar Datos (30 segundos)

```python
# Cargar el famoso dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

print("Dataset cargado:")
print(f"  Filas: {len(df)}")
print(f"  Columnas: {list(df.columns)}")
print(f"  Especies: {df['species'].unique()}")

# Ver primeras filas
df.head()
```

**Output esperado:**
```
Dataset cargado:
  Filas: 150
  Columnas: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']
  Especies: ['setosa' 'versicolor' 'virginica']
```

---

## ✅ Paso 2: Validación Inteligente (30 segundos)

```python
from mlpy.validation import validate_task_data

# MLPY valida tus datos ANTES de entrenar
validation = validate_task_data(df, target='species')

if validation['valid']:
    print("✅ Datos válidos para ML")
else:
    print("❌ Problemas encontrados:")
    for error in validation['errors']:
        print(f"  - {error}")

# Mostrar warnings (si los hay)
if validation['warnings']:
    print("\n⚠️ Advertencias:")
    for warning in validation['warnings']:
        print(f"  - {warning}")
```

**Output esperado:**
```
✅ Datos válidos para ML
```

---

## 📝 Paso 3: Crear Tarea MLPY (30 segundos)

```python
from mlpy.tasks import TaskClassif

# Crear tarea de clasificación
task = TaskClassif(
    data=df,
    target='species',
    id='iris_classifier'
)

print(f"Tarea creada:")
print(f"  ID: {task.id}")
print(f"  Tipo: {task.task_type}")
print(f"  Clases: {task.n_classes} ({task.y.unique()})")
print(f"  Features: {task.n_features}")
print(f"  Muestras: {task.n_obs}")
```

**Output esperado:**
```
Tarea creada:
  ID: iris_classifier
  Tipo: classif
  Clases: 3 (['setosa' 'versicolor' 'virginica'])
  Features: 4
  Muestras: 150
```

---

## 🔄 Paso 4: Dividir Datos (30 segundos)

```python
# Dividir en train/test
train_idx, test_idx = train_test_split(
    range(len(df)), 
    test_size=0.3, 
    random_state=42,
    stratify=df['species']  # Mantener proporción de clases
)

task_train = task.subset(train_idx)
task_test = task.subset(test_idx)

print(f"División de datos:")
print(f"  Train: {len(train_idx)} muestras")
print(f"  Test: {len(test_idx)} muestras")
```

**Output esperado:**
```
División de datos:
  Train: 105 muestras  
  Test: 45 muestras
```

---

## 🤖 Paso 5: Entrenar Modelo (1 minuto)

```python
from mlpy.learners import LearnerClassifSklearn
from sklearn.ensemble import RandomForestClassifier

# Crear learner con Random Forest
learner = LearnerClassifSklearn(
    estimator=RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
)

print("Entrenando modelo...")
learner.train(task_train)
print("✅ Modelo entrenado!")
```

**Output esperado:**
```
Entrenando modelo...
✅ Modelo entrenado!
```

---

## 📊 Paso 6: Evaluar (1 minuto)

```python
from mlpy.measures import MeasureAccuracy
import numpy as np

# Hacer predicciones
print("Haciendo predicciones...")
predictions = learner.predict(task_test)

# Calcular accuracy
measure = MeasureAccuracy()
accuracy = measure.score(predictions)

print(f"\n📊 Resultados:")
print(f"  Accuracy: {accuracy:.2%}")

# Ver algunas predicciones vs realidad
print(f"\nPrimeras 5 predicciones:")
for i in range(5):
    real = task_test.y.iloc[i]
    pred = predictions.response[i]
    emoji = "✅" if real == pred else "❌"
    print(f"  {emoji} Real: {real:12} | Predicho: {pred}")
```

**Output esperado:**
```
Haciendo predicciones...

📊 Resultados:
  Accuracy: 95.56%

Primeras 5 predicciones:
  ✅ Real: versicolor   | Predicho: versicolor
  ✅ Real: setosa       | Predicho: setosa
  ✅ Real: virginica    | Predicho: virginica
  ✅ Real: versicolor   | Predicho: versicolor
  ✅ Real: versicolor   | Predicho: versicolor
```

---

## 💾 Paso 7: Guardar Modelo (30 segundos)

```python
from mlpy.serialization import RobustSerializer

# Guardar con metadata
serializer = RobustSerializer()
metadata = {
    'accuracy': accuracy,
    'algorithm': 'RandomForest',
    'dataset': 'iris',
    'date': '2024-01-15'
}

result = serializer.save(
    obj=learner,
    path='mi_primer_modelo.pkl',
    metadata=metadata
)

print(f"💾 Modelo guardado:")
print(f"  Archivo: mi_primer_modelo.pkl")
print(f"  Checksum: {result['checksum'][:16]}...")
print(f"  Metadata: {len(metadata)} campos")
```

**Output esperado:**
```
💾 Modelo guardado:
  Archivo: mi_primer_modelo.pkl
  Checksum: a7c3b4f2d8e1...
  Metadata: 4 campos
```

---

## 🎯 ¡Completado!

### 🎉 **¡Felicitaciones! Has entrenado tu primer modelo con MLPY**

**Lo que lograste en 5 minutos:**
- ✅ Validaste datos automáticamente
- ✅ Creaste una tarea de clasificación  
- ✅ Entrenaste un Random Forest
- ✅ Evaluaste con >95% de accuracy
- ✅ Guardaste el modelo de forma segura

---

## 🔍 Análisis Rápido

### ¿Por qué funcionó tan bien?

1. **Dataset balanceado**: 50 muestras por clase
2. **Features informativas**: Medidas físicas discriminan bien las especies
3. **Algoritmo robusto**: Random Forest maneja bien datasets pequeños
4. **Validación previa**: MLPY verificó que los datos fueran apropiados

### Componentes MLPY usados:

- **validate_task_data()**: Prevención proactiva de errores
- **TaskClassif**: Abstracción del problema de clasificación
- **LearnerClassifSklearn**: Wrapper para algoritmos de sklearn
- **RobustSerializer**: Guardado seguro con checksums

---

## 🏋️ Ejercicios de Práctica (Opcional)

### Ejercicio 1: Prueba otros algoritmos
```python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# SVM
learner_svm = LearnerClassifSklearn(estimator=SVC())
learner_svm.train(task_train)
pred_svm = learner_svm.predict(task_test)
acc_svm = MeasureAccuracy().score(pred_svm)
print(f"SVM Accuracy: {acc_svm:.2%}")

# Logistic Regression  
learner_lr = LearnerClassifSklearn(estimator=LogisticRegression())
learner_lr.train(task_train)
pred_lr = learner_lr.predict(task_test)
acc_lr = MeasureAccuracy().score(pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.2%}")
```

### Ejercicio 2: Prueba con datos problemáticos
```python
# Crear datos con problemas
df_bad = df.copy()
df_bad.loc[0, 'sepal length (cm)'] = None  # Introducir NaN
df_bad['constant_feature'] = 1  # Feature constante

# Ver qué dice la validación
validation_bad = validate_task_data(df_bad, target='species')
print("Validación con datos problemáticos:")
print(f"  Válido: {validation_bad['valid']}")
for warning in validation_bad['warnings']:
    print(f"  ⚠️ {warning}")
```

### Ejercicio 3: Cargar y usar el modelo guardado
```python
# Cargar modelo
loaded_learner = serializer.load('mi_primer_modelo.pkl')

# Usar para predicciones
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Datos de una nueva flor
prediction = loaded_learner.estimator.predict(new_flower)
print(f"Nueva predicción: {iris.target_names[prediction[0]]}")
```

---

## 🚀 Siguiente Paso

Has dominado lo básico. ¡Hora de la validación inteligente!

**→** [Tutorial 2: Validación que Te Enseña](./tutorial_02_validacion_inteligente.md)

En el próximo tutorial aprenderás cómo MLPY detecta y explica problemas en tus datos antes de que causen errores.

---

## 🔗 Recursos

- 📖 [Documentación de Tasks](../LIBRO_MLPY/capitulo_02_conceptos_core.md)
- 📖 [Documentación de Learners](../api/learners.md)
- 🎯 [Más ejemplos con Iris](../ejemplos/clasificacion_iris.md)
- 💬 [Discord de MLPY](https://discord.gg/mlpy)

---

*"Un viaje de mil millas comienza con un solo paso."*  
**¡Has dado tu primer paso en MLPY! 🎉**