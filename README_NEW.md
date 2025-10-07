# MLPY - Machine Learning Framework for Python

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/mlpy/badge/?version=latest)](https://mlpy.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://img.shields.io/badge/coverage-90%25-green.svg)](https://github.com/mlpy-project/mlpy)

MLPY es un framework moderno y modular de machine learning para Python, inspirado en [mlr3](https://mlr3.mlr-org.com/). Proporciona una interfaz unificada, componible y extensible para flujos de trabajo de machine learning.

## 🚀 Características Principales

- **🎯 API Unificada**: Interfaz consistente para diferentes tareas de ML
- **🧩 Diseño Modular**: Bloques componibles para flujos de trabajo complejos
- **🔧 Extensible**: Fácil añadir learners, medidas y operadores personalizados
- **📊 Evaluación Robusta**: Múltiples estrategias de resampling y métricas
- **🤖 AutoML**: Optimización de hiperparámetros y feature engineering automático
- **⚡ Paralelización**: Backends para computación paralela eficiente
- **📈 Visualización**: Gráficos integrados para análisis de resultados
- **🔗 Integración**: Compatible con scikit-learn y otros frameworks

## 📦 Instalación

### Instalación Básica

```bash
pip install mlpy
```

### Con Dependencias Opcionales

```bash
# Todas las dependencias
pip install mlpy[all]

# Solo visualización
pip install mlpy[viz]

# Solo interpretabilidad
pip install mlpy[interpret]
```

### Desde el Código Fuente

```bash
git clone https://github.com/mlpy-project/mlpy.git
cd mlpy
pip install -e .[dev]
```

## 🎯 Inicio Rápido

```python
import mlpy
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import learner_sklearn
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy
from sklearn.ensemble import RandomForestClassifier

# Crear tarea
task = TaskClassif(data=df, target="species")

# Crear learner
rf = RandomForestClassifier(n_estimators=100)
learner = learner_sklearn(rf)

# Evaluar con cross-validation
result = mlpy.resample(
    task=task,
    learner=learner,
    resampling=ResamplingCV(folds=5),
    measures=[MeasureClassifAccuracy()]
)

# Ver resultados
print(result.aggregate())
```

## 📚 Documentación

La documentación completa está disponible en [https://mlpy.readthedocs.io](https://mlpy.readthedocs.io)

### Guías y Tutoriales

- [Getting Started](docs/source/quickstart.rst) - Tutorial de inicio rápido
- [Conceptos Principales](docs/source/concepts.rst) - Arquitectura y diseño
- [Ejemplos de Clasificación](examples/notebooks/01_getting_started.ipynb) - Notebook interactivo
- [AutoML Tutorial](examples/notebooks/02_automl_example.ipynb) - Optimización automática

## 🔬 Ejemplos

### Comparación de Modelos

```python
from mlpy import benchmark

# Definir learners
learners = [
    learner_sklearn(LogisticRegression(), id='logreg'),
    learner_sklearn(RandomForestClassifier(), id='rf'),
    learner_sklearn(GradientBoostingClassifier(), id='gb')
]

# Benchmark
result = benchmark(
    tasks=[task],
    learners=learners,
    resampling=ResamplingCV(folds=5),
    measures=[MeasureClassifAccuracy()]
)

# Ver rankings
print(result.rank_learners('classif.acc'))
```

### Pipeline con Preprocesamiento

```python
from mlpy.pipelines import linear_pipeline
from mlpy.pipelines.operators import PipeOpScale, PipeOpImpute

# Crear pipeline
pipeline = linear_pipeline([
    PipeOpImpute(strategy='mean'),
    PipeOpScale(method='standard'),
    learner
])

# Usar como learner normal
result = mlpy.resample(task, pipeline, resampling, measures)
```

### AutoML - Optimización de Hiperparámetros

```python
from mlpy.automl import TunerGridSearch, ParamSet

# Definir espacio de búsqueda
params = ParamSet({
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
})

# Optimizar
tuner = TunerGridSearch(params, measure=MeasureClassifAccuracy())
best_learner = tuner.tune(task, learner)
```

### Paralelización

```python
from mlpy.parallel import BackendMultiprocessing

# Ejecutar benchmark en paralelo
with BackendMultiprocessing(n_jobs=4):
    result = benchmark(tasks, learners, resampling, measures)
```

## 🏗️ Arquitectura

```
Task (Datos) → Learner (Modelo) → Prediction (Resultados)
                    ↓
              Resampling → Measure (Evaluación)
```

### Componentes Principales

- **Task**: Encapsula datos y metadatos del problema
- **Learner**: Interfaz unificada para algoritmos ML
- **Measure**: Métricas de evaluación
- **Resampling**: Estrategias de validación
- **Pipeline**: Composición de operaciones
- **Benchmark**: Comparación sistemática

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=mlpy

# Solo tests rápidos
pytest -m "not slow"
```

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor, consulta [CONTRIBUTING.md](CONTRIBUTING.md) para detalles.

1. Fork el proyecto
2. Crea tu rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📈 Roadmap

- [x] Core framework
- [x] Integración scikit-learn
- [x] Sistema de pipelines
- [x] AutoML básico
- [x] Paralelización
- [x] Visualización
- [ ] Más learners nativos
- [ ] Soporte para deep learning
- [ ] Integración con Dask/Ray
- [ ] CLI completo

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- Inspirado por [mlr3](https://github.com/mlr-org/mlr3)
- Construido sobre [scikit-learn](https://scikit-learn.org/)
- Comunidad Python ML

## 📬 Contacto

- Documentación: [https://mlpy.readthedocs.io](https://mlpy.readthedocs.io)
- Issues: [GitHub Issues](https://github.com/mlpy-project/mlpy/issues)
- Discusiones: [GitHub Discussions](https://github.com/mlpy-project/mlpy/discussions)

---

<p align="center">
  Hecho con ❤️ por la comunidad MLPY
</p>