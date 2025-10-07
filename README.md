# 🚀 MLPY - Modern Machine Learning Framework for Python

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-85%25%20passing-green.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-16%25-orange.svg)](htmlcov/index.html)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

MLPY es un framework moderno y extensible de aprendizaje automático para Python, inspirado en [mlr3](https://mlr3.mlr-org.com/). Proporciona una interfaz unificada, componible y orientada a objetos para tareas de machine learning.

## ✨ Características Principales

- 🎯 **API Unificada**: Interfaz consistente para clasificación, regresión, clustering y más
- 🧩 **Diseño Modular**: Bloques componibles para flujos de trabajo complejos
- 🔧 **Altamente Extensible**: Fácil integración de nuevos learners, medidas y transformaciones
- 📊 **Evaluación Robusta**: Múltiples estrategias de resampling y medidas de rendimiento
- 🤖 **AutoML Integrado**: Optimización automática de hiperparámetros y feature engineering
- ⚡ **Alto Rendimiento**: Paralelización nativa y backends optimizados
- 🔍 **Explicabilidad**: Interpretación de modelos con SHAP, LIME y más
- 📈 **Visualización Rica**: Gráficos interactivos para análisis de resultados
- 🔗 **Integración Total**: Compatible con scikit-learn, XGBoost, LightGBM, PyTorch

## 📦 Instalación

### Instalación Básica
```bash
pip install mlpy
```

### Instalación Completa (todas las dependencias)
```bash
pip install mlpy[all]
```

### Instalación para Desarrollo
```bash
git clone https://github.com/your-org/mlpy.git
cd mlpy
pip install -e .[dev]
```

## 🚀 Inicio Rápido

### Ejemplo Básico de Clasificación

```python
import pandas as pd
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.resamplings import ResamplingCV
from mlpy.measures import MeasureClassifAccuracy

# Cargar datos
data = pd.read_csv("iris.csv")

# Crear tarea de clasificación
task = TaskClassif(data=data, target="species")

# Crear learner
learner = LearnerRandomForestClassifier(n_estimators=100, random_state=42)

# Configurar cross-validation
cv = ResamplingCV(folds=5)

# Entrenar y evaluar
learner.train(task)
predictions = learner.predict(task)

# Medir rendimiento
measure = MeasureClassifAccuracy()
accuracy = measure.score(predictions.truth, predictions.response)
print(f"Accuracy: {accuracy:.2%}")
```

### Pipeline Completo con AutoML

```python
from mlpy.automl import AutoML
from mlpy.tasks import TaskRegr

# Crear tarea
task = TaskRegr(data=data, target="price")

# Configurar AutoML
automl = AutoML(
    task=task,
    time_budget=300,  # 5 minutos
    metric="rmse"
)

# Ejecutar optimización
best_model = automl.fit()

# Hacer predicciones
predictions = best_model.predict(task)
```

## 📚 Documentación Completa

### Guías Principales

- 📖 [Guía de Usuario Completa](docs/DOCUMENTATION_SUMMARY.md)
- 🎓 [Tutoriales Paso a Paso](docs/tutoriales/00_INDICE_TUTORIALES.md)
- 🔬 [Guía de Evaluación Lazy](docs/LAZY_EVALUATION_GUIDE.md)
- 📊 [Guía de Big Data](docs/BIG_DATA_GUIDE.md)
- 🧪 [Guía de Testing](docs/TESTING_GUIDE.md)
- 💾 [Guía de Persistencia](docs/PERSISTENCE_GUIDE.md)
- 🖥️ [Guía de CLI](docs/CLI_GUIDE.md)

### Casos de Uso

- 🛒 [Predicción de Churn en Retail](docs/casos_uso/retail_prediccion_churn.md)
- 📊 [Ejemplos con Big Data](docs/BIG_DATA_EXAMPLES.md)

## 🏗️ Arquitectura

```
mlpy/
├── core/           # Componentes fundamentales
├── tasks/          # Definición de tareas ML
├── learners/       # Algoritmos de aprendizaje
├── measures/       # Métricas de evaluación
├── resamplings/    # Estrategias de validación
├── pipelines/      # Pipelines de procesamiento
├── automl/         # AutoML y optimización
├── validation/     # Validación de datos
├── backends/       # Backends de computación
└── visualization/  # Herramientas de visualización
```

## 🎯 Modelos Disponibles

### Clasificación
- Random Forest, Gradient Boosting, XGBoost, LightGBM
- SVM, Logistic Regression, Naive Bayes
- Redes Neuronales (MLP, CNN, RNN)
- Deep Learning con PyTorch

### Regresión
- Linear/Ridge/Lasso/ElasticNet
- Random Forest, Gradient Boosting
- Support Vector Regression
- Redes Neuronales

### Clustering
- K-Means, DBSCAN, Hierarchical
- Gaussian Mixture Models
- Spectral Clustering
- HDBSCAN con auto-tuning

### Deep Learning
- LSTM, GRU, BiLSTM para series temporales
- Transformers para NLP
- CNNs para visión por computadora

## 📊 Benchmarking

```python
from mlpy.benchmark import Benchmark

# Configurar benchmark
benchmark = Benchmark(
    tasks=[task1, task2],
    learners=[learner1, learner2, learner3],
    resamplings=[cv, holdout],
    measures=[accuracy, auc, f1]
)

# Ejecutar
results = benchmark.run(parallel=True)

# Visualizar
results.plot_comparison()
results.to_latex("results.tex")
```

## 🧪 Testing

El framework incluye una suite completa de tests:

```bash
# Ejecutar todos los tests
pytest tests/

# Con coverage
pytest --cov=mlpy tests/

# Tests rápidos de validación
python test_quick_validation.py
```

Estado actual:
- ✅ 7/7 tests de validación pasando
- ✅ 29/34 tests unitarios pasando (85%)
- 📊 16.28% coverage de código

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📈 Roadmap

### v2.1 (Actual)
- ✅ Model Registry completo
- ✅ Deep Learning models (LSTM, GRU, Transformers)
- ✅ Advanced clustering con auto-tuning
- ✅ Sistema de validación mejorado

### v3.0 (Próximo)
- [ ] MLOps completo (tracking, deployment)
- [ ] AutoML mejorado con NAS
- [ ] Distributed training
- [ ] GUI interactiva

## 📝 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Inspirado en [mlr3](https://mlr3.mlr-org.com/) de R
- Construido sobre [scikit-learn](https://scikit-learn.org/)
- Integración con el ecosistema Python ML

## 📞 Contacto

- 📧 Email: mlpy@example.com
- 💬 Discord: [MLPY Community](https://discord.gg/mlpy)
- 🐦 Twitter: [@mlpy_framework](https://twitter.com/mlpy_framework)

---

**⭐ Si te gusta MLPY, dale una estrella en GitHub!**

<p align="center">
  Hecho con ❤️ por la comunidad MLPY
</p>