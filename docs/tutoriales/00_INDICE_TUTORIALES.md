# 🎯 Tutoriales MLPY: Aprendizaje Práctico

## De Zero a Hero en 12 Tutoriales

---

## 📚 TUTORIALES FUNDAMENTALES

### 🟢 Nivel Principiante

#### [Tutorial 1: Tu Primer Modelo en 5 Minutos](./tutorial_01_primer_modelo.md)
**Tiempo:** 5 minutos  
**Objetivo:** Entrenar un clasificador de iris desde cero  
**Conceptos:** Tasks, Learners, validación básica

#### [Tutorial 2: Validación que Te Enseña](./tutorial_02_validacion_inteligente.md)
**Tiempo:** 10 minutos  
**Objetivo:** Entender el sistema de validación de MLPY  
**Conceptos:** Mensajes educativos, detección de problemas

#### [Tutorial 3: Guardando Modelos de Forma Segura](./tutorial_03_serializacion_robusta.md)
**Tiempo:** 8 minutos  
**Objetivo:** Serialización con checksums y metadata  
**Conceptos:** RobustSerializer, integridad, versionado

### 🟡 Nivel Intermedio

#### [Tutorial 4: Optimización Automática con Lazy Eval](./tutorial_04_lazy_evaluation.md)
**Tiempo:** 15 minutos  
**Objetivo:** Acelerar pipelines con evaluación diferida  
**Conceptos:** ComputationGraph, caching, optimización

#### [Tutorial 5: AutoML: La Máquina que Entrena Máquinas](./tutorial_05_automl_basico.md)
**Tiempo:** 20 minutos  
**Objetivo:** Automatizar selección de modelos  
**Conceptos:** Búsqueda automática, Optuna, early stopping

#### [Tutorial 6: Dashboard de Monitoreo](./tutorial_06_dashboard_visualizacion.md)
**Tiempo:** 12 minutos  
**Objetivo:** Visualizar métricas en tiempo real  
**Conceptos:** TrainingMetrics, comparación de modelos

### 🔴 Nivel Avanzado

#### [Tutorial 7: Explicabilidad de Modelos](./tutorial_07_explicabilidad.md)
**Tiempo:** 25 minutos  
**Objetivo:** Entender qué hace el modelo internamente  
**Conceptos:** SHAP, LIME, feature importance

#### [Tutorial 8: Pipeline Completo End-to-End](./tutorial_08_pipeline_completo.md)
**Tiempo:** 30 minutos  
**Objetivo:** Proyecto completo desde datos hasta producción  
**Conceptos:** Integración de todos los componentes

## 📊 TUTORIALES POR DOMINIO

#### [Tutorial 9: Predicción de Ventas (Regresión)](./tutorial_09_prediccion_ventas.md)
**Tiempo:** 25 minutos  
**Objetivo:** Forecasting con series temporales  
**Casos de uso:** Retail, finanzas, planificación

#### [Tutorial 10: Detección de Fraude (Clasificación)](./tutorial_10_deteccion_fraude.md)
**Tiempo:** 30 minutos  
**Objetivo:** Clasificación binaria con datos desbalanceados  
**Casos de uso:** Banca, seguros, e-commerce

#### [Tutorial 11: Segmentación de Clientes (Clustering)](./tutorial_11_segmentacion_clientes.md)
**Tiempo:** 20 minutos  
**Objetivo:** Clustering no supervisado  
**Casos de uso:** Marketing, CRM, product management

#### [Tutorial 12: Análisis de Sentimientos (NLP)](./tutorial_12_analisis_sentimientos.md)
**Tiempo:** 35 minutos  
**Objetivo:** Procesamiento de texto y clasificación  
**Casos de uso:** Social media, reviews, customer service

---

## 🛠 CARACTERÍSTICAS DE LOS TUTORIALES

### ✅ Lo que INCLUYEN:

- **Código completo** que puedes copiar y ejecutar
- **Explicaciones paso a paso** de cada concepto
- **Outputs esperados** para verificar tu progreso
- **Ejercicios prácticos** para reforzar el aprendizaje
- **Troubleshooting** de errores comunes
- **Recursos adicionales** para profundizar

### 📋 Estructura Estándar:

1. **Objetivo y Contexto** (2 min)
2. **Setup Inicial** (1 min)  
3. **Implementación Paso a Paso** (70% del tiempo)
4. **Análisis de Resultados** (15% del tiempo)
5. **Ejercicios y Siguiente Paso** (15% del tiempo)

### 🎯 Niveles de Dificultad:

- 🟢 **Principiante**: Python básico + conceptos ML básicos
- 🟡 **Intermedio**: Experiencia con pandas/sklearn
- 🔴 **Avanzado**: Conocimiento profundo de ML

---

## 📖 CÓMO USAR LOS TUTORIALES

### Para Seguir la Secuencia Completa:
```
Tutorial 1 → Tutorial 2 → Tutorial 3 → ... → Tutorial 12
(Tiempo total: ~4 horas)
```

### Para Necesidades Específicas:
```
¿Nuevo en MLPY? → Tutoriales 1-3
¿Quieres optimizar? → Tutoriales 4-5  
¿Necesitas explicar resultados? → Tutorial 7
¿Proyecto real? → Tutorial 8 + dominio específico
```

### Para Casos de Uso:
```
Retail/E-commerce → Tutoriales 9, 11, 12
Finanzas → Tutoriales 9, 10
Healthcare → Tutoriales 10, 12
Marketing → Tutoriales 11, 12
```

---

## 💻 SETUP PARA TODOS LOS TUTORIALES

### Instalación:
```bash
# Instalación completa con todas las dependencias
pip install mlpy-framework[full]

# O instalación paso a paso
pip install mlpy-framework
pip install optuna plotly shap lime
```

### Verificación:
```python
import mlpy
print(f"MLPY Version: {mlpy.__version__}")
mlpy.check_health()
```

### Estructura de Proyecto Recomendada:
```
mi_proyecto_mlpy/
├── datos/              # Datasets
├── notebooks/          # Jupyter notebooks
├── modelos/           # Modelos entrenados
├── resultados/        # Outputs y reportes
└── utils/             # Código reutilizable
```

---

## 🚀 TUTORIALES RÁPIDOS (5 MINUTOS)

### Quick Start - Clasificación:
```python
from mlpy.tasks import TaskClassif
from mlpy.learners import LearnerClassifSklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Datos
df = pd.read_csv('mi_dataset.csv')

# 2. Tarea  
task = TaskClassif(data=df, target='label')

# 3. Modelo
learner = LearnerClassifSklearn(
    estimator=RandomForestClassifier()
)

# 4. Entrenar
learner.train(task)

# 5. Predecir
predictions = learner.predict(new_data)
```

### Quick Start - AutoML:
```python
from mlpy.automl import SimpleAutoML

# 1. AutoML
automl = SimpleAutoML(time_budget=300)  # 5 minutos

# 2. Entrenar
automl.fit(X_train, y_train)

# 3. Mejor modelo
best_model = automl.best_estimator_
```

### Quick Start - Dashboard:
```python
from mlpy.visualization import create_dashboard

# 1. Dashboard
dashboard = create_dashboard(title="Mi Experimento")

# 2. Log métricas
dashboard.log_metrics({
    'epoch': 1,
    'loss': 0.5,
    'accuracy': 0.85
})

# 3. Visualizar
dashboard.start()
```

---

## 📞 SOPORTE Y COMUNIDAD

### ¿Necesitas Ayuda?

- 📖 **Documentación completa**: [docs.mlpy.org](https://docs.mlpy.org)
- 💬 **Discord**: [discord.gg/mlpy](https://discord.gg/mlpy)
- 🐛 **Issues**: [github.com/mlpy/issues](https://github.com/mlpy/issues)
- 📧 **Email**: support@mlpy.org

### Contribuir:

¿Tienes ideas para nuevos tutoriales? 
¿Encontraste un error? 
¿Quieres mejorar la documentación?

**¡Tu contribución es bienvenida!**

---

## 🎯 OBJETIVOS DE APRENDIZAJE

Al completar estos tutoriales, podrás:

✅ **Usar MLPY** para cualquier proyecto de ML  
✅ **Validar datos** proactivamente  
✅ **Optimizar pipelines** automáticamente  
✅ **Visualizar resultados** efectivamente  
✅ **Explicar modelos** con confianza  
✅ **Desplegar en producción** de forma segura  

---

*"El aprendizaje es un tesoro que seguirá  
a su propietario a todas partes."*

**¡Comienza tu viaje de aprendizaje!** 🚀

**→** [Tutorial 1: Tu Primer Modelo en 5 Minutos](./tutorial_01_primer_modelo.md)