"""
Ejemplo de AutoML Avanzado con Optuna en MLPY.

Demuestra búsqueda automática de hiperparámetros,
selección de modelos y optimización de pipelines.
"""

import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.automl.advanced_automl import AdvancedAutoML, AdvancedAutoMLConfig

print("=== AutoML Avanzado MLPY con Optuna ===\n")

# 1. Generar datos de ejemplo
print("1. PREPARANDO DATOS")
print("-" * 60)

# Clasificación
X_class, y_class = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

print(f"Dataset de clasificación:")
print(f"  - Muestras de entrenamiento: {X_train.shape[0]}")
print(f"  - Muestras de test: {X_test.shape[0]}")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Clases: {len(np.unique(y_train))}")

print("\n" + "=" * 60)
print("2. CONFIGURACIÓN DE AUTOML")
print("-" * 60)

# Configurar AutoML
config = AdvancedAutoMLConfig(
    task_type="auto",  # Detección automática
    time_budget=60,     # 60 segundos máximo
    n_trials=20,        # 20 trials de Optuna
    n_jobs=1,           # Secuencial para el ejemplo
    
    # Modelos a probar
    include_models=["random_forest", "xgboost", "linear"],
    
    # Preprocesamiento automático
    auto_preprocessing=True,
    handle_missing=True,
    scale_features=True,
    
    # Validación
    cv_folds=3,
    validation_strategy="cv",
    
    # Reporte
    verbose=1,
    show_progress=True,
    generate_report=True
)

print("Configuración:")
print(f"  - Presupuesto de tiempo: {config.time_budget}s")
print(f"  - Número de trials: {config.n_trials}")
print(f"  - Modelos a evaluar: {config.include_models}")
print(f"  - Cross-validation: {config.cv_folds} folds")

print("\n" + "=" * 60)
print("3. ENTRENAMIENTO DE AUTOML")
print("-" * 60)

# Crear y entrenar AutoML
automl = AdvancedAutoML(config)

print("\nIniciando búsqueda automática...")
start_time = time.time()

# Entrenar
automl.fit(X_train, y_train)

elapsed = time.time() - start_time
print(f"\nTiempo total de entrenamiento: {elapsed:.2f}s")

print("\n" + "=" * 60)
print("4. RESULTADOS")
print("-" * 60)

# Obtener resultados
results = automl.results

print(f"\nMejor score en validación: {results.best_score:.4f}")
print(f"Número de modelos evaluados: {len(results.leaderboard)}")

# Mostrar top 5 modelos
print("\nTop 5 modelos:")
print(results.leaderboard.head())

# Mejor configuración
print(f"\nMejor configuración encontrada:")
if results.meta_info:
    best_params = results.meta_info.get('best_params', {})
    for param, value in best_params.items():
        print(f"  - {param}: {value}")

print("\n" + "=" * 60)
print("5. EVALUACIÓN EN TEST")
print("-" * 60)

# Hacer predicciones
y_pred = automl.predict(X_test)

# Calcular accuracy
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy en test: {accuracy:.4f}")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\n" + "=" * 60)
print("6. EJEMPLO CON REGRESIÓN")
print("-" * 60)

# Generar datos de regresión
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,
    noise=0.1,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("Dataset de regresión:")
print(f"  - Muestras: {X_train_reg.shape[0]}")
print(f"  - Features: {X_train_reg.shape[1]}")

# Configuración rápida para regresión
config_reg = AdvancedAutoMLConfig(
    time_budget=30,
    n_trials=10,
    include_models=["random_forest", "linear"],
    verbose=0
)

# Entrenar
automl_reg = AdvancedAutoML(config_reg)
print("\nEntrenando AutoML para regresión...")
automl_reg.fit(X_train_reg, y_train_reg)

# Evaluar
y_pred_reg = automl_reg.predict(X_test_reg)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"MSE en test: {mse:.4f}")
print(f"R2 en test: {r2:.4f}")

print("\n" + "=" * 60)
print("7. VENTAJAS DEL AUTOML AVANZADO")
print("-" * 60)

print("""
BENEFICIOS CLAVE:

1. BÚSQUEDA INTELIGENTE:
   - Optimización Bayesiana con Optuna
   - Explora eficientemente el espacio de hiperparámetros
   - Pruning automático de configuraciones malas

2. SELECCIÓN AUTOMÁTICA:
   - Prueba múltiples algoritmos
   - Selecciona el mejor automáticamente
   - Considera trade-offs tiempo/performance

3. PIPELINE COMPLETO:
   - Preprocesamiento automático
   - Feature engineering opcional
   - Validación robusta

4. EXPLICABILIDAD:
   - Feature importance (próximamente)
   - Visualización de optimización
   - Reportes detallados

5. PRODUCCIÓN READY:
   - Modelo final entrenado con todos los datos
   - Serialización robusta (usando Fase 1)
   - Predicción simple con .predict()
""")

print("=" * 60)
print("CONCLUSIÓN: AutoML democratiza el Machine Learning")
print("-" * 60)
print("""
El AutoML Avanzado de MLPY permite a usuarios de todos los niveles:
- Novatos: Obtener buenos modelos sin experiencia
- Expertos: Ahorrar tiempo en tareas repetitivas
- Todos: Explorar más opciones sistemáticamente

Como en la meditación: 
"La maestría no está en controlar cada detalle,
sino en confiar en el proceso de optimización."

Namaste - La inteligencia artificial optimizándose a sí misma
""")