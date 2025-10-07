"""
Ejemplo de Dashboard interactivo para MLPY.

Demuestra visualización en tiempo real de métricas,
comparación de modelos y análisis de features.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.visualization.dashboard import MLPYDashboard, TrainingMetrics, create_dashboard

print("=== Dashboard Interactivo MLPY ===\n")

# 1. Crear dashboard
print("1. INICIALIZANDO DASHBOARD")
print("-" * 60)

dashboard = create_dashboard(
    title="MLPY Training Monitor",
    update_interval=1.0,
    auto_open=False  # No abrir navegador automáticamente
)

print("Dashboard creado exitosamente")

print("\n" + "=" * 60)
print("2. SIMULANDO ENTRENAMIENTO")
print("-" * 60)

# Simular métricas de entrenamiento
print("\nRegistrando métricas de entrenamiento...")
epochs = 20
for epoch in range(epochs):
    # Simular métricas
    train_loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
    val_loss = 1.1 / (epoch + 1) + np.random.random() * 0.15
    
    metrics = TrainingMetrics(
        epoch=epoch + 1,
        timestamp=time.time(),
        train_loss=train_loss,
        val_loss=val_loss,
        train_metric=1.0 - train_loss,
        val_metric=1.0 - val_loss,
        learning_rate=0.001 * (0.95 ** epoch),
        duration=np.random.uniform(0.5, 1.5)
    )
    
    dashboard.log_metrics(metrics)
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")

print("\n" + "=" * 60)
print("3. COMPARANDO MODELOS")
print("-" * 60)

# Simular comparación de modelos
models = {
    'RandomForest': {
        'score': 0.92,
        'train_time': 12.3,
        'n_params': 1000
    },
    'XGBoost': {
        'score': 0.94,
        'train_time': 18.5,
        'n_params': 1500
    },
    'LinearModel': {
        'score': 0.85,
        'train_time': 2.1,
        'n_params': 100
    },
    'NeuralNet': {
        'score': 0.93,
        'train_time': 45.2,
        'n_params': 10000
    },
    'SVM': {
        'score': 0.89,
        'train_time': 8.7,
        'n_params': 500
    }
}

print("\nRegistrando modelos...")
for model_name, metrics in models.items():
    dashboard.log_model(model_name, metrics)
    print(f"  {model_name}: Score={metrics['score']:.3f}, Time={metrics['train_time']:.1f}s")

print("\n" + "=" * 60)
print("4. FEATURE IMPORTANCE")
print("-" * 60)

# Simular importancia de features
feature_names = [f'feature_{i}' for i in range(20)]
importance_values = np.random.exponential(0.2, 20)
importance_values = importance_values / importance_values.sum()

feature_importance = dict(zip(feature_names, importance_values))
dashboard.log_feature_importance(feature_importance)

# Mostrar top 5 features
print("\nTop 5 features más importantes:")
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for i, (feat, imp) in enumerate(sorted_features[:5], 1):
    print(f"  {i}. {feat}: {imp:.4f}")

print("\n" + "=" * 60)
print("5. GENERANDO VISUALIZACIONES")
print("-" * 60)

# Generar dashboard
print("\nGenerando dashboard visual...")
dashboard_path = dashboard.start()

print("\n" + "=" * 60)
print("6. EXPORTANDO REPORTE")
print("-" * 60)

# Exportar reporte completo
report_path = dashboard.export_report()
print(f"Reporte JSON exportado")

print("\n" + "=" * 60)
print("7. RESUMEN DE MÉTRICAS")
print("-" * 60)

# Mostrar resumen
if dashboard.metrics_history:
    df = pd.DataFrame(dashboard.metrics_history)
    
    print("\nEstadísticas de entrenamiento:")
    print(f"  - Epochs totales: {len(df)}")
    print(f"  - Loss inicial: {df['train_loss'].iloc[0]:.4f}")
    print(f"  - Loss final: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  - Mejor val_loss: {df['val_loss'].min():.4f}")
    print(f"  - Tiempo promedio/epoch: {df['duration'].mean():.2f}s")

if dashboard.models_comparison:
    print("\nComparación de modelos:")
    best_model = max(
        dashboard.models_comparison.items(),
        key=lambda x: x[1].get('score', 0)
    )
    print(f"  - Mejor modelo: {best_model[0]}")
    print(f"  - Mejor score: {best_model[1]['score']:.4f}")
    print(f"  - Modelos evaluados: {len(dashboard.models_comparison)}")

print("\n" + "=" * 60)
print("8. CASOS DE USO DEL DASHBOARD")
print("-" * 60)

print("""
APLICACIONES PRÁCTICAS:

1. MONITOREO DE ENTRENAMIENTO:
   - Visualizar loss en tiempo real
   - Detectar overfitting temprano
   - Ajustar learning rate dinámicamente

2. COMPARACIÓN DE MODELOS:
   - Benchmark múltiples algoritmos
   - Trade-off precisión vs tiempo
   - Selección informada del mejor modelo

3. ANÁLISIS DE FEATURES:
   - Identificar features importantes
   - Detectar features redundantes
   - Guiar feature engineering

4. DEBUGGING:
   - Identificar problemas de convergencia
   - Analizar gradientes y pesos
   - Diagnosticar data issues

5. REPORTES:
   - Documentación automática
   - Compartir resultados con equipo
   - Trazabilidad de experimentos
""")

print("=" * 60)
print("CONCLUSIÓN: Visualización como herramienta de comprensión")
print("-" * 60)
print("""
El Dashboard de MLPY transforma números en insights:
- Métricas abstractas se vuelven patrones visuales
- Comparaciones complejas se simplifican
- El progreso se vuelve tangible

Como en la meditación:
"Ver claramente es el primer paso hacia la comprensión."

Namaste - La visualización ilumina el camino del aprendizaje
""")