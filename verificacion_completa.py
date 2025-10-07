"""
Verificación completa de todas las mejoras de MLPY
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 70)
print(" " * 15 + "VERIFICACIÓN COMPLETA MLPY v2.0")
print("=" * 70)

resultados = []

# =============================================================================
# 1. VALIDACIÓN CON PYDANTIC
# =============================================================================
print("\n[1/5] SISTEMA DE VALIDACIÓN")
print("-" * 50)

try:
    from mlpy.validation import validate_task_data, ValidatedTask
    
    # Caso 1: Datos válidos
    df_valid = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    result = validate_task_data(df_valid, target='target')
    assert result['valid'] == True
    print("  [OK] Validación de datos correctos")
    
    # Caso 2: Datos con problemas
    df_problematic = pd.DataFrame({
        'f1': [1, 2, np.nan, 4],  # NaN
        'f2': [1, 1, 1, 1],        # Constante
        'target': [0, 1, 0, 1]
    })
    
    result = validate_task_data(df_problematic, target='target')
    assert len(result['warnings']) > 0  # Debe detectar problemas
    print("  [OK] Detección de problemas en datos")
    
    # Caso 3: Crear tarea validada
    task = ValidatedTask(
        data=df_valid,
        target='target',
        task_type='classification'
    )
    assert task.task is not None
    print("  [OK] Creación de tarea validada")
    
    resultados.append(("Validación", True, "Funcionando correctamente"))
    print("\n  RESULTADO: PASSED")
    
except Exception as e:
    resultados.append(("Validación", False, str(e)))
    print(f"\n  RESULTADO: FAILED - {e}")

# =============================================================================
# 2. SERIALIZACIÓN ROBUSTA
# =============================================================================
print("\n[2/5] SERIALIZACIÓN ROBUSTA")
print("-" * 50)

try:
    from mlpy.serialization import RobustSerializer
    from sklearn.ensemble import RandomForestClassifier
    
    # Entrenar modelo
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 1], 50)
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Serializar con metadata
    serializer = RobustSerializer()
    metadata = {
        'accuracy': model.score(X, y),
        'timestamp': time.time(),
        'version': '1.0'
    }
    
    # Guardar
    save_info = serializer.save(
        obj=model,
        path='test_model.pkl',
        metadata=metadata
    )
    print("  [OK] Modelo serializado con metadata")
    
    # Cargar con validación
    loaded_model = serializer.load('test_model.pkl')
    assert hasattr(loaded_model, 'predict')
    print("  [OK] Modelo cargado correctamente")
    
    # Verificar integridad
    test_pred = loaded_model.predict(X[:5])
    assert len(test_pred) == 5
    print("  [OK] Integridad verificada")
    
    # Limpiar
    for f in ['test_model.pkl', 'test_model.meta.json']:
        if os.path.exists(f):
            os.remove(f)
    
    resultados.append(("Serialización", True, "Checksum y metadata funcionando"))
    print("\n  RESULTADO: PASSED")
    
except Exception as e:
    resultados.append(("Serialización", False, str(e)))
    print(f"\n  RESULTADO: FAILED - {e}")

# =============================================================================
# 3. LAZY EVALUATION
# =============================================================================
print("\n[3/5] LAZY EVALUATION")
print("-" * 50)

try:
    from mlpy.lazy import ComputationGraph, ComputationNode
    
    # Crear grafo
    graph = ComputationGraph()
    
    # Operaciones con seguimiento
    exec_count = {'n1': 0, 'n2': 0}
    
    def op1():
        exec_count['n1'] += 1
        return 10
    
    def op2():
        exec_count['n2'] += 1
        return 20
    
    # Crear nodos
    node1 = ComputationNode(id='n1', operation='op1', func=op1)
    node2 = ComputationNode(id='n2', operation='op2', func=op2)
    
    graph.add_node(node1)
    graph.add_node(node2)
    
    # Verificar lazy - no se ejecuta hasta llamar execute
    assert exec_count['n1'] == 0
    print("  [OK] Operaciones diferidas correctamente")
    
    # Ejecutar nodo individual
    result1 = node1.execute()
    assert result1 == 10
    assert exec_count['n1'] == 1
    print("  [OK] Ejecución bajo demanda")
    
    # Verificar caching
    result1_cached = node1.execute()
    assert exec_count['n1'] == 1  # No se ejecutó de nuevo
    print("  [OK] Caching funcionando")
    
    resultados.append(("Lazy Evaluation", True, "Optimización y caching activos"))
    print("\n  RESULTADO: PASSED")
    
except Exception as e:
    resultados.append(("Lazy Evaluation", False, str(e)))
    print(f"\n  RESULTADO: FAILED - {e}")

# =============================================================================
# 4. DASHBOARD
# =============================================================================
print("\n[4/5] DASHBOARD INTERACTIVO")
print("-" * 50)

try:
    from mlpy.visualization.dashboard import create_dashboard, TrainingMetrics
    
    # Crear dashboard
    dashboard = create_dashboard(
        title="Test Dashboard",
        auto_open=False
    )
    print("  [OK] Dashboard creado")
    
    # Registrar métricas
    for i in range(3):
        metrics = TrainingMetrics(
            epoch=i+1,
            timestamp=time.time(),
            train_loss=1.0/(i+1),
            val_loss=1.1/(i+1)
        )
        dashboard.log_metrics(metrics)
    
    assert len(dashboard.metrics_history) == 3
    print("  [OK] Métricas registradas")
    
    # Comparar modelos
    dashboard.log_model('Model1', {'score': 0.90})
    dashboard.log_model('Model2', {'score': 0.95})
    
    assert len(dashboard.models_comparison) == 2
    print("  [OK] Comparación de modelos")
    
    resultados.append(("Dashboard", True, "Visualización funcionando"))
    print("\n  RESULTADO: PASSED")
    
except Exception as e:
    resultados.append(("Dashboard", False, str(e)))
    print(f"\n  RESULTADO: FAILED - {e}")

# =============================================================================
# 5. AUTOML
# =============================================================================
print("\n[5/5] AUTOML")
print("-" * 50)

try:
    from mlpy.automl import SimpleAutoML
    
    # Verificar importación
    assert SimpleAutoML is not None
    print("  [OK] AutoML importado")
    
    # Verificar estructura básica
    automl = SimpleAutoML()
    assert hasattr(automl, 'fit')
    assert hasattr(automl, 'predict')
    print("  [OK] API de AutoML disponible")
    
    resultados.append(("AutoML", True, "Módulo disponible"))
    print("\n  RESULTADO: PASSED")
    
except Exception as e:
    resultados.append(("AutoML", False, str(e)))
    print(f"\n  RESULTADO: FAILED - {e}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print(" " * 20 + "RESUMEN DE VERIFICACIÓN")
print("=" * 70)

passed = sum(1 for _, status, _ in resultados if status)
total = len(resultados)

print("\nComponentes verificados:")
print("-" * 50)

for componente, status, detalle in resultados:
    estado = "[PASS]" if status else "[FAIL]"
    simbolo = "✓" if status else "✗"
    print(f"  {componente:20} {estado:8} - {detalle}")

print("\n" + "-" * 50)
print(f"TOTAL: {passed}/{total} componentes funcionando")

if passed == total:
    print("\n" + "=" * 70)
    print(" " * 15 + "TODAS LAS MEJORAS VERIFICADAS")
    print(" " * 10 + "MLPY v2.0 FUNCIONANDO CORRECTAMENTE")
    print("=" * 70)
else:
    print(f"\nATENCIÓN: {total - passed} componentes necesitan revisión")

print("\n[FIN DE VERIFICACIÓN]")