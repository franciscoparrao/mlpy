#!/usr/bin/env python
"""
Script para verificar que todos los imports del notebook funcionan correctamente.
"""

print("Verificando imports del notebook...")
print("=" * 60)

# Test 1: Imports básicos
print("\n1. Imports básicos:")
try:
    import sys
    import os
    import numpy as np
    import pandas as pd
    import warnings
    print("  [OK] Librerías básicas")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 2: MLPY principal
print("\n2. MLPY principal:")
try:
    import mlpy
    print(f"  [OK] MLPY desde: {mlpy.__file__}")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 3: Componentes espaciales
print("\n3. Componentes espaciales:")
try:
    from mlpy.tasks import TaskClassifSpatial
    from mlpy.resamplings import SpatialKFold, SpatialBlockCV, SpatialBufferCV
    print("  [OK] Tareas y resamplings espaciales")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 4: Learners
print("\n4. Learners:")
try:
    from mlpy.learners import learner_sklearn
    from mlpy.learners import learner_xgboost, learner_lightgbm, learner_catboost
    print("  [OK] Todos los learners importados")
except ImportError as e:
    print(f"  [ERROR] Algunos learners no disponibles: {e}")
    # Intentar importar individualmente
    try:
        from mlpy.learners import learner_sklearn
        print("  [OK] learner_sklearn")
    except:
        print("  [ERROR] learner_sklearn")

# Test 5: Filtros
print("\n5. Filtros de selección:")
try:
    from mlpy.filters import MRMR, CMIM, JMI, Relief, ReliefF, CumulativeRanking
    print("  [OK] Filtros de selección")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 6: Benchmark
print("\n6. Sistema de benchmark:")
try:
    from mlpy.benchmark_advanced import benchmark_grid, benchmark, compare_learners
    print("  [OK] Benchmark avanzado")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 7: Medidas
print("\n7. Medidas:")
try:
    from mlpy.measures import create_measure
    print("  [OK] Sistema de medidas")
except ImportError as e:
    print(f"  [ERROR] {e}")

# Test 8: Crear una tarea espacial simple
print("\n8. Prueba funcional:")
try:
    # Datos de ejemplo
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, 100),
        'y': np.random.uniform(0, 100, 100),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Crear tarea
    task = TaskClassifSpatial(
        data=data,
        target='target',
        coordinate_names=['x', 'y'],
        crs='EPSG:4326',
        id='test_task'
    )
    
    print(f"  [OK] Tarea espacial creada")
    print(f"       - Dimensiones: {task.nrow} x {task.ncol}")
    
    # Probar filtro
    mrmr = MRMR(n_features=2)
    result = mrmr.calculate(task)
    print(f"  [OK] Filtro MRMR ejecutado")
    print(f"       - Features seleccionadas: {result.get_selected_features()}")
    
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("Verificación completada!")
print("\nSi todos los tests pasan, el notebook debería funcionar correctamente.")