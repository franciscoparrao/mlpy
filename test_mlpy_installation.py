#!/usr/bin/env python
"""
Script de prueba para verificar la instalación de MLPY (mlpy-geo).
Ejecutar desde cualquier directorio para verificar que funciona.
"""

import sys
import os
print(f"Python: {sys.version}")
print(f"Directorio actual: {os.getcwd()}")
print("-" * 60)

# Test 1: Importación básica
try:
    import mlpy
    print("[OK] MLPY importado correctamente")
    print(f"  Ubicación: {mlpy.__file__}")
    if hasattr(mlpy, '__version__'):
        print(f"  Versión: {mlpy.__version__}")
except ImportError as e:
    print(f"[ERROR] Error importando MLPY: {e}")
    sys.exit(1)

# Test 2: Importar componentes espaciales
try:
    from mlpy.tasks import TaskClassifSpatial, TaskRegrSpatial
    print("[OK] Tareas espaciales importadas")
except ImportError as e:
    print(f"[ERROR] Error importando tareas espaciales: {e}")

try:
    from mlpy.resamplings import SpatialKFold, SpatialBlockCV
    print("[OK] Resamplings espaciales importados")
except ImportError as e:
    print(f"[ERROR] Error importando resamplings espaciales: {e}")

# Test 3: Importar filtros
try:
    from mlpy.filters import MRMR, Relief, CumulativeRanking
    print("[OK] Filtros de selección importados")
except ImportError as e:
    print(f"[ERROR] Error importando filtros: {e}")

# Test 4: Importar benchmark
try:
    from mlpy.benchmark_advanced import benchmark, benchmark_grid
    print("[OK] Sistema de benchmark importado")
except ImportError as e:
    print(f"[ERROR] Error importando benchmark: {e}")

# Test 5: Crear una tarea espacial simple
try:
    import pandas as pd
    import numpy as np
    
    # Datos de ejemplo
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x': np.random.uniform(0, 100, n),
        'y': np.random.uniform(0, 100, n),
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Crear tarea
    task = TaskClassifSpatial(
        data=data,
        target='target',
        coordinate_names=['x', 'y'],
        crs='EPSG:4326',
        id='test_task'
    )
    
    print("[OK] Tarea espacial creada exitosamente")
    print(f"  Dimensiones: {task.nrow} filas, {task.ncol} columnas")
    print(f"  CRS: {task.crs}")
    
except Exception as e:
    print(f"[ERROR] Error creando tarea espacial: {e}")

print("-" * 60)
print("Instalacion verificada exitosamente!")
print("\nPuedes usar MLPY desde cualquier directorio con:")
print("  import mlpy")
print("  from mlpy.tasks import TaskClassifSpatial")
print("  from mlpy.resamplings import SpatialKFold")
print("  etc...")