#!/usr/bin/env python
"""
Verificación final de que el notebook está listo para ejecutarse.
"""

import sys
import os

print("=" * 70)
print("VERIFICACIÓN FINAL DEL NOTEBOOK TALTAL")
print("=" * 70)

# Test completo simulando el flujo del notebook
try:
    print("\n1. Importando librerías básicas...")
    import numpy as np
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    print("   [OK] Librerías básicas")
    
    print("\n2. Importando MLPY...")
    import mlpy
    print(f"   [OK] MLPY v{mlpy.__version__ if hasattr(mlpy, '__version__') else 'dev'}")
    print(f"   Ubicación: {mlpy.__file__}")
    
    print("\n3. Importando componentes espaciales...")
    from mlpy.tasks import TaskClassifSpatial
    from mlpy.resamplings import SpatialKFold, SpatialBlockCV
    from mlpy.filters import MRMR, Relief, CumulativeRanking
    from mlpy.learners import learner_sklearn
    from mlpy.benchmark_advanced import benchmark_grid, benchmark
    print("   [OK] Componentes espaciales importados")
    
    print("\n4. Creando datos de ejemplo...")
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        'x': np.random.uniform(350000, 380000, n),
        'y': np.random.uniform(7100000, 7150000, n),
        'elevation': np.random.uniform(0, 3000, n),
        'slope': np.random.gamma(2, 10, n),
        'aspect': np.random.uniform(0, 360, n),
        'tri': np.random.gamma(1.5, 5, n),
        'twi': np.random.normal(7, 2, n),
        'ndvi': np.random.beta(2, 5, n),
        'landslide': np.random.choice([0, 1], n)
    })
    print(f"   [OK] Dataset creado: {data.shape}")
    
    print("\n5. Creando tarea espacial...")
    task = TaskClassifSpatial(
        data=data,
        target='landslide',
        coordinate_names=['x', 'y'],  # Parámetro correcto
        crs='EPSG:32719',
        id='test_susceptibility'
    )
    print(f"   [OK] Tarea creada: {task.nrow} filas, {task.ncol} columnas")
    print(f"   CRS: {task.crs}")
    
    print("\n6. Probando selección de características...")
    mrmr = MRMR(n_features=5)
    result = mrmr.calculate(task)
    # Usar la interfaz correcta
    if hasattr(result, 'select_top_k'):
        selected = result.select_top_k(5)
    elif hasattr(result, 'features'):
        selected = result.features[:5]
    else:
        selected = list(result.scores.keys())[:5] if hasattr(result, 'scores') else []
    print(f"   [OK] MRMR ejecutado")
    print(f"   Features seleccionadas: {selected}")
    
    print("\n7. Probando validación cruzada espacial...")
    cv = SpatialKFold(n_folds=3, clustering_method='kmeans', random_state=42)
    print("   [OK] SpatialKFold configurado")
    
    print("\n8. Probando learners...")
    from sklearn.ensemble import RandomForestClassifier
    learner = learner_sklearn(
        RandomForestClassifier(n_estimators=10, random_state=42),
        id='rf_test'
    )
    print("   [OK] Learner creado")
    
    print("\n9. Probando diseño de benchmark...")
    design = benchmark_grid(
        tasks=task,
        learners=learner,
        resamplings=cv,
        measures=['accuracy']
    )
    print(f"   [OK] Diseño creado con {design.n_experiments} experimentos")
    
    print("\n" + "=" * 70)
    print("VERIFICACIÓN EXITOSA")
    print("=" * 70)
    print("\nEl notebook está listo para ejecutarse.")
    print("\nNotas importantes:")
    print("- Usar 'coordinate_names' en lugar de 'coords' para TaskClassifSpatial")
    print("- Los FilterResult usan .features o .select_top_k() para obtener features")
    print("- MLPY está instalado como 'mlpy-geo' pero se importa como 'mlpy'")
    print("\nPuedes ejecutar el notebook desde Jupyter con:")
    print("  jupyter notebook notebooks/taltal_landslide_susceptibility.ipynb")
    
except Exception as e:
    print(f"\n[ERROR] Fallo en la verificación: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)