"""
Script de prueba para verificar que MLPY funciona desde cualquier ubicación.
"""

import os
import sys

print("="*60)
print("TEST: MLPY desde cualquier ubicación")
print("="*60)
print(f"\nDirectorio actual: {os.getcwd()}")
print(f"Python ejecutable: {sys.executable}")

# Opción 1: Si MLPY está instalado con pip install -e .
try:
    import mlpy
    print("\n✓ MLPY importado exitosamente (instalación global)")
    print(f"  Ubicación: {mlpy.__file__}")
except ImportError:
    print("\n✗ MLPY no está instalado globalmente")
    
    # Opción 2: Añadir manualmente al path
    print("\nIntentando añadir MLPY al path...")
    mlpy_path = r'C:\Users\gran_\Documents\Proyectos\MLPY'
    
    if os.path.exists(mlpy_path):
        sys.path.insert(0, mlpy_path)
        try:
            import mlpy
            print("✓ MLPY importado exitosamente (añadido al path)")
            print(f"  Ubicación: {mlpy.__file__}")
        except ImportError as e:
            print(f"✗ Error al importar MLPY: {e}")
    else:
        print(f"✗ No se encuentra el directorio: {mlpy_path}")

# Probar funcionalidad básica
try:
    print("\n" + "-"*40)
    print("Probando funcionalidad básica...")
    
    from mlpy.tasks import TaskClassif
    from mlpy.learners import learner_sklearn
    from mlpy.filters import list_filters
    
    print("✓ Imports básicos funcionando")
    
    # Listar algunos componentes
    print(f"\nFiltros disponibles: {len(list_filters())}")
    print(f"Primeros 5 filtros: {list_filters()[:5]}")
    
    # Crear una tarea simple
    import pandas as pd
    df = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 4, 6, 8, 10],
        'y': [0, 0, 1, 1, 1]
    })
    task = TaskClassif(df, 'y')
    print(f"\n✓ Task creado: {task.nrow} filas, {len(task.feature_names)} features")
    
    print("\n¡MLPY está funcionando correctamente!")
    
except Exception as e:
    print(f"\n✗ Error al usar MLPY: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)