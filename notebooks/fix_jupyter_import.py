"""
Script para corregir los imports en Jupyter Notebook.
Ejecuta este script DENTRO del notebook como primera celda si tienes problemas de importación.
"""

import sys
import os

# Remover la ruta de Anaconda si está presente
anaconda_paths = [p for p in sys.path if 'anaconda3' in p.lower()]
for path in anaconda_paths:
    if 'site-packages' in path:
        print(f"Removiendo del path: {path}")
        sys.path.remove(path)

# Asegurar que MLPY local está primero en el path
MLPY_PATH = r'C:\Users\gran_\Documents\Proyectos\MLPY'
if MLPY_PATH not in sys.path:
    sys.path.insert(0, MLPY_PATH)
    print(f"Agregado al inicio del path: {MLPY_PATH}")

# Verificar que ahora importa correctamente
import mlpy
print(f"\nMLPY importado desde: {mlpy.__file__}")

# Verificar componentes espaciales
try:
    from mlpy.tasks import TaskClassifSpatial
    from mlpy.resamplings import SpatialKFold
    print("✓ Componentes espaciales disponibles")
except ImportError as e:
    print(f"✗ Error importando componentes espaciales: {e}")

print("\nPath de Python actualizado. Ahora puedes ejecutar el resto del notebook.")