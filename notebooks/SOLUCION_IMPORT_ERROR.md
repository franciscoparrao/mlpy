# Solución al Error de Importación en Jupyter

## El Problema
```python
ImportError: cannot import name 'TaskClassifSpatial' from 'mlpy.tasks'
```

Este error ocurre porque Jupyter está usando una versión antigua de MLPY instalada en Anaconda en lugar de la versión nueva con componentes espaciales.

## Solución Rápida

### Opción 1: Ejecutar el notebook de setup primero
1. Abre `setup_mlpy_jupyter.ipynb`
2. Ejecuta todas las celdas
3. Si todo sale bien, abre `taltal_landslide_susceptibility.ipynb`

### Opción 2: Agregar esta celda al inicio de tu notebook
```python
# Corregir imports - EJECUTAR ESTA CELDA PRIMERO
import sys
import os

# Remover versión antigua
anaconda_paths = [p for p in sys.path if 'anaconda3' in p.lower() and 'site-packages' in p]
for path in anaconda_paths:
    sys.path.remove(path)

# Usar versión nueva
MLPY_PATH = r'C:\Users\gran_\Documents\Proyectos\MLPY'
if MLPY_PATH not in sys.path:
    sys.path.insert(0, MLPY_PATH)

# Verificar
import mlpy
print(f"MLPY desde: {mlpy.__file__}")
from mlpy.tasks import TaskClassifSpatial
print("✓ Imports corregidos")
```

## Solución Permanente

### Desde PowerShell como Administrador:
```powershell
# 1. Renombrar carpeta antigua (backup)
Rename-Item "C:\Users\gran_\anaconda3\lib\site-packages\mlpy" "C:\Users\gran_\anaconda3\lib\site-packages\mlpy_old_backup"

# 2. Reinstalar MLPY
cd C:\Users\gran_\Documents\Proyectos\MLPY
pip install -e .
```

### O eliminar completamente la versión antigua:
```powershell
# Eliminar carpeta antigua (¡cuidado, esto es permanente!)
Remove-Item -Recurse -Force "C:\Users\gran_\anaconda3\lib\site-packages\mlpy"

# Reinstalar
cd C:\Users\gran_\Documents\Proyectos\MLPY
pip install -e .
```

## Verificación

Ejecuta este código para verificar que todo funciona:

```python
import mlpy
from mlpy.tasks import TaskClassifSpatial
from mlpy.resamplings import SpatialKFold
from mlpy.filters import MRMR
from mlpy.benchmark_advanced import benchmark

print("✅ Todos los imports funcionan!")
print(f"MLPY ubicado en: {mlpy.__file__}")
```

## ¿Por qué ocurre esto?

1. **Instalaciones múltiples**: Tienes MLPY instalado en dos lugares:
   - Versión antigua: `C:\Users\gran_\anaconda3\lib\site-packages\mlpy` (sin componentes espaciales)
   - Versión nueva: `C:\Users\gran_\Documents\Proyectos\MLPY` (con componentes espaciales)

2. **Prioridad de Python**: Por defecto, Python/Jupyter busca primero en Anaconda.

3. **Conflicto de nombres**: Ambas se llaman `mlpy`, entonces Python usa la primera que encuentra.

## Notas Importantes

- La versión nueva está instalada como `mlpy-geo` pero se importa como `mlpy`
- El notebook ya tiene la corrección en la celda 2
- Si usas diferentes kernels de Jupyter, puede que necesites aplicar la corrección en cada uno

## Si Nada Funciona

1. Reinicia el kernel de Jupyter (Kernel → Restart)
2. Cierra Jupyter completamente
3. Ejecuta desde terminal:
   ```bash
   cd C:\Users\gran_\Documents\Proyectos\MLPY
   python verify_notebook_ready.py
   ```
   Si este script funciona, el problema es solo con Jupyter.

4. Considera crear un nuevo ambiente virtual:
   ```bash
   python -m venv mlpy_env
   mlpy_env\Scripts\activate
   pip install -e .
   pip install jupyter
   jupyter notebook
   ```