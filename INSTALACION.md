# Guía de Instalación de MLPY

## Instalación del paquete

MLPY ahora está disponible como `mlpy-geo` para evitar conflictos con otros paquetes. 

### Instalación desde el código fuente

```bash
# Clonar o navegar al directorio del proyecto
cd C:\Users\gran_\Documents\Proyectos\MLPY

# Instalar en modo desarrollo
pip install -e .
```

### Verificar la instalación

Ejecuta el script de verificación:

```bash
python test_mlpy_installation.py
```

O verifica manualmente:

```python
# Importar MLPY
import mlpy
print(f"MLPY instalado en: {mlpy.__file__}")

# Importar componentes espaciales
from mlpy.tasks import TaskClassifSpatial, TaskRegrSpatial
from mlpy.resamplings import SpatialKFold, SpatialBlockCV
from mlpy.filters import MRMR, Relief, CumulativeRanking
from mlpy.benchmark_advanced import benchmark, benchmark_grid
```

## Uso desde cualquier directorio

Una vez instalado, puedes usar MLPY desde cualquier ubicación en tu sistema:

```python
import mlpy
from mlpy.tasks import TaskClassifSpatial
from mlpy.resamplings import SpatialKFold
# etc...
```

## Características Espaciales Disponibles

### Tareas Espaciales
- `TaskClassifSpatial`: Clasificación con soporte de coordenadas
- `TaskRegrSpatial`: Regresión con soporte de coordenadas

### Validación Cruzada Espacial
- `SpatialKFold`: K-fold espacial usando clustering
- `SpatialBlockCV`: Validación por bloques espaciales
- `SpatialBufferCV`: Validación con buffer entre train/test
- `SpatialEnvironmentalCV`: Validación por características ambientales

### Selección de Características
- **Métodos de Información Mutua**: MRMR, CMIM, JMI, JMIM, MIM
- **Métodos Estadísticos**: Relief, ReliefF, DISR
- **Ensemble**: CumulativeRanking para combinar múltiples métodos

### Benchmarking
- `benchmark`: Sistema avanzado de comparación de modelos
- `benchmark_grid`: Diseño de experimentos multi-modelo
- Análisis estadístico con tests de Friedman y Kruskal-Wallis

## Ejemplo de Uso

```python
import pandas as pd
import numpy as np
from mlpy.tasks import TaskClassifSpatial
from mlpy.resamplings import SpatialKFold
from mlpy.filters import MRMR, CumulativeRanking

# Crear datos de ejemplo
data = pd.DataFrame({
    'x': np.random.uniform(0, 100, 1000),
    'y': np.random.uniform(0, 100, 1000),
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'target': np.random.choice([0, 1], 1000)
})

# Crear tarea espacial
task = TaskClassifSpatial(
    data=data,
    target='target',
    coordinate_names=['x', 'y'],  # Nota: usar 'coordinate_names', no 'coords'
    crs='EPSG:4326',
    id='mi_tarea_espacial'
)

# Validación cruzada espacial
cv = SpatialKFold(folds=5, clustering_method='kmeans')

# Selección de características
mrmr = MRMR(n_features=10)
selected_features = mrmr.calculate(task)
```

## Notebook de Ejemplo

Ver `notebooks/taltal_landslide_susceptibility.ipynb` para un ejemplo completo de análisis de susceptibilidad de deslizamientos usando MLPY.

## Resolución de Problemas

### Error de importación
Si encuentras errores de importación, verifica que:
1. MLPY está instalado correctamente: `pip show mlpy-geo`
2. No hay conflictos con versiones anteriores

### Limpiar instalaciones anteriores
Si tienes una versión anterior de mlpy:
```bash
pip uninstall mlpy mlpy-geo -y
pip install -e .
```

## Dependencias Principales

- numpy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- click >= 8.0.0
- pyyaml >= 5.4.0

### Dependencias Opcionales

Para gradient boosting:
- xgboost
- lightgbm
- catboost

Para análisis geoespacial:
- geopandas
- rasterio
- shapely