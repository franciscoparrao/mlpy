# Notebook de Susceptibilidad de Deslizamientos - Taltal

## Estado: ✅ LISTO PARA EJECUTAR

El notebook `taltal_landslide_susceptibility.ipynb` está completamente actualizado y funcional con la versión actual de MLPY.

## Cambios Importantes Realizados

### 1. Parámetros Actualizados
- **TaskClassifSpatial**: Usar `coordinate_names` en lugar de `coords`
  ```python
  task = TaskClassifSpatial(
      data=data,
      target='landslide',
      coordinate_names=['x', 'y'],  # ✅ Correcto
      crs='EPSG:32719'
  )
  ```

- **SpatialKFold**: Usar `n_folds` en lugar de `folds`
  ```python
  cv = SpatialKFold(
      n_folds=5,  # ✅ Correcto
      clustering_method='kmeans'
  )
  ```

### 2. Imports Actualizados
- Los imports ahora vienen directamente de los módulos principales:
  ```python
  from mlpy.tasks import TaskClassifSpatial
  from mlpy.resamplings import SpatialKFold
  from mlpy.filters import MRMR, Relief
  ```

### 3. Interfaz de FilterResult
- Para obtener features seleccionadas:
  ```python
  result = mrmr.calculate(task)
  # Usar .features o .select_top_k()
  selected = result.features[:10]  # o result.select_top_k(10)
  ```

## Cómo Ejecutar el Notebook

### Opción 1: Jupyter Notebook
```bash
cd C:\Users\gran_\Documents\Proyectos\MLPY
jupyter notebook notebooks/taltal_landslide_susceptibility.ipynb
```

### Opción 2: JupyterLab
```bash
cd C:\Users\gran_\Documents\Proyectos\MLPY
jupyter lab notebooks/taltal_landslide_susceptibility.ipynb
```

### Opción 3: VS Code
Abrir el archivo `.ipynb` directamente en VS Code con la extensión de Jupyter instalada.

## Verificación Previa

Ejecuta este comando para verificar que todo está listo:
```bash
python verify_notebook_ready.py
```

## Contenido del Notebook

### Fase 1: Procesamiento Geoespacial
- Carga de datos o creación de dataset sintético
- Funciones para derivadas del terreno (slope, aspect, curvature, TRI, TPI, TWI)

### Fase 2: Preparación de Datos
- Creación de dataset de 5000 muestras con características geoespaciales
- Variables del terreno, geológicas, climáticas y de vegetación

### Fase 3: Machine Learning Espacial
- Creación de tareas espaciales con MLPY
- Manejo de coordenadas y CRS

### Fase 4: Selección de Características
- Métodos individuales: MRMR, Relief, CMIM, JMI
- Ranking acumulativo (ensemble) con pesos personalizados

### Fase 5: Validación Cruzada Espacial
- Spatial K-Fold con clustering
- Spatial Block CV
- Spatial Buffer CV

### Fase 6: Benchmark Multi-Modelo
- Comparación de múltiples algoritmos:
  - Modelos base: Logistic Regression, Decision Tree
  - Ensemble: Random Forest, Gradient Boosting, AdaBoost
  - SVM, Neural Networks
  - Gradient Boosting avanzado: XGBoost, LightGBM, CatBoost
- Análisis estadístico con test de Friedman
- Visualizaciones comparativas

## Dependencias Necesarias

### Instaladas con MLPY
- numpy, pandas, scikit-learn
- matplotlib, seaborn (para visualización)

### Opcionales (para funcionalidad completa)
```bash
pip install xgboost lightgbm catboost
pip install geopandas rasterio  # Para trabajo con datos geoespaciales reales
```

## Notas Importantes

1. **Datos**: El notebook usa datos sintéticos por defecto. Para usar datos reales de Taltal, actualiza las rutas en la celda 5.

2. **Performance**: El benchmark puede tomar varios minutos. Para pruebas rápidas, reduce:
   - El número de modelos
   - El número de folds en CV
   - El tamaño del dataset

3. **Memoria**: Si encuentras problemas de memoria, establece `store_models=False` en la función benchmark.

## Troubleshooting

### Error de importación
Si aparece "ModuleNotFoundError":
```bash
pip uninstall mlpy mlpy-geo -y
cd C:\Users\gran_\Documents\Proyectos\MLPY
pip install -e .
```

### Error de parámetros
- Siempre usa `coordinate_names` (no `coords`)
- Siempre usa `n_folds` (no `folds`)

### Verificación rápida
```python
import mlpy
from mlpy.tasks import TaskClassifSpatial
from mlpy.resamplings import SpatialKFold
print("Todo OK!")
```

## Resultados Esperados

El notebook generará:
1. Rankings de modelos por múltiples métricas
2. Análisis estadístico de significancia
3. Visualizaciones comparativas (boxplots, matriz de correlación)
4. Archivos de resultados en `resultados_taltal/`

## Próximos Pasos

1. Integrar datos reales de Taltal
2. Optimización de hiperparámetros
3. Generación de mapas de susceptibilidad
4. Validación con inventarios históricos