# Glosario Técnico: Machine Learning EN ↔ ES

**Propósito**: Referencia rápida para traducción consistente del paper JMLR.

**Principio**: Mantener términos establecidos en la comunidad ML internacional.

---

## 📚 Categorías

1. [Conceptos Fundamentales ML](#conceptos-fundamentales-ml)
2. [Cross-Validation y Resampling](#cross-validation-y-resampling)
3. [Spatial ML](#spatial-ml)
4. [AutoML y Tuning](#automl-y-tuning)
5. [Programación y Software](#programación-y-software)
6. [NO Traducir (Mantener en Inglés)](#no-traducir)
7. [Verbos y Expresiones Comunes](#verbos-y-expresiones)

---

## Conceptos Fundamentales ML

| Español | English | Notas |
|---------|---------|-------|
| Aprendizaje automático | Machine learning | O "Machine Learning" (mayúsculas cuando es nombre propio) |
| Conjunto de datos | Dataset | O "dataset" (aceptable en español técnico) |
| Conjunto de entrenamiento | Training set | |
| Conjunto de prueba | Test set | |
| Conjunto de validación | Validation set | |
| Característica | Feature | O "feature" (común en español técnico) |
| Variable predictora | Predictor | |
| Variable objetivo | Target variable | O "target" |
| Etiqueta | Label | |
| Muestra | Sample | O "observation" |
| Observación | Observation | |
| Instancia | Instance | |
| Modelo | Model | |
| Algoritmo | Algorithm | |
| Clasificación | Classification | |
| Regresión | Regression | |
| Agrupamiento | Clustering | |
| Predicción | Prediction | |
| Entrenamiento | Training | |
| Ajuste | Fitting | "Model fitting" = "ajuste del modelo" |
| Sobreajuste | Overfitting | ⚠️ NO "sobreadaptación" |
| Subajuste | Underfitting | |
| Sesgo | Bias | |
| Varianza | Variance | |
| Error de generalización | Generalization error | |
| Rendimiento | Performance | O "desempeño" |
| Precisión | Accuracy | ⚠️ Contexto: clasificación |
| Exactitud | Precision | ⚠️ Contexto: métrica P/(P+FP) |
| Exhaustividad | Recall | O "sensibilidad", "recall" |
| Medida F1 | F1 score | O "F1-score" |

---

## Cross-Validation y Resampling

| Español | English | Notas |
|---------|---------|-------|
| Validación cruzada | Cross-validation | ⚠️ Término establecido |
| Remuestreo | Resampling | ⚠️ Término clave del paper |
| Pliegue | Fold | "5-fold CV" = "CV de 5 pliegues" |
| Validación cruzada de k pliegues | k-fold cross-validation | |
| Validación cruzada dejando uno fuera | Leave-one-out cross-validation | LOOCV |
| Bootstrap | Bootstrap | NO traducir |
| Holdout | Holdout | NO traducir (o "retención") |
| Estratificado | Stratified | |
| Partición | Split | O "división" |
| Iteración | Iteration | O "fold" (contexto CV) |

---

## Spatial ML

| Español | English | Notas |
|---------|---------|-------|
| Espacial | Spatial | |
| Geoespacial | Geospatial | |
| Autocorrelación espacial | Spatial autocorrelation | ⚠️ Término clave |
| I de Moran | Moran's I | NO traducir "Moran's" |
| Variograma | Variogram | |
| Semivarianza | Semivariance | |
| Distancia geográfica | Geographic distance | |
| Coordenadas | Coordinates | |
| Fuga de datos espaciales | Spatial data leakage | O "filtración espacial" |
| Remuestreo espacial | Spatial resampling | ⚠️ Contribución principal |
| Validación cruzada espacial | Spatial cross-validation | |
| Bloque espacial | Spatial block | |
| Buffer espacial | Spatial buffer | O "zona de amortiguamiento" |
| K-fold espacial | Spatial k-fold | |
| Vecindad | Neighborhood | O "vecindario" |
| Primera ley de la geografía de Tobler | Tobler's First Law of Geography | NO traducir "Tobler's" |

---

## AutoML y Tuning

| Español | English | Notas |
|---------|---------|-------|
| Hiperparámetro | Hyperparameter | |
| Ajuste de hiperparámetros | Hyperparameter tuning | |
| Optimización de hiperparámetros | Hyperparameter optimization | |
| Búsqueda en cuadrícula | Grid search | O "Grid search" |
| Búsqueda aleatoria | Random search | |
| Optimización bayesiana | Bayesian optimization | |
| Espacio de búsqueda | Search space | |
| Espacio de parámetros | Parameter space | |
| Configuración | Configuration | |
| Rango | Range | |
| Límite inferior | Lower bound | |
| Límite superior | Upper bound | |
| Resolución | Resolution | Contexto: densidad de grid |
| Declarativo | Declarative | |
| Automático | Automatic | |
| Manual | Manual | |
| Evaluación | Evaluation | |
| Prueba | Trial | Contexto: tuning iteration |

---

## Programación y Software

| Español | English | Notas |
|---------|---------|-------|
| Marco de trabajo | Framework | O "framework" (común) |
| Biblioteca | Library | |
| Paquete | Package | |
| Módulo | Module | |
| Clase | Class | |
| Función | Function | |
| Método | Method | |
| Parámetro | Parameter | |
| Argumento | Argument | |
| Interfaz de programación de aplicaciones | API | Usar "API" directamente |
| Código | Code | |
| Implementación | Implementation | |
| Documentación | Documentation | |
| Dependencia | Dependency | |
| Instalación | Installation | |
| Usuario | User | |
| Desarrollador | Developer | |
| Ejemplo de uso | Usage example | |
| Fragmento de código | Code snippet | |
| Listado | Listing | Contexto: código en paper |
| Figura | Figure | |
| Tabla | Table | |
| Sección | Section | |
| Subsección | Subsection | |

---

## NO Traducir (Mantener en Inglés)

### Nombres de Algoritmos
- Random Forest (nombre propio)
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- k-Nearest Neighbors (k-NN)
- Decision Tree
- Logistic Regression (puede traducirse, pero común dejarlo)
- Linear Regression (puede traducirse, pero común dejarlo)

### Nombres de Bibliotecas/Frameworks
- scikit-learn (o sklearn)
- NumPy
- pandas
- TensorFlow
- PyTorch
- Keras
- mlr3
- caret
- tidymodels

### Términos Técnicos Establecidos
- Pipeline (muy usado en español técnico)
- Backend
- Frontend
- Callback
- Wrapper
- Grid search (o "búsqueda en cuadrícula")
- Cross-validation (o "validación cruzada")
- Feature engineering (o "ingeniería de características")
- One-hot encoding
- Label encoding
- Imbalanced data (o "datos desbalanceados")

### Acrónimos
- ML (Machine Learning)
- AI (Artificial Intelligence)
- CV (Cross-Validation)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- AUC (Area Under Curve)
- ROC (Receiver Operating Characteristic)
- API
- CLI (Command Line Interface)
- GPU (Graphics Processing Unit)
- CPU (Central Processing Unit)

### Nombres de Datasets Clásicos
- Iris
- Wine
- Boston Housing (deprecated)
- MNIST
- CIFAR-10
- Meuse (dataset espacial)
- Titanic

---

## Verbos y Expresiones Comunes

### Acciones ML

| Español | English |
|---------|---------|
| Entrenar (un modelo) | Train (a model) |
| Ajustar | Fit |
| Predecir | Predict |
| Clasificar | Classify |
| Evaluar | Evaluate |
| Validar | Validate |
| Probar | Test |
| Optimizar | Optimize |
| Mejorar | Improve |
| Reducir | Reduce |
| Aumentar | Increase |
| Prevenir | Prevent |
| Detectar | Detect |
| Identificar | Identify |
| Comparar | Compare |
| Implementar | Implement |
| Desarrollar | Develop |
| Proporcionar | Provide |
| Ofrecer | Offer |
| Soportar | Support |
| Permitir | Allow / Enable |

### Expresiones Académicas

| Español | English |
|---------|---------|
| Como se muestra en | As shown in |
| De acuerdo con | According to |
| En este trabajo | In this work |
| Nuestro enfoque | Our approach |
| Proponemos | We propose |
| Demostramos | We demonstrate |
| Los resultados muestran | The results show |
| Es importante notar | It is important to note |
| En contraste con | In contrast to |
| A diferencia de | Unlike |
| Similar a | Similar to |
| Por lo tanto | Therefore / Thus |
| Sin embargo | However |
| Además | Moreover / Furthermore |
| En particular | In particular |
| Por ejemplo | For example / For instance |
| En resumen | In summary |
| En conclusión | In conclusion |

### Conectores y Transiciones

| Español | English |
|---------|---------|
| Primero | First / Firstly |
| Segundo | Second / Secondly |
| Finalmente | Finally |
| Por un lado | On one hand |
| Por otro lado | On the other hand |
| Aunque | Although / While |
| Mientras que | While / Whereas |
| Dado que | Given that / Since |
| Debido a | Due to / Because of |
| Como resultado | As a result |
| En consecuencia | Consequently |
| De esta manera | In this way |
| En otras palabras | In other words |

---

## ⚠️ False Friends Comunes (Falsos Amigos)

| ❌ Español | ❌ Traducción Incorrecta | ✅ Traducción Correcta |
|-----------|-------------------------|----------------------|
| Realizar | Realize | Conduct / Perform / Carry out |
| Actualmente | Actually | Currently |
| Eventualmente | Eventually | Possibly / If needed |
| Librería | Library (books) | Library (código) está bien |
| Aplicación | Application (solicitud) | Application (software) está bien |
| Soportar | Support (aguantar) | Support (código) está bien |
| Sensible | Sensible (razonable) | Sensitive |
| Éxito | Exit | Success |
| Argumento | Argument (disputa) | Argument (parámetro) está bien |

### Ejemplos de Uso Correcto

**Realizar experimentos**:
- ❌ We realize experiments
- ✅ We conduct experiments / We perform experiments

**Actualmente**:
- ❌ Actually, ML is widely used
- ✅ Currently, ML is widely used

**Eventualmente**:
- ❌ The model will eventually overfit
- ✅ The model may overfit / The model might eventually overfit (si implica tiempo largo)

**Aplicación**:
- ✅ Application (software context) - correcto
- ❌ Application (job application) - usar "solicitud"

**Soportar**:
- ✅ The framework supports spatial CV - correcto
- ❌ Support the weight - usar "aguantar"

---

## 🎯 Decisiones de Traducción Específicas

### 1. "Performance"

**Opciones**:
- **Rendimiento** (preferida para velocidad/eficiencia)
- **Desempeño** (más formal, aceptable)
- **Performance** (común en español técnico)

**Recomendación**: Usar "performance" directamente en contextos técnicos, "rendimiento" en contextos generales.

**Ejemplos**:
- "The model's performance" → "El rendimiento del modelo"
- "Performance metrics" → "Métricas de desempeño" o "Performance metrics"

---

### 2. "Feature"

**Opciones**:
- **Característica** (más formal)
- **Variable** (estadística)
- **Feature** (técnico, común)
- **Atributo** (menos común)

**Recomendación**: Usar "característica" en general, "feature" en contextos muy técnicos.

**Ejemplos**:
- "Feature engineering" → "Ingeniería de características" o "Feature engineering"
- "The dataset has 10 features" → "El conjunto de datos tiene 10 características"

---

### 3. "Framework"

**Opciones**:
- **Marco de trabajo** (traducción literal)
- **Framework** (muy común en español técnico)

**Recomendación**: Usar "framework" directamente (es ampliamente aceptado).

**Ejemplo**:
- "MLPY is a framework" → "MLPY es un framework"

---

### 4. "Cross-validation"

**Opciones**:
- **Validación cruzada** (estándar, preferida)
- **Cross-validation** (aceptable en papers técnicos)

**Recomendación**: Usar "validación cruzada" consistentemente.

**Ejemplo**:
- "5-fold cross-validation" → "Validación cruzada de 5 pliegues" o "5-fold CV"

---

### 5. "Overfitting" / "Underfitting"

**Opciones**:
- **Sobreajuste** / **Subajuste** (estándar)
- **Sobreentrenamiento** (menos común)
- **Overfitting** / **Underfitting** (técnico)

**Recomendación**: Usar "sobreajuste" y "subajuste".

**Ejemplo**:
- "To prevent overfitting" → "Para prevenir el sobreajuste"

---

## 📖 Convenciones de Traducción

### Números y Unidades

**Mantener formato internacional**:
- ✅ 0.9667 (punto decimal - no coma)
- ✅ 96.67% (no 96,67%)
- ✅ 1,000 (coma para miles - opcional)

### Citas Bibliográficas

**NO traducir**:
- Nombres de autores: "Roberts et al." (no "Roberts y colaboradores")
- Títulos de papers citados
- Nombres de journals
- Comandos LaTeX: `\citep{roberts2017}` mantener igual

### Ecuaciones

**Mantener exactamente igual**:
```latex
% NO cambiar
I = \frac{N}{\sum_{i,j} w_{ij}} \frac{\sum_{i,j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}
```

**Solo traducir texto explicativo**:
- "where $N$ is the number" → "donde $N$ es el número"

### Código

**NO traducir**:
- Nombres de funciones: `cross_val_score()`
- Nombres de variables en código
- Comentarios de código (opcional traducirlos)
- Nombres de parámetros: `n_estimators`, `max_depth`

**Ejemplo**:
```python
# Español: Crear tarea de clasificación
# English: Create classification task
task = TaskClassif(data=df, target='species')
```

---

## 🔍 Cómo Usar Este Glosario

### Durante la Traducción

1. **Duda sobre un término**: Busca en tabla correspondiente
2. **No está en glosario**:
   - Busca en Google Scholar papers en español
   - Marca con comentario: `% DUDA: término`
   - Reviso en checkpoint

### Mantener Consistencia

**Primera vez que traduces un término**:
1. Decide traducción (usando glosario)
2. Márcala mentalmente
3. Usa SIEMPRE la misma traducción

**Ejemplo**:
- Si traduces "feature" como "característica" la primera vez
- SIEMPRE usa "característica" (no mezclar con "variable", "atributo")

---

## ✅ Checklist Rápida

Antes de traducir cada párrafo:

- [ ] Identificar términos técnicos
- [ ] Consultar glosario para términos clave
- [ ] Decidir qué NO traducir (nombres propios, algoritmos)
- [ ] Verificar false friends comunes
- [ ] Mantener comandos LaTeX intactos

---

## 📝 Notas Adicionales

### Estilo Académico

**Preferir**:
- Voz pasiva refleja: "se demuestra", "se observa"
- Plural mayestático: "proponemos", "demostramos" (no "propongo")
- Presente atemporal: "MLPY proporciona" (no "proporcionará")

**Evitar**:
- Calcos sintácticos del español
- Redundancias: "Los resultados muestran que demuestran..." → "Los resultados demuestran..."
- Anglicismos innecesarios: "performar" → "desempeñar/funcionar"

### Términos Específicos del Paper

| Concepto | Español | English |
|----------|---------|---------|
| Contribución principal 1 | Remuestreo espacial | Spatial resampling |
| Contribución principal 2 | Ajuste automático de hiperparámetros | Automated hyperparameter tuning |
| Dataset usado (Spatial) | Meuse | Meuse |
| Dataset usado (AutoML) | Iris | Iris |
| Métrica clave (Spatial) | I de Moran | Moran's I |
| Métrica clave (error) | RMSE | RMSE |
| Técnica propuesta | SpatialKFold | SpatialKFold (nombre de clase) |
| Parámetro declarativo | ParamInt | ParamInt (nombre de clase) |

---

**Última actualización**: 2025-10-09
**Versión**: 1.0
**Para**: Traducción paper JMLR MLPY
