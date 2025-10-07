"""
Test de componentes principales de MLPY
"""

import numpy as np
import pandas as pd

print("="*60)
print("TEST DE COMPONENTES PRINCIPALES")
print("="*60)

# 1. Test básico de Tasks
print("\n1. CREANDO TASKS")
print("-"*40)
try:
    from mlpy.tasks import TaskClassif, TaskRegr
    
    # Datos de prueba
    df = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100)
    })
    
    task_classif = TaskClassif(data=df, target='target', id='test_classif')
    print(f"TaskClassif creado: {task_classif.nrow} filas, {task_classif.ncol} columnas")
    
    df_regr = pd.DataFrame({
        'x1': np.random.randn(100),
        'x2': np.random.randn(100),
        'target': np.random.randn(100)
    })
    
    task_regr = TaskRegr(data=df_regr, target='target', id='test_regr')
    print(f"TaskRegr creado: {task_regr.nrow} filas, {task_regr.ncol} columnas")
    
except Exception as e:
    print(f"ERROR en Tasks: {e}")

# 2. Test de Learners
print("\n2. CREANDO LEARNERS")
print("-"*40)
try:
    from mlpy.learners import learner_sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    
    # Clasificador
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    learner_classif = learner_sklearn(rf, id='rf')
    print(f"Learner clasificación creado: {learner_classif.id}")
    
    # Regresor
    lr = LinearRegression()
    learner_regr = learner_sklearn(lr, id='lr')
    print(f"Learner regresión creado: {learner_regr.id}")
    
except Exception as e:
    print(f"ERROR en Learners: {e}")

# 3. Test de Measures
print("\n3. CREANDO MEASURES")
print("-"*40)
try:
    from mlpy.measures import MeasureClassifAcc, MeasureRegrRMSE
    
    measure_acc = MeasureClassifAcc()
    print(f"Measure clasificación: {measure_acc.id}")
    
    measure_rmse = MeasureRegrRMSE()
    print(f"Measure regresión: {measure_rmse.id}")
    
except Exception as e:
    print(f"ERROR en Measures: {e}")

# 4. Test de Resampling
print("\n4. CREANDO RESAMPLING")
print("-"*40)
try:
    from mlpy.resamplings import ResamplingCV, ResamplingHoldout
    
    cv = ResamplingCV(folds=5)
    print(f"CV creado: {cv.folds} folds")
    
    holdout = ResamplingHoldout(ratio=0.8)
    print(f"Holdout creado: {holdout.ratio} ratio")
    
except Exception as e:
    print(f"ERROR en Resampling: {e}")

# 5. Test de resample()
print("\n5. EJECUTANDO RESAMPLE")
print("-"*40)
try:
    from mlpy import resample
    
    if 'task_classif' in locals() and 'learner_classif' in locals():
        result = resample(
            task=task_classif,
            learner=learner_classif,
            resampling=ResamplingHoldout(ratio=0.8),
            measure=measure_acc
        )
        
        scores = result.aggregate()
        print(f"Resample ejecutado. Accuracy: {scores['acc'][0]:.3f}")
    else:
        print("No se pudo ejecutar resample (componentes faltantes)")
        
except Exception as e:
    print(f"ERROR en resample: {e}")

# 6. Test de Pipelines
print("\n6. CREANDO PIPELINES")
print("-"*40)
try:
    from mlpy.pipelines import PipeOpScale, PipeOpLearner, linear_pipeline
    
    if 'learner_classif' in locals():
        pipeline = linear_pipeline(
            PipeOpScale(id='scale'),
            PipeOpLearner(learner_classif, id='learner')
        )
        print(f"Pipeline creado con {len(pipeline.pipeops)} operaciones")
    else:
        print("No se pudo crear pipeline (learner faltante)")
        
except Exception as e:
    print(f"ERROR en Pipelines: {e}")

# 7. Test de benchmark()
print("\n7. EJECUTANDO BENCHMARK")
print("-"*40)
try:
    from mlpy import benchmark
    from sklearn.tree import DecisionTreeClassifier
    
    if 'task_classif' in locals():
        learners = [
            learner_sklearn(DecisionTreeClassifier(max_depth=3), id='dt3'),
            learner_sklearn(DecisionTreeClassifier(max_depth=5), id='dt5')
        ]
        
        result = benchmark(
            tasks=[task_classif],
            learners=learners,
            resampling=ResamplingCV(folds=3),
            measure=measure_acc
        )
        
        print("Benchmark ejecutado:")
        print(result.score_table())
    else:
        print("No se pudo ejecutar benchmark (componentes faltantes)")
        
except Exception as e:
    print(f"ERROR en benchmark: {e}")

# 8. Test de Persistence
print("\n8. PROBANDO PERSISTENCIA")
print("-"*40)
try:
    from mlpy.persistence import save_model, load_model
    import tempfile
    import os
    
    if 'learner_classif' in locals() and 'task_classif' in locals():
        # Entrenar learner
        learner_classif.train(task_classif)
        
        # Guardar
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        save_model(learner_classif, temp_path)
        print(f"Modelo guardado en: {temp_path}")
        
        # Cargar
        loaded_learner = load_model(temp_path)
        print(f"Modelo cargado: {loaded_learner.id}")
        
        # Limpiar
        os.unlink(temp_path)
    else:
        print("No se pudo probar persistencia (componentes faltantes)")
        
except Exception as e:
    print(f"ERROR en Persistence: {e}")

print("\n" + "="*60)
print("TEST COMPLETADO")
print("="*60)