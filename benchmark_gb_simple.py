"""
Benchmark simplificado para comparar XGBoost, LightGBM y CatBoost.
Usa directamente scikit-learn para los datos y métricas.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Verificar disponibilidad
print("="*60)
print("Verificando librerías instaladas...")
print("="*60)

libraries = {}
try:
    import xgboost as xgb
    libraries['xgboost'] = True
    print("[OK] XGBoost version:", xgb.__version__)
except ImportError:
    libraries['xgboost'] = False
    print("[X] XGBoost no disponible")

try:
    import lightgbm as lgb
    libraries['lightgbm'] = True
    print("[OK] LightGBM version:", lgb.__version__)
except ImportError:
    libraries['lightgbm'] = False
    print("[X] LightGBM no disponible")

try:
    import catboost as cb
    libraries['catboost'] = True
    print("[OK] CatBoost version:", cb.__version__)
except ImportError:
    libraries['catboost'] = False
    print("[X] CatBoost no disponible")

def create_datasets():
    """Crea datasets de prueba variados."""
    datasets = []
    
    # 1. Clasificación binaria - pequeño
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                              n_redundant=5, n_classes=2, random_state=42)
    datasets.append({
        'name': 'Binary_Small',
        'X': X,
        'y': y,
        'type': 'classification'
    })
    
    # 2. Clasificación multiclase - mediano
    X, y = make_classification(n_samples=5000, n_features=30, n_informative=20,
                              n_classes=5, n_clusters_per_class=1, random_state=42)
    datasets.append({
        'name': 'Multiclass_Medium',
        'X': X,
        'y': y,
        'type': 'classification'
    })
    
    # 3. Regresión - mediano
    X, y = make_regression(n_samples=5000, n_features=25, n_informative=20,
                          noise=0.1, random_state=42)
    datasets.append({
        'name': 'Regression_Medium',
        'X': X,
        'y': y,
        'type': 'regression'
    })
    
    # 4. Dataset con features categóricas
    n_samples = 3000
    np.random.seed(42)
    X_cat = pd.DataFrame({
        'num_1': np.random.randn(n_samples),
        'num_2': np.random.randn(n_samples) * 10,
        'cat_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'cat_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'cat_3': np.random.choice([0, 1, 2, 3, 4], n_samples),
    })
    y_cat = (X_cat['num_1'] > 0).astype(int) + (X_cat['cat_1'] == 'A').astype(int)
    y_cat = (y_cat > 0).astype(int)
    
    datasets.append({
        'name': 'Mixed_Categorical',
        'X': X_cat,
        'y': y_cat,
        'type': 'classification',
        'has_categorical': True
    })
    
    # 5. Dataset grande
    X, y = make_classification(n_samples=20000, n_features=50, n_informative=35,
                              n_redundant=10, n_classes=2, random_state=42)
    datasets.append({
        'name': 'Binary_Large',
        'X': X,
        'y': y,
        'type': 'classification'
    })
    
    return datasets

def benchmark_xgboost(X_train, X_test, y_train, y_test, task_type='classification', **kwargs):
    """Benchmark XGBoost."""
    if not libraries['xgboost']:
        return None
    
    import xgboost as xgb
    
    # Convertir categorical features a numéricas para XGBoost
    if kwargs.get('has_categorical', False) and isinstance(X_train, pd.DataFrame):
        from sklearn.preprocessing import LabelEncoder
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        for col in X_train.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    if task_type == 'classification':
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            params['objective'] = 'binary:logistic'
            model = xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBRegressor(**params)
    
    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - start_time
    
    # Predecir
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    return {
        'model': model,
        'y_pred': y_pred,
        'train_time': train_time,
        'predict_time': predict_time
    }

def benchmark_lightgbm(X_train, X_test, y_train, y_test, task_type='classification', **kwargs):
    """Benchmark LightGBM."""
    if not libraries['lightgbm']:
        return None
    
    import lightgbm as lgb
    from sklearn.preprocessing import LabelEncoder
    
    # Procesar categorical features para LightGBM
    cat_features = None
    if kwargs.get('has_categorical', False) and isinstance(X_train, pd.DataFrame):
        X_train = X_train.copy()
        X_test = X_test.copy()
        
        # Convertir object columns a category
        for col in X_train.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'verbosity': -1
    }
    
    if task_type == 'classification':
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            params['objective'] = 'binary'
            model = lgb.LGBMClassifier(**params)
        else:
            params['objective'] = 'multiclass'
            params['num_class'] = n_classes
            model = lgb.LGBMClassifier(**params)
    else:
        params['objective'] = 'regression'
        model = lgb.LGBMRegressor(**params)
    
    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    train_time = time.time() - start_time
    
    # Predecir
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    return {
        'model': model,
        'y_pred': y_pred,
        'train_time': train_time,
        'predict_time': predict_time
    }

def benchmark_catboost(X_train, X_test, y_train, y_test, task_type='classification', **kwargs):
    """Benchmark CatBoost."""
    if not libraries['catboost']:
        return None
    
    import catboost as cb
    
    params = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': False
    }
    
    # Identificar categorical features
    cat_features = None
    if kwargs.get('has_categorical', False) and isinstance(X_train, pd.DataFrame):
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if task_type == 'classification':
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            params['loss_function'] = 'Logloss'
            model = cb.CatBoostClassifier(**params)
        else:
            params['loss_function'] = 'MultiClass'
            params['classes_count'] = n_classes
            model = cb.CatBoostClassifier(**params)
    else:
        params['loss_function'] = 'RMSE'
        model = cb.CatBoostRegressor(**params)
    
    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))
    train_time = time.time() - start_time
    
    # Predecir
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    return {
        'model': model,
        'y_pred': y_pred,
        'train_time': train_time,
        'predict_time': predict_time
    }

def evaluate_predictions(y_true, y_pred, task_type='classification'):
    """Evalúa las predicciones."""
    if task_type == 'classification':
        # Para multiclase, y_pred podría necesitar argmax
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_true, y_pred)
        
        # AUC solo para binario
        try:
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_pred)
            else:
                auc = None
        except:
            auc = None
            
        return {'accuracy': acc, 'auc': auc}
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {'rmse': rmse, 'mae': mae}

def run_benchmark():
    """Ejecuta el benchmark completo."""
    print("\n" + "="*80)
    print(" GRADIENT BOOSTING BENCHMARK ".center(80))
    print("="*80)
    
    # Crear datasets
    print("\nCreando datasets de prueba...")
    datasets = create_datasets()
    
    # Resultados
    all_results = []
    
    # Para cada dataset
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset['name']}")
        print(f"Shape: {dataset['X'].shape if hasattr(dataset['X'], 'shape') else f'{len(dataset['X'])} x {len(dataset['X'].columns)}'}")
        print(f"Type: {dataset['type']}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['X'], dataset['y'], test_size=0.2, random_state=42
        )
        
        results = {
            'Dataset': dataset['name'],
            'Type': dataset['type'],
            'Train_Size': len(X_train),
            'Test_Size': len(X_test)
        }
        
        # XGBoost
        print("\n-> Testing XGBoost...")
        xgb_result = benchmark_xgboost(X_train, X_test, y_train, y_test, 
                                       dataset['type'], **dataset)
        if xgb_result:
            metrics = evaluate_predictions(y_test, xgb_result['y_pred'], dataset['type'])
            results['XGB_Train_Time'] = round(xgb_result['train_time'], 3)
            results['XGB_Predict_Time'] = round(xgb_result['predict_time'], 3)
            for key, value in metrics.items():
                if value is not None:
                    results[f'XGB_{key}'] = round(value, 4)
            print(f"   Train: {xgb_result['train_time']:.3f}s | Predict: {xgb_result['predict_time']:.3f}s")
        else:
            print("   Not available")
        
        # LightGBM
        print("-> Testing LightGBM...")
        lgb_result = benchmark_lightgbm(X_train, X_test, y_train, y_test,
                                        dataset['type'], **dataset)
        if lgb_result:
            metrics = evaluate_predictions(y_test, lgb_result['y_pred'], dataset['type'])
            results['LGB_Train_Time'] = round(lgb_result['train_time'], 3)
            results['LGB_Predict_Time'] = round(lgb_result['predict_time'], 3)
            for key, value in metrics.items():
                if value is not None:
                    results[f'LGB_{key}'] = round(value, 4)
            print(f"   Train: {lgb_result['train_time']:.3f}s | Predict: {lgb_result['predict_time']:.3f}s")
        else:
            print("   Not available")
        
        # CatBoost
        print("-> Testing CatBoost...")
        cb_result = benchmark_catboost(X_train, X_test, y_train, y_test,
                                       dataset['type'], **dataset)
        if cb_result:
            metrics = evaluate_predictions(y_test, cb_result['y_pred'], dataset['type'])
            results['CB_Train_Time'] = round(cb_result['train_time'], 3)
            results['CB_Predict_Time'] = round(cb_result['predict_time'], 3)
            for key, value in metrics.items():
                if value is not None:
                    results[f'CB_{key}'] = round(value, 4)
            print(f"   Train: {cb_result['train_time']:.3f}s | Predict: {cb_result['predict_time']:.3f}s")
        else:
            print("   Not available")
        
        all_results.append(results)
    
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(all_results)
    
    # Mostrar resultados
    print("\n" + "="*80)
    print(" RESULTADOS FINALES ".center(80))
    print("="*80)
    
    # Tabla de tiempos
    print("\n--- TIEMPOS DE ENTRENAMIENTO (segundos) ---")
    time_cols = ['Dataset', 'Type']
    time_cols.extend([c for c in df_results.columns if 'Train_Time' in c])
    if len(time_cols) > 2:
        print(df_results[time_cols].to_string(index=False))
    
    # Tabla de accuracy/RMSE
    print("\n--- MÉTRICAS DE RENDIMIENTO ---")
    metric_cols = ['Dataset', 'Type']
    # Para clasificación: accuracy
    acc_cols = [c for c in df_results.columns if 'accuracy' in c]
    if acc_cols:
        print("\nAccuracy (Clasificación):")
        print(df_results[df_results['Type'] == 'classification'][['Dataset'] + acc_cols].to_string(index=False))
    
    # Para regresión: RMSE
    rmse_cols = [c for c in df_results.columns if 'rmse' in c]
    if rmse_cols:
        print("\nRMSE (Regresión):")
        print(df_results[df_results['Type'] == 'regression'][['Dataset'] + rmse_cols].to_string(index=False))
    
    # Análisis comparativo
    print("\n" + "="*80)
    print(" ANÁLISIS COMPARATIVO ".center(80))
    print("="*80)
    
    # Calcular promedios
    avg_times = {}
    for lib in ['XGB', 'LGB', 'CB']:
        col = f'{lib}_Train_Time'
        if col in df_results.columns:
            avg_times[lib] = df_results[col].mean()
    
    if avg_times:
        print("\nTiempo promedio de entrenamiento:")
        for lib, time_val in sorted(avg_times.items(), key=lambda x: x[1]):
            print(f"  {lib}: {time_val:.3f}s")
    
    # Mejor accuracy promedio
    avg_acc = {}
    for lib in ['XGB', 'LGB', 'CB']:
        col = f'{lib}_accuracy'
        if col in df_results.columns:
            avg_acc[lib] = df_results[col].mean()
    
    if avg_acc:
        print("\nAccuracy promedio (clasificación):")
        for lib, acc_val in sorted(avg_acc.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lib}: {acc_val:.4f}")
    
    # Guardar resultados
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n[OK] Resultados guardados en: {output_file}")
    
    # Resumen final
    print("\n" + "="*80)
    print(" RESUMEN ".center(80))
    print("="*80)
    
    available = [k for k, v in libraries.items() if v]
    print(f"\nLibrerías disponibles: {', '.join(available)}")
    print(f"Datasets evaluados: {len(datasets)}")
    print(f"Total de experimentos: {len(datasets) * len(available)}")
    
    # Determinar ganador general
    if avg_times and avg_acc:
        fastest = min(avg_times.items(), key=lambda x: x[1])[0]
        most_accurate = max(avg_acc.items(), key=lambda x: x[1])[0]
        
        print(f"\n[RESULTADO] Más rápido: {fastest}")
        print(f"[RESULTADO] Más preciso: {most_accurate}")
        
        # Balance speed/accuracy
        scores = {}
        for lib in available:
            lib_abbr = lib[:3].upper()
            if lib_abbr in avg_times and lib_abbr in avg_acc:
                # Normalizar tiempos (menor es mejor) y accuracy (mayor es mejor)
                time_score = 1 - (avg_times[lib_abbr] - min(avg_times.values())) / (max(avg_times.values()) - min(avg_times.values()) + 0.001)
                acc_score = (avg_acc[lib_abbr] - min(avg_acc.values())) / (max(avg_acc.values()) - min(avg_acc.values()) + 0.001)
                # Score combinado (50% velocidad, 50% accuracy)
                scores[lib_abbr] = 0.5 * time_score + 0.5 * acc_score
        
        if scores:
            best_overall = max(scores.items(), key=lambda x: x[1])[0]
            print(f"[RESULTADO] Mejor balance velocidad/precisión: {best_overall}")
    
    return df_results

if __name__ == "__main__":
    results = run_benchmark()