"""
Script de diagnostico para problemas con XGBoost, LightGBM y CatBoost
"""
import sys
import os

print("="*60)
print("DIAGNOSTICO DE LIBRERIAS DE BOOSTING")
print("="*60)

# 1. Verificar Python y paths
print("\n1. INFORMACION DEL SISTEMA:")
print(f"Python: {sys.version}")
print(f"Python ejecutable: {sys.executable}")
print(f"Directorio actual: {os.getcwd()}")

# 2. Verificar sys.path
print("\n2. SYS.PATH:")
for i, p in enumerate(sys.path[:5]):
    print(f"  [{i}] {p}")

# 3. Intentar imports directos
print("\n3. PRUEBA DE IMPORTS DIRECTOS:")

# XGBoost
print("\n  XGBoost:")
try:
    import xgboost as xgb
    print(f"    [OK] import xgboost exitoso")
    print(f"    [OK] version: {xgb.__version__}")
    print(f"    [OK] ubicacion: {xgb.__file__}")
    
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=10)
    print(f"    [OK] XGBClassifier creado exitosamente")
except Exception as e:
    print(f"    [ERROR] Error: {type(e).__name__}: {e}")

# LightGBM  
print("\n  LightGBM:")
try:
    import lightgbm as lgb
    print(f"    [OK] import lightgbm exitoso")
    print(f"    [OK] version: {lgb.__version__}")
    print(f"    [OK] ubicacion: {lgb.__file__}")
    
    from lightgbm import LGBMClassifier
    clf = LGBMClassifier(n_estimators=10)
    print(f"    [OK] LGBMClassifier creado exitosamente")
except Exception as e:
    print(f"    [ERROR] Error: {type(e).__name__}: {e}")

# CatBoost
print("\n  CatBoost:")
try:
    import catboost as cb
    print(f"    [OK] import catboost exitoso")
    print(f"    [OK] version: {cb.__version__}")
    print(f"    [OK] ubicacion: {cb.__file__}")
    
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=10, verbose=False)
    print(f"    [OK] CatBoostClassifier creado exitosamente")
except Exception as e:
    print(f"    [ERROR] Error: {type(e).__name__}: {e}")

# 4. Verificar MLPY
print("\n4. PRUEBA DE MLPY:")
try:
    # Asegurar que MLPY está en el path
    mlpy_path = r'C:\Users\gran_\Documents\Proyectos\MLPY'
    if mlpy_path not in sys.path:
        sys.path.insert(0, mlpy_path)
        print(f"  → Agregado al path: {mlpy_path}")
    
    import mlpy
    print(f"  [OK] MLPY importado: {mlpy.__file__}")
    
    # Intentar importar learner_sklearn
    from mlpy.learners import learner_sklearn
    print(f"  [OK] learner_sklearn importado")
    
    # Intentar crear learners con sklearn wrapper
    print("\n5. CREACION DE LEARNERS CON learner_sklearn:")
    
    if 'xgb' in locals():
        from xgboost import XGBClassifier
        xgb_learner = learner_sklearn(
            XGBClassifier(n_estimators=10),
            id='xgboost_test'
        )
        print(f"  [OK] XGBoost learner creado: {xgb_learner.id}")
    
    if 'lgb' in locals():
        from lightgbm import LGBMClassifier
        lgb_learner = learner_sklearn(
            LGBMClassifier(n_estimators=10, verbose=-1),
            id='lightgbm_test'
        )
        print(f"  [OK] LightGBM learner creado: {lgb_learner.id}")
    
    if 'cb' in locals():
        from catboost import CatBoostClassifier
        cb_learner = learner_sklearn(
            CatBoostClassifier(iterations=10, verbose=False),
            id='catboost_test'
        )
        print(f"  [OK] CatBoost learner creado: {cb_learner.id}")
        
except Exception as e:
    print(f"  [ERROR] Error con MLPY: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("FIN DEL DIAGNOSTICO")
print("="*60)