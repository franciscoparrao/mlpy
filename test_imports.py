"""
Script para diagnosticar problemas de importacion de wrappers
"""
import sys
import os

# Asegurar que MLPY esta en el path
mlpy_path = r'C:\Users\gran_\Documents\Proyectos\MLPY'
if mlpy_path not in sys.path:
    sys.path.insert(0, mlpy_path)

print("="*60)
print("DIAGNOSTICO DE IMPORTS DE WRAPPERS")
print("="*60)

# 1. Intentar importar CatBoost directamente
print("\n1. IMPORT DIRECTO DE CATBOOST:")
try:
    import catboost
    print(f"  [OK] CatBoost {catboost.__version__} importado")
except ImportError as e:
    print(f"  [ERROR] No se puede importar catboost: {e}")

# 2. Intentar importar el wrapper directamente
print("\n2. IMPORT DIRECTO DEL WRAPPER:")
try:
    from mlpy.learners.catboost_wrapper import learner_catboost, LearnerCatBoostClassif
    print("  [OK] Wrapper importado directamente")
    print(f"  [OK] learner_catboost disponible: {learner_catboost}")
    print(f"  [OK] LearnerCatBoostClassif disponible: {LearnerCatBoostClassif}")
except Exception as e:
    print(f"  [ERROR] No se puede importar wrapper: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 3. Intentar importar desde mlpy.learners
print("\n3. IMPORT DESDE mlpy.learners:")
try:
    from mlpy.learners import learner_catboost
    print("  [OK] learner_catboost importado desde mlpy.learners")
except ImportError as e:
    print(f"  [ERROR] No se puede importar learner_catboost: {e}")
except AttributeError as e:
    print(f"  [ERROR] AttributeError: {e}")

# 4. Lo mismo para XGBoost
print("\n4. PRUEBA DE XGBOOST:")
try:
    from mlpy.learners.xgboost_wrapper import learner_xgboost
    print("  [OK] XGBoost wrapper importado directamente")
except Exception as e:
    print(f"  [ERROR] XGBoost wrapper: {e}")

try:
    from mlpy.learners import learner_xgboost
    print("  [OK] learner_xgboost desde mlpy.learners")
except ImportError as e:
    print(f"  [ERROR] learner_xgboost desde mlpy.learners: {e}")

# 5. Lo mismo para LightGBM  
print("\n5. PRUEBA DE LIGHTGBM:")
try:
    from mlpy.learners.lightgbm_wrapper import learner_lightgbm
    print("  [OK] LightGBM wrapper importado directamente")
except Exception as e:
    print(f"  [ERROR] LightGBM wrapper: {e}")

try:
    from mlpy.learners import learner_lightgbm
    print("  [OK] learner_lightgbm desde mlpy.learners")
except ImportError as e:
    print(f"  [ERROR] learner_lightgbm desde mlpy.learners: {e}")

# 6. Verificar que estan en __all__
print("\n6. VERIFICAR __all__ en mlpy.learners:")
try:
    import mlpy.learners
    print(f"  Contenido de __all__: {len(mlpy.learners.__all__)} elementos")
    
    # Buscar elementos de boosting
    boosting_items = [item for item in mlpy.learners.__all__ 
                      if 'xgboost' in item.lower() or 
                         'lightgbm' in item.lower() or 
                         'catboost' in item.lower()]
    
    if boosting_items:
        print(f"  [OK] Elementos de boosting en __all__: {boosting_items}")
    else:
        print("  [ERROR] No hay elementos de boosting en __all__")
        
    # Verificar flags
    print(f"\n  _HAS_XGBOOST: {getattr(mlpy.learners, '_HAS_XGBOOST', 'No definido')}")
    print(f"  _HAS_LIGHTGBM: {getattr(mlpy.learners, '_HAS_LIGHTGBM', 'No definido')}")
    print(f"  _HAS_CATBOOST: {getattr(mlpy.learners, '_HAS_CATBOOST', 'No definido')}")
    
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "="*60)
print("FIN DEL DIAGNOSTICO")
print("="*60)