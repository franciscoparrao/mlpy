"""Test de import directo"""

print("1. Import directo desde sklearn.py:")
try:
    from mlpy.learners.sklearn import learner_sklearn
    print("   OK - learner_sklearn importado directamente")
except Exception as e:
    print(f"   FALLO - {e}")

print("\n2. Import desde __init__.py:")
try:
    from mlpy.learners import learner_sklearn
    print("   OK - learner_sklearn importado desde __init__")
except Exception as e:
    print(f"   FALLO - {e}")

print("\n3. Contenido de mlpy.learners:")
import mlpy.learners
print("   Atributos disponibles:")
attrs = [x for x in dir(mlpy.learners) if not x.startswith('_')]
for attr in sorted(attrs):
    print(f"   - {attr}")

print("\n4. Verificar _HAS_SKLEARN:")
print(f"   _HAS_SKLEARN = {mlpy.learners._HAS_SKLEARN}")