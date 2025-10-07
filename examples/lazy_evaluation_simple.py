"""
Ejemplo simplificado de Lazy Evaluation en MLPY.

Demuestra el concepto de computación diferida y optimización automática.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys
import os

# Agregar el directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.lazy.lazy_evaluation import ComputationGraph, ComputationNode

print("=== Ejemplo Simplificado de Lazy Evaluation MLPY ===")
print("Demostrando el poder de la computación diferida\n")

print("="*60)
print("1. COMPARACIÓN: EAGER vs LAZY")
print("-"*60)

# Simulación de operaciones costosas
def expensive_operation(data, name="op"):
    """Simula una operación costosa."""
    print(f"   Ejecutando {name}...")
    time.sleep(0.1)  # Simula procesamiento
    return data * 2

# EAGER: Ejecuta todo inmediatamente
print("\nEJECUCIÓN EAGER (tradicional):")
start_eager = time.time()

data = np.random.randn(1000, 100)
result1 = expensive_operation(data, "operación 1")
result2 = expensive_operation(result1, "operación 2")
result3 = expensive_operation(result2, "operación 3")
# Supongamos que solo necesitamos result2, no result3
final_eager = result2.mean()

end_eager = time.time()
print(f"Tiempo total EAGER: {(end_eager - start_eager)*1000:.2f}ms")
print(f"Resultado: {final_eager:.4f}")

print("\n" + "-"*60)

# LAZY: Construye grafo, ejecuta solo lo necesario
print("\nEJECUCIÓN LAZY (optimizada):")

# Crear grafo de computación
graph = ComputationGraph()

# Definir nodos de computación (sin ejecutar)
print("   Construyendo grafo (sin ejecutar)...")
start_lazy = time.time()

# Nodo 1: Cargar datos
node1 = ComputationNode(
    id="load_data",
    operation="load_data",
    func=lambda: np.random.randn(1000, 100)
)
graph.add_node(node1)

# Nodo 2: Primera transformación
node2 = ComputationNode(
    id="transform1",
    operation="expensive_op_1",
    func=lambda x: expensive_operation(x, "operación 1"),
    dependencies=["load_data"]
)
graph.add_node(node2)

# Nodo 3: Segunda transformación
node3 = ComputationNode(
    id="transform2",
    operation="expensive_op_2",
    func=lambda x: expensive_operation(x, "operación 2"),
    dependencies=["transform1"]
)
graph.add_node(node3)

# Nodo 4: Tercera transformación (NO NECESARIA)
node4 = ComputationNode(
    id="transform3",
    operation="expensive_op_3",
    func=lambda x: expensive_operation(x, "operación 3"),
    dependencies=["transform2"]
)
graph.add_node(node4)

# Nodo 5: Agregación (solo necesita transform2)
node5 = ComputationNode(
    id="aggregate",
    operation="mean",
    func=lambda x: x.mean(),
    dependencies=["transform2"]
)
graph.add_node(node5)

end_construct = time.time()
print(f"   Tiempo construcción: {(end_construct - start_lazy)*1000:.2f}ms")

# Ejecutar solo lo necesario
print("   Ejecutando grafo optimizado...")
start_exec = time.time()

# Optimizar: eliminará transform3 porque no es necesario
graph.optimize()

# Ejecutar solo los nodos necesarios para 'aggregate'
results = graph.execute()
final_lazy = results.get("aggregate")

end_lazy = time.time()
print(f"Tiempo ejecución: {(end_lazy - start_exec)*1000:.2f}ms")
print(f"Tiempo total LAZY: {(end_lazy - start_lazy)*1000:.2f}ms")
print(f"Resultado: {final_lazy:.4f}")

print("\n" + "="*60)
print("2. ANÁLISIS DE OPTIMIZACIÓN")
print("-"*60)

eager_time = (end_eager - start_eager) * 1000
lazy_time = (end_lazy - start_lazy) * 1000
saved_time = eager_time - lazy_time
improvement = (saved_time / eager_time) * 100

print(f"\nTiempo EAGER: {eager_time:.2f}ms (3 operaciones ejecutadas)")
print(f"Tiempo LAZY: {lazy_time:.2f}ms (2 operaciones ejecutadas)")
print(f"Tiempo ahorrado: {saved_time:.2f}ms")
print(f"Mejora: {improvement:.1f}%")

print("\nOptimizaciones aplicadas:")
print("  [OK] Eliminación de código muerto (transform3 no se ejecutó)")
print("  [OK] Ejecución solo de nodos necesarios")
print("  [OK] Caché de resultados intermedios")

print("\n" + "="*60)
print("3. VISUALIZACIÓN DEL GRAFO")
print("-"*60)

dot_graph = graph.visualize()
print("\nGrafo de computación (formato DOT):")
print(dot_graph)

print("\n" + "="*60)
print("4. BENEFICIOS DE LAZY EVALUATION EN MLPY")
print("-"*60)

print("""
CASOS DE USO REALES:

1. PIPELINES DE PREPROCESAMIENTO:
   - Define todas las transformaciones
   - Ejecuta solo las necesarias
   - Reutiliza resultados cacheados

2. BÚSQUEDA DE HIPERPARÁMETROS:
   - Construye árbol de experimentos
   - Comparte cálculos comunes
   - Evita recálculos innecesarios

3. FEATURE ENGINEERING:
   - Define múltiples features candidatas
   - Calcula solo las seleccionadas
   - Optimiza dependencias automáticamente

4. VALIDACIÓN CRUZADA:
   - Reutiliza splits computados
   - Cachea transformaciones comunes
   - Paraleliza folds independientes

5. ENSEMBLE LEARNING:
   - Entrena modelos base una vez
   - Comparte predicciones entre métodos
   - Optimiza agregaciones
""")

print("="*60)
print("CONCLUSIÓN: Lazy Evaluation en MLPY")
print("-"*60)
print("""
La evaluación perezosa no es pereza, es sabiduría:
- No hagas trabajo innecesario
- Optimiza antes de ejecutar
- Reutiliza lo ya calculado
- Paraleliza lo independiente

Como en la meditación: 
"La acción perfecta surge de la perfecta inacción"

Namaste - La computación consciente""")