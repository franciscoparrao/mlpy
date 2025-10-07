"""
Ejemplo de serialización robusta en MLPY.

Demuestra cómo guardar y cargar modelos/pipelines de forma confiable
con validación de integridad y múltiples formatos de respaldo.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys
import os

# Agregar el directorio padre al path para importar mlpy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.serialization.robust_serializer import RobustSerializer
from mlpy.learners import LearnerClassifSklearn
from mlpy.tasks import TaskClassif
from sklearn.ensemble import RandomForestClassifier

print("=== Ejemplo de Serialización Robusta MLPY ===\n")

# 1. Crear datos de ejemplo y entrenar modelo
print("1. Preparando datos y entrenando modelo...")
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'target': np.random.choice(['A', 'B'], 100)
})

task = TaskClassif(data=data, target='target', id='demo_task')
learner = LearnerClassifSklearn(
    estimator=RandomForestClassifier(n_estimators=10, random_state=42)
)

# Entrenar el modelo
learner.train(task)
print("   Modelo entrenado exitosamente")

print("\n" + "="*60 + "\n")

# 2. Guardar con serialización robusta
print("2. Guardando modelo con RobustSerializer...")

serializer = RobustSerializer()
temp_dir = tempfile.mkdtemp()
model_path = Path(temp_dir) / "modelo_robusto.pkl"

# Guardar con metadata
metadata = serializer.save(
    obj=learner,
    path=model_path,
    metadata={
        'experiment': 'demo_serialization',
        'accuracy': 0.95,  # Simulado
        'author': 'Dhyana'
    }
)

print(f"   SUCCESS: Modelo guardado en: {model_path}")
print(f"   Formato usado: {metadata.get('format', 'pickle')}")
print(f"   Checksum SHA256: {metadata.get('checksum', 'N/A')[:16]}...")
print(f"   Metadata preservada: {list(metadata.keys())}")

print("\n" + "="*60 + "\n")

# 3. Cargar y validar integridad
print("3. Cargando modelo y validando integridad...")

try:
    # Cargar con validación de checksum
    loaded_learner = serializer.load(
        path=model_path,
        validate_checksum=True
    )
    print("   SUCCESS: Modelo cargado exitosamente")
    print("   SUCCESS: Checksum validado - integridad confirmada")
    
    # Verificar que el modelo funciona
    predictions = loaded_learner.predict(task)
    print(f"   SUCCESS: Modelo puede hacer predicciones: {len(predictions.response)} predicciones generadas")
    
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*60 + "\n")

# 4. Demostrar formatos disponibles
print("4. Formatos de serialización disponibles:")
print(f"   Formatos soportados: {RobustSerializer.SUPPORTED_FORMATS}")

# Intentar guardar en diferentes formatos
for format in ['pickle', 'json']:
    try:
        format_path = Path(temp_dir) / f"modelo.{format}"
        if format == 'json':
            # JSON solo para objetos simples
            simple_obj = {'modelo': 'RandomForest', 'params': {'n_estimators': 10}}
            serializer.save(simple_obj, format_path, format=format)
        else:
            serializer.save(learner, format_path, format=format)
        print(f"   SUCCESS: Guardado en formato {format}")
    except Exception as e:
        print(f"   INFO: Formato {format} - {str(e)[:50]}...")

print("\n" + "="*60 + "\n")

# 5. Metadata automática
print("5. Metadata automática capturada:")
if model_path.with_suffix('.meta.json').exists():
    import json
    with open(model_path.with_suffix('.meta.json'), 'r') as f:
        meta = json.load(f)
    print(f"   Timestamp: {meta.get('timestamp', 'N/A')}")
    print(f"   MLPY Version: {meta.get('mlpy_version', 'N/A')}")
    print(f"   Object Type: {meta.get('object_type', 'N/A')}")
    print(f"   User Metadata: {meta.get('user_metadata', {})}")
else:
    print("   INFO: Archivo de metadata no encontrado")

print("\n" + "="*60 + "\n")

# 6. Ventajas de la serialización robusta
print("6. Ventajas de RobustSerializer vs pickle tradicional:")
print("   + Validación de integridad con checksums")
print("   + Metadata automática para trazabilidad")
print("   + Múltiples formatos con fallback automático")
print("   + Compresión opcional para ahorrar espacio")
print("   + Preparado para deployment (ONNX cuando disponible)")
print("   + Mensajes de error claros y útiles")

print("\n" + "="*60 + "\n")

print("RESUMEN: La serialización robusta de MLPY garantiza:")
print("  - Modelos nunca corrompidos")
print("  - Trazabilidad completa")
print("  - Compatibilidad multi-entorno")
print("  - Confianza en producción")
print("\nNamaste - El modelo persiste como la consciencia")

# Limpiar archivos temporales
import shutil
shutil.rmtree(temp_dir, ignore_errors=True)