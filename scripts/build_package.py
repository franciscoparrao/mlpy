#!/usr/bin/env python
"""
Script para construir y verificar el paquete MLPY para PyPI
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Ejecutar un comando y mostrar el resultado."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Error: {description}")
        if result.stderr:
            print(result.stderr)
        return False
    
    print(f"✅ {description} - Completado")
    return True


def main():
    """Proceso principal de construcción."""
    print("🚀 Iniciando construcción de MLPY para PyPI")
    
    # Cambiar al directorio raíz del proyecto
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # 1. Limpiar builds anteriores
    if not run_command(
        ["python", "-m", "pip", "install", "--upgrade", "build", "twine"],
        "Instalando herramientas de construcción"
    ):
        return 1
    
    # 2. Limpiar directorios de build
    for dir_name in ["dist", "build", "*.egg-info"]:
        run_command(["rm", "-rf", dir_name], f"Limpiando {dir_name}")
    
    # 3. Verificar pyproject.toml
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml no encontrado")
        return 1
    
    # 4. Construir el paquete
    if not run_command(
        ["python", "-m", "build"],
        "Construyendo distribuciones (wheel y sdist)"
    ):
        return 1
    
    # 5. Verificar los archivos generados
    dist_files = list(Path("dist").glob("*"))
    if not dist_files:
        print("❌ Error: No se generaron archivos de distribución")
        return 1
    
    print("\n📁 Archivos generados:")
    for file in dist_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name} ({size_mb:.2f} MB)")
    
    # 6. Verificar con twine
    if not run_command(
        ["python", "-m", "twine", "check", "dist/*"],
        "Verificando archivos de distribución"
    ):
        return 1
    
    # 7. Instrucciones para publicar
    print("\n" + "="*60)
    print("📤 INSTRUCCIONES PARA PUBLICAR")
    print("="*60)
    print("\n1. Para publicar en TestPyPI (recomendado primero):")
    print("   python -m twine upload --repository testpypi dist/*")
    print("\n2. Para instalar desde TestPyPI y probar:")
    print("   pip install --index-url https://test.pypi.org/simple/ mlpy-framework")
    print("\n3. Para publicar en PyPI (producción):")
    print("   python -m twine upload dist/*")
    print("\n⚠️  Asegúrate de tener configuradas las credenciales en ~/.pypirc")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())