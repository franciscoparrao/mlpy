"""
Script para arreglar imports de Optional en todos los archivos
"""

import os
import re
from pathlib import Path

def fix_optional_imports(file_path):
    """Arregla los imports de Optional en un archivo"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar si el archivo usa Optional
    if 'Optional[' not in content:
        return False
    
    # Verificar si ya importa Optional
    if re.search(r'from typing import.*Optional', content):
        return False
    
    # Buscar el import de typing existente
    typing_import_match = re.search(r'^from typing import (.+)$', content, re.MULTILINE)
    
    if typing_import_match:
        # Agregar Optional al import existente
        current_imports = typing_import_match.group(1)
        new_imports = current_imports + ', Optional'
        new_content = content.replace(
            f'from typing import {current_imports}',
            f'from typing import {new_imports}'
        )
    else:
        # No hay import de typing, buscar dónde insertar
        lines = content.split('\n')
        insert_pos = 0
        
        # Buscar después de docstring y otros imports
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
            elif line and not line.startswith('#') and insert_pos > 0:
                break
        
        # Insertar el import
        lines.insert(insert_pos, 'from typing import Optional')
        new_content = '\n'.join(lines)
    
    # Escribir el archivo corregido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

# Buscar todos los archivos Python en mlpy
mlpy_path = Path('mlpy')
fixed_files = []

for py_file in mlpy_path.glob('**/*.py'):
    if fix_optional_imports(py_file):
        fixed_files.append(py_file)

print(f"Archivos corregidos: {len(fixed_files)}")
for f in fixed_files:
    print(f"  - {f}")