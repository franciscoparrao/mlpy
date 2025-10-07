# Contributing to MLPY

¡Gracias por tu interés en contribuir a MLPY! Este documento proporciona pautas y mejores prácticas para contribuir al proyecto.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [Cómo Contribuir](#cómo-contribuir)
- [Configuración del Entorno](#configuración-del-entorno)
- [Proceso de Desarrollo](#proceso-de-desarrollo)
- [Estilo de Código](#estilo-de-código)
- [Testing](#testing)
- [Documentación](#documentación)
- [Pull Requests](#pull-requests)

## 📜 Código de Conducta

Este proyecto se adhiere a un código de conducta. Al participar, se espera que respetes este código. Por favor, reporta comportamientos inaceptables a mlpy@example.com.

## 🤝 Cómo Contribuir

### Reportar Bugs

1. Verifica que el bug no haya sido reportado previamente en [Issues](https://github.com/mlpy-project/mlpy/issues)
2. Si no existe, crea un nuevo issue incluyendo:
   - Descripción clara del problema
   - Pasos para reproducir
   - Comportamiento esperado vs actual
   - Versión de MLPY y Python
   - Sistema operativo

### Sugerir Mejoras

1. Abre un issue describiendo la mejora
2. Explica por qué sería útil
3. Proporciona ejemplos de uso si es posible

### Contribuir Código

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Realiza tus cambios
4. Añade tests si es necesario
5. Asegúrate de que todos los tests pasen
6. Commit tus cambios (`git commit -m 'Add amazing feature'`)
7. Push a la rama (`git push origin feature/amazing-feature`)
8. Abre un Pull Request

## 🛠️ Configuración del Entorno

### Requisitos

- Python 3.8 o superior
- pip
- git

### Instalación para Desarrollo

```bash
# Clonar tu fork
git clone https://github.com/tu-usuario/mlpy.git
cd mlpy

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar en modo desarrollo con todas las dependencias
pip install -e .[dev,all]

# Instalar pre-commit hooks
pre-commit install
```

## 💻 Proceso de Desarrollo

### 1. Crear una Rama

```bash
# Para nuevas features
git checkout -b feature/nombre-descriptivo

# Para fixes
git checkout -b fix/descripcion-del-bug

# Para documentación
git checkout -b docs/que-se-documenta
```

### 2. Hacer Cambios

- Escribe código claro y bien documentado
- Sigue las convenciones de estilo
- Añade docstrings a todas las funciones públicas
- Actualiza la documentación si es necesario

### 3. Testing

```bash
# Ejecutar todos los tests
pytest

# Ejecutar con cobertura
pytest --cov=mlpy --cov-report=html

# Ejecutar solo tests rápidos
pytest -m "not slow"

# Ejecutar tests de un módulo específico
pytest tests/unit/test_tasks.py
```

### 4. Verificar Calidad del Código

```bash
# Formatear código con black
black mlpy tests

# Ordenar imports
isort mlpy tests

# Verificar estilo
flake8 mlpy tests

# Type checking
mypy mlpy

# O ejecutar todo con pre-commit
pre-commit run --all-files
```

## 🎨 Estilo de Código

### Python

- Seguimos [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Usamos [Black](https://github.com/psf/black) para formateo automático
- Líneas de máximo 100 caracteres
- Docstrings en formato NumPy

### Ejemplo de Docstring

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of function.
    
    Longer description if needed, explaining what the function
    does in more detail.
    
    Parameters
    ----------
    param1 : str
        Description of param1
    param2 : int, default=0
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Examples
    --------
    >>> example_function("test", 42)
    True
    """
    pass
```

### Imports

```python
# Orden de imports:
# 1. Standard library
import os
import sys
from typing import List, Optional

# 2. Third party
import numpy as np
import pandas as pd

# 3. Local
from mlpy.base import MLPYObject
from mlpy.tasks import Task
```

## 🧪 Testing

### Escribir Tests

- Cada nueva funcionalidad debe tener tests
- Los tests deben ser claros y descriptivos
- Usar fixtures de pytest cuando sea apropiado
- Cubrir casos edge y errores

### Estructura de Tests

```python
import pytest
from mlpy.module import MyClass


class TestMyClass:
    """Tests for MyClass."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return {"key": "value"}
    
    def test_initialization(self):
        """Test MyClass initialization."""
        obj = MyClass()
        assert obj is not None
    
    def test_method_with_valid_input(self, sample_data):
        """Test method with valid input."""
        obj = MyClass()
        result = obj.method(sample_data)
        assert result == expected_value
    
    def test_method_with_invalid_input(self):
        """Test method raises error with invalid input."""
        obj = MyClass()
        with pytest.raises(ValueError, match="Invalid input"):
            obj.method(None)
```

## 📚 Documentación

### Actualizar Documentación

- Actualiza docstrings cuando cambies funcionalidad
- Añade ejemplos en la documentación
- Actualiza los tutoriales si es necesario
- Verifica que la documentación se construya correctamente:

```bash
cd docs
make clean
make html
# Abre docs/build/html/index.html en tu navegador
```

### Escribir Tutoriales

Si añades una nueva feature importante:

1. Crea un notebook de ejemplo en `examples/notebooks/`
2. Añade un script de ejemplo en `examples/scripts/`
3. Actualiza la documentación en `docs/source/`

## 🔄 Pull Requests

### Antes de Enviar

- [ ] Los tests pasan localmente
- [ ] El código sigue las convenciones de estilo
- [ ] La documentación está actualizada
- [ ] Los commits tienen mensajes descriptivos
- [ ] La rama está actualizada con main

### Formato de PR

**Título**: Descripción breve y clara

**Descripción**:
```markdown
## Descripción
Explicación de los cambios realizados.

## Motivación
Por qué estos cambios son necesarios.

## Cambios
- Cambio 1
- Cambio 2

## Tests
Descripción de los tests añadidos/modificados.

## Checklist
- [ ] Tests añadidos/actualizados
- [ ] Documentación actualizada
- [ ] Código formateado con black
- [ ] Type hints añadidos
```

### Proceso de Revisión

1. Un maintainer revisará tu PR
2. Puede haber solicitudes de cambios
3. Una vez aprobado, será mergeado a main

## 🎯 Áreas de Contribución

### Contribuciones Bienvenidas

- **Nuevos Learners**: Implementaciones nativas de algoritmos
- **Operadores de Pipeline**: Nuevas transformaciones
- **Medidas**: Métricas de evaluación adicionales
- **Visualizaciones**: Nuevos tipos de gráficos
- **Documentación**: Tutoriales, ejemplos, traducciones
- **Tests**: Mejorar cobertura
- **Performance**: Optimizaciones
- **Bug Fixes**: Siempre bienvenidos

### Ideas de Contribución

Ver [Issues con etiqueta "good first issue"](https://github.com/mlpy-project/mlpy/labels/good%20first%20issue)

## 📮 Contacto

- Issues: [GitHub Issues](https://github.com/mlpy-project/mlpy/issues)
- Discusiones: [GitHub Discussions](https://github.com/mlpy-project/mlpy/discussions)
- Email: mlpy@example.com

¡Gracias por contribuir a MLPY! 🎉