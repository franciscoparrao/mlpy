# Checklist para Publicación en PyPI

## ✅ Archivos de Configuración Creados

- [x] **pyproject.toml** - Configuración moderna del paquete
  - Nombre: `mlpy-framework` (para evitar conflictos)
  - Versión: 0.1.0
  - Dependencias principales y opcionales definidas
  - Metadatos completos (descripción, autores, URLs)
  
- [x] **setup.py** - Compatibilidad con pip antiguo

- [x] **MANIFEST.in** - Incluir archivos adicionales
  - README, LICENSE, CHANGELOG
  - Documentación
  - Ejemplos
  - Tests

- [x] **requirements.txt** - Dependencias principales

- [x] **requirements-dev.txt** - Dependencias de desarrollo

- [x] **.gitignore** - Ignorar archivos no deseados

- [x] **LICENSE** - Licencia MIT

- [x] **CHANGELOG.md** - Historial de cambios

- [x] **CONTRIBUTING.md** - Guía para contribuidores

- [x] **README.md** - Documentación principal con badges

- [x] **mlpy/py.typed** - Marcador para type hints

## 📦 Estructura del Paquete

```
mlpy/
├── pyproject.toml
├── setup.py
├── MANIFEST.in
├── LICENSE
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── requirements-dev.txt
├── mlpy/
│   ├── __init__.py
│   ├── py.typed
│   ├── base.py
│   ├── tasks/
│   ├── learners/
│   ├── measures/
│   ├── resamplings/
│   ├── pipelines/
│   ├── automl/
│   ├── parallel/
│   ├── callbacks/
│   └── visualizations/
├── tests/
├── docs/
├── examples/
│   ├── notebooks/
│   └── scripts/
└── scripts/
    └── build_package.py
```

## 🚀 Pasos para Publicar

### 1. Preparar el Entorno

```bash
# Instalar herramientas necesarias
pip install --upgrade build twine

# Verificar versión en pyproject.toml
# Actualizar CHANGELOG.md
```

### 2. Construir el Paquete

```bash
# Limpiar builds anteriores
rm -rf dist/ build/ *.egg-info

# Construir
python -m build

# Verificar archivos generados
ls -la dist/
```

### 3. Verificar el Paquete

```bash
# Verificar con twine
python -m twine check dist/*

# Instalar localmente para probar
pip install dist/mlpy_framework-0.1.0-py3-none-any.whl
```

### 4. Publicar en TestPyPI (Recomendado)

```bash
# Subir a TestPyPI
python -m twine upload --repository testpypi dist/*

# Instalar desde TestPyPI para verificar
pip install --index-url https://test.pypi.org/simple/ mlpy-framework
```

### 5. Publicar en PyPI

```bash
# Subir a PyPI (producción)
python -m twine upload dist/*
```

## 🔐 Configuración de Credenciales

1. Crear cuenta en [PyPI](https://pypi.org/) y [TestPyPI](https://test.pypi.org/)
2. Generar API tokens
3. Crear archivo `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJG...

[testpypi]
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwIk...
```

## ⚠️ Consideraciones Importantes

1. **Nombre del Paquete**: Usar `mlpy-framework` ya que `mlpy` puede estar tomado
2. **Versión**: Comenzar con 0.1.0 para indicar versión alpha
3. **Dependencias**: Verificar compatibilidad de versiones
4. **Tests**: Asegurar que todos los tests pasen antes de publicar
5. **Documentación**: Verificar que los enlaces funcionen

## 📋 Pre-publicación Checklist

- [ ] Todos los tests pasan (`pytest`)
- [ ] Documentación actualizada
- [ ] CHANGELOG.md actualizado
- [ ] Versión incrementada en pyproject.toml
- [ ] README.md revisado
- [ ] Ejemplos funcionando
- [ ] Build local exitoso
- [ ] Instalación local exitosa
- [ ] TestPyPI publicación exitosa
- [ ] TestPyPI instalación exitosa

## 🎉 Post-publicación

1. Crear release en GitHub
2. Actualizar documentación en Read the Docs
3. Anunciar en redes sociales/comunidad
4. Monitorear issues y feedback