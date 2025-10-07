# GitHub Actions Setup para MLPY

Este documento explica cómo configurar y usar GitHub Actions para CI/CD en el proyecto MLPY.

## 📋 Descripción General

MLPY incluye una configuración completa de GitHub Actions que proporciona:

- ✅ **Pruebas automatizadas** en múltiples OS y versiones de Python
- 🔍 **Análisis de calidad de código** (linting, formateo, type checking)
- 📚 **Construcción automática de documentación**
- 🚀 **Deployment automático a PyPI**
- 📊 **Benchmarks de rendimiento**
- 🔒 **Análisis de seguridad**

## 🛠️ Workflows Incluidos

### 1. CI/CD Principal (`ci.yml`)

**Trigger:** Push a `main`/`develop`, Pull Requests
**Funciones:**
- Pruebas en Ubuntu, Windows, macOS
- Python 3.8-3.12
- Linting con flake8, black, isort
- Type checking con mypy
- Coverage testing
- Build y deployment a PyPI en releases

### 2. Documentación (`docs.yml`)

**Trigger:** Cambios en `docs/`, `mlpy/`, archivos `.md`
**Funciones:**
- Construcción de documentación con Sphinx
- Deployment a GitHub Pages
- Verificación de enlaces

### 3. Calidad de Código (`quality.yml`)

**Trigger:** Push y Pull Requests
**Funciones:**
- Análisis profundo con pylint
- Verificación de docstrings
- Revisión de ortografía
- Análisis de seguridad con bandit
- Pre-commit hooks

### 4. Releases (`release.yml`)

**Trigger:** Tags `v*.*.*`
**Funciones:**
- Pruebas completas antes del release
- Generación automática de changelog
- Creación de GitHub Release
- Publicación a PyPI

### 5. Benchmarks (`benchmarks.yml`)

**Trigger:** Cambios en código, schedule semanal
**Funciones:**
- Pruebas de rendimiento
- Benchmarks comparativos
- Reportes de performance

## 🔧 Configuración Inicial

### 1. Secrets Requeridos

Configura estos secrets en GitHub (Settings → Secrets and variables → Actions):

```bash
PYPI_API_TOKEN=pypi-...  # Token de PyPI para deployment
```

### 2. Configuración del Repositorio

1. **Habilita GitHub Pages:**
   - Ve a Settings → Pages
   - Source: GitHub Actions

2. **Configura Branch Protection (Recomendado):**
   - Settings → Branches
   - Add rule para `main`
   - Require status checks: ✅
   - Require branches to be up to date: ✅

### 3. Pre-commit Hooks (Opcional pero Recomendado)

```bash
# Instalar pre-commit
pip install pre-commit

# Instalar hooks
pre-commit install

# Ejecutar en todos los archivos
pre-commit run --all-files
```

## 📈 Uso de los Workflows

### Desarrollo Diario

1. **Push a branch:** Ejecuta linting y pruebas básicas
2. **Pull Request:** Ejecuta suite completa de CI/CD
3. **Merge a main:** Ejecuta todos los checks + documentación

### Releases

1. **Crear tag de versión:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Automáticamente:**
   - Se ejecutan todas las pruebas
   - Se crea GitHub Release
   - Se publica a PyPI

### Monitoreo

- **Actions tab:** Ver estado de todos los workflows
- **Pull Requests:** Ver checks automáticos
- **Releases:** Ver deployment status

## 🏷️ Badges Recomendados

Agrega estos badges al README.md:

```markdown
[![CI/CD](https://github.com/YOUR_USERNAME/MLPY/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/YOUR_USERNAME/MLPY/actions)
[![Documentation](https://github.com/YOUR_USERNAME/MLPY/workflows/Documentation/badge.svg)](https://YOUR_USERNAME.github.io/MLPY/)
[![PyPI version](https://badge.fury.io/py/mlpy.svg)](https://badge.fury.io/py/mlpy)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## 🔍 Configuración de Herramientas

### pytest
Configurado en `pytest.ini`:
- Coverage mínimo: 70%
- Markers para diferentes tipos de tests
- Reportes en XML y HTML

### mypy
Configurado en `mypy.ini`:
- Type checking gradual
- Ignora imports faltantes de third-party
- Configuración específica por módulo

### pre-commit
Configurado en `.pre-commit-config.yaml`:
- black (formateo)
- isort (imports)
- flake8 (linting)
- bandit (seguridad)
- spell checking

## 🚨 Solución de Problemas

### Error: "PYPI_API_TOKEN not found"
- Configura el secret en GitHub Settings
- Verifica que el nombre sea exacto

### Tests fallan en Windows
- Los tests de Windows pueden ser flaky
- Configurado con `continue-on-error` para Windows

### Documentación no se construye
- Verifica que todas las dependencias estén en `docs/requirements.txt`
- Revisa errores de Sphinx en los logs

### Pre-commit hooks fallan
- Ejecuta `pre-commit run --all-files` localmente
- Corrige errores de formateo antes del push

## 📊 Métricas y Reportes

Los workflows generan:

- **Coverage reports:** `htmlcov/index.html`
- **Type checking:** Logs de mypy
- **Security reports:** Bandit JSON reports
- **Benchmark results:** Performance comparisons
- **Documentation:** Hosted en GitHub Pages

## 🔄 Dependabot

Configurado en `.github/dependabot.yml`:
- Actualización semanal de dependencias Python
- Actualización semanal de GitHub Actions
- PRs automáticos con etiquetas apropiadas

## 🎯 Mejores Prácticas

1. **Commits pequeños y frecuentes**
2. **Usar conventional commits** (feat:, fix:, docs:)
3. **Revisar checks antes de merge**
4. **Mantener coverage > 70%**
5. **Usar pre-commit hooks**
6. **Documentar cambios importantes**

## 🔗 Enlaces Útiles

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Pre-commit Hooks](https://pre-commit.com/)
- [Semantic Versioning](https://semver.org/)

---

¡Los GitHub Actions están configurados y listos para usar! 🚀