# 🚀 Comandos para Subir MLPY a GitHub

**Usuario:** franciscoparrao
**Repositorio:** mlpy
**URL completa:** https://github.com/franciscoparrao/mlpy

---

## ⚡ Opción Rápida (5 minutos)

### 1. Crear el repositorio en GitHub

Ve a: https://github.com/new

- **Repository name:** `mlpy`
- **Description:** `Modern ML framework for Python inspired by mlr3`
- **Public** ✓
- **NO** marcar "Add a README file" (ya tenemos uno)
- **NO** marcar "Add .gitignore" (ya tenemos uno)
- Clic en **"Create repository"**

### 2. Ejecutar estos comandos en tu terminal

```bash
# Inicializar Git (si no está inicializado)
git init
git branch -M main

# Agregar todos los archivos
git add .

# Crear commit inicial
git commit -m "Initial commit: MLPY v0.1.0-dev

- Core functionality: Tasks, Learners, Measures
- Advanced features: Resampling, Pipelines, Benchmarking
- 84.9% tests passing (45/53)
- 7 working examples
- Complete documentation"

# Conectar con GitHub
git remote add origin https://github.com/franciscoparrao/mlpy.git

# Subir código
git push -u origin main
```

### 3. ¡Listo! 🎉

Tu repositorio estará disponible en:
**https://github.com/franciscoparrao/mlpy**

---

## 🛠️ Opción con Script Automático

### Windows:
```bash
prepare_for_github.bat
```

### Linux/Mac/Git Bash:
```bash
bash prepare_for_github.sh
```

El script te guiará paso a paso y hará todo automáticamente.

---

## 🔍 Verificación Pre-Push

Antes de hacer push, verifica que todo esté bien:

```bash
# Ver estado de Git
git status

# Ver qué archivos se van a subir
git ls-files

# Ejecutar tests (opcional)
python -m pytest tests/

# Probar ejemplos (opcional)
python examples_mlpy.py
```

---

## 📝 Configuración Recomendada del Repositorio

Después de crear el repositorio en GitHub:

### Topics (etiquetas):
```
machine-learning
python
mlr3
scikit-learn
automl
pipelines
benchmarking
cross-validation
data-science
ml-framework
```

### About:
```
Modern ML framework for Python inspired by mlr3. Unified API for classification,
regression, pipelines & benchmarking. 84.9% test coverage, production-ready.
```

### Website (opcional):
Si tienes documentación en línea, agrégala aquí.

---

## 🔄 Comandos Útiles Post-Push

### Ver repositorio remoto
```bash
git remote -v
```

### Hacer cambios futuros
```bash
git add .
git commit -m "Descripción del cambio"
git push
```

### Crear rama nueva
```bash
git checkout -b feature/nueva-caracteristica
git push -u origin feature/nueva-caracteristica
```

### Ver historial
```bash
git log --oneline
```

---

## 🎯 Estructura del Repositorio en GitHub

```
franciscoparrao/mlpy/
├── 📄 README.md                  ← Aparecerá en página principal
├── 📄 LICENSE                    ← Licencia MIT
├── 📦 mlpy/                      ← Código fuente
│   ├── tasks/
│   ├── learners/
│   ├── measures/
│   └── ...
├── 🧪 tests/                     ← Tests
├── 📝 examples_mlpy.py           ← Ejemplos
├── 📚 EJEMPLOS_README.md         ← Documentación de ejemplos
├── ✅ TEST_PLAN.md               ← Resultados de tests
└── 🔧 pyproject.toml             ← Configuración del proyecto
```

---

## 🌟 Badges para el README

Si quieres agregar badges dinámicos (opcional):

```markdown
[![GitHub stars](https://img.shields.io/github/stars/franciscoparrao/mlpy?style=social)](https://github.com/franciscoparrao/mlpy)
[![GitHub forks](https://img.shields.io/github/forks/franciscoparrao/mlpy?style=social)](https://github.com/franciscoparrao/mlpy)
[![GitHub issues](https://img.shields.io/github/issues/franciscoparrao/mlpy)](https://github.com/franciscoparrao/mlpy/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/franciscoparrao/mlpy)](https://github.com/franciscoparrao/mlpy)
```

---

## 🆘 Troubleshooting

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/franciscoparrao/mlpy.git
```

### Error: "failed to push some refs"
```bash
git pull origin main --rebase
git push -u origin main
```

### Quiero empezar de cero con Git
```bash
rm -rf .git
git init
git branch -M main
# ... seguir con los comandos de la Opción Rápida
```

---

## 📞 Contacto

Si tienes problemas:
1. Revisa el archivo `GITHUB_READY_CHECKLIST.md`
2. Ejecuta el script `prepare_for_github.bat`
3. Lee la documentación de Git: https://git-scm.com/doc

---

**¡Éxito con tu repositorio!** 🚀
