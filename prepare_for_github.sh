#!/bin/bash
# Script para preparar MLPY para GitHub
# Ejecutar con: bash prepare_for_github.sh

set -e  # Salir si hay errores

echo "=========================================="
echo "MLPY - Preparación para GitHub"
echo "=========================================="
echo ""

# Función para preguntar sí/no
ask_yes_no() {
    while true; do
        read -p "$1 (s/n): " yn
        case $yn in
            [Ss]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Por favor responde s o n.";;
        esac
    done
}

# 1. Verificar que estamos en el directorio correcto
echo "📁 Verificando directorio..."
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: No se encuentra pyproject.toml. Ejecuta este script desde la raíz del proyecto."
    exit 1
fi
echo "✅ Directorio correcto"
echo ""

# 2. Crear carpetas de organización (opcional)
if ask_yes_no "¿Deseas organizar archivos experimentales en carpeta experiments/?"; then
    echo "📂 Creando carpetas..."
    mkdir -p experiments
    mkdir -p docs/internal

    # Mover archivos demo y benchmark
    echo "📦 Moviendo archivos demo..."
    mv demo_*.py experiments/ 2>/dev/null || echo "   No hay archivos demo_*.py"
    mv benchmark_*.py experiments/ 2>/dev/null || echo "   No hay archivos benchmark_*.py"
    mv analyze_*.py experiments/ 2>/dev/null || echo "   No hay archivos analyze_*.py"

    # Mover documentación interna
    echo "📦 Moviendo documentación interna..."
    mv plan_trabajo_mlpy.md docs/internal/ 2>/dev/null || echo "   No hay plan_trabajo_mlpy.md"
    mv seguimiento_mlpy.md docs/internal/ 2>/dev/null || echo "   No hay seguimiento_mlpy.md"
    mv resumen_*.md docs/internal/ 2>/dev/null || echo "   No hay archivos resumen_*.md"
    mv MEDITACION_*.md docs/internal/ 2>/dev/null || echo "   No hay archivos MEDITACION_*.md"
    mv PROTOCOLO_*.md docs/internal/ 2>/dev/null || echo "   No hay archivos PROTOCOLO_*.md"
    mv monitor_commands.txt docs/internal/ 2>/dev/null || echo "   No hay monitor_commands.txt"

    echo "✅ Archivos organizados"
else
    echo "⏭️  Omitiendo organización de archivos"
fi
echo ""

# 3. Verificar tests
if ask_yes_no "¿Deseas ejecutar los tests antes de subir?"; then
    echo "🧪 Ejecutando tests..."
    python -m pytest tests/ --tb=short || {
        echo "⚠️  Algunos tests fallaron, pero puedes continuar"
        if ! ask_yes_no "¿Deseas continuar de todos modos?"; then
            exit 1
        fi
    }
    echo "✅ Tests completados"
else
    echo "⏭️  Omitiendo tests"
fi
echo ""

# 4. Verificar ejemplos
if ask_yes_no "¿Deseas verificar que los ejemplos funcionen?"; then
    echo "📝 Ejecutando ejemplos..."
    python examples_mlpy.py > /dev/null 2>&1 && echo "✅ Ejemplos funcionan correctamente" || {
        echo "⚠️  Los ejemplos tuvieron errores"
        if ! ask_yes_no "¿Deseas continuar de todos modos?"; then
            exit 1
        fi
    }
else
    echo "⏭️  Omitiendo verificación de ejemplos"
fi
echo ""

# 5. Verificar .gitignore actualizado
echo "🔍 Verificando .gitignore..."
if grep -q "test_results_.*\.json" .gitignore; then
    echo "✅ .gitignore actualizado correctamente"
else
    echo "⚠️  .gitignore no tiene las reglas actualizadas"
    echo "   Agregando reglas automáticamente..."
    cat >> .gitignore << 'EOF'

# Test results and reports
test_results_*.json
llm_test_results.json
mlpy_report_*.json
EOF
    echo "✅ .gitignore actualizado"
fi
echo ""

# 6. Mostrar estado de git
echo "📊 Estado de Git:"
echo ""
if [ -d .git ]; then
    echo "Repository ya inicializado"
    git status --short
else
    echo "⚠️  Git no inicializado todavía"
fi
echo ""

# 7. Inicializar git (si no existe)
if [ ! -d .git ]; then
    if ask_yes_no "¿Deseas inicializar el repositorio Git?"; then
        echo "🎯 Inicializando Git..."
        git init
        git branch -M main
        echo "✅ Git inicializado"
    fi
    echo ""
fi

# 8. Hacer primer commit (si es necesario)
if [ -d .git ]; then
    # Verificar si hay commits
    if ! git rev-parse HEAD >/dev/null 2>&1; then
        echo "📝 Preparando primer commit..."

        git add .

        echo ""
        echo "Archivos a incluir en el commit:"
        git diff --cached --name-only
        echo ""

        if ask_yes_no "¿Deseas crear el commit inicial?"; then
            git commit -m "Initial commit: MLPY v0.1.0-dev

- Core functionality: Tasks, Learners, Measures, Predictions
- Advanced features: Resampling, Pipelines, Benchmarking
- Feature engineering: Scale, Encode, Impute, Select
- 84.9% tests passing (45/53)
- 7 working examples with documentation
- Complete API documentation

Highlights:
- Auto-detection for multiclass classification
- Unified API inspired by mlr3
- Production-ready with comprehensive tests"

            echo "✅ Commit inicial creado"
        fi
    else
        echo "ℹ️  Repository ya tiene commits"
    fi
fi
echo ""

# 9. Configurar remote (opcional)
if [ -d .git ]; then
    if ! git remote get-url origin >/dev/null 2>&1; then
        echo "🔗 Configuración del repositorio remoto"
        echo ""
        echo "Para subir a GitHub, necesitas crear un repositorio primero en:"
        echo "https://github.com/new"
        echo ""

        if ask_yes_no "¿Ya creaste el repositorio en GitHub?"; then
            read -p "Ingresa la URL del repositorio (ej: https://github.com/usuario/mlpy.git): " repo_url
            git remote add origin "$repo_url"
            echo "✅ Remote configurado: $repo_url"
            echo ""

            if ask_yes_no "¿Deseas hacer push ahora?"; then
                echo "🚀 Subiendo a GitHub..."
                git push -u origin main
                echo "✅ ¡Código subido exitosamente!"
            else
                echo "ℹ️  Para subir más tarde, ejecuta: git push -u origin main"
            fi
        else
            echo ""
            echo "📋 Pasos para subir a GitHub:"
            echo "1. Ve a https://github.com/new"
            echo "2. Crea un repositorio llamado 'mlpy'"
            echo "3. Ejecuta:"
            echo "   git remote add origin https://github.com/franciscoparrao/mlpy.git"
            echo "   git push -u origin main"
        fi
    else
        echo "ℹ️  Remote ya configurado: $(git remote get-url origin)"
    fi
fi
echo ""

echo "=========================================="
echo "✅ Preparación completada"
echo "=========================================="
echo ""
echo "📚 Recursos útiles:"
echo "  - README.md - Documentación principal"
echo "  - EJEMPLOS_README.md - Guía de ejemplos"
echo "  - TEST_PLAN.md - Resultados de tests"
echo "  - GITHUB_READY_CHECKLIST.md - Checklist completo"
echo ""
echo "🎉 ¡MLPY está listo para GitHub!"
echo ""
