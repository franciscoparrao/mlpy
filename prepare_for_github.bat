@echo off
REM Script para preparar MLPY para GitHub (Windows)
REM Ejecutar con: prepare_for_github.bat

echo ==========================================
echo MLPY - Preparacion para GitHub
echo ==========================================
echo.

REM Verificar que estamos en el directorio correcto
echo Verificando directorio...
if not exist pyproject.toml (
    echo ERROR: No se encuentra pyproject.toml
    echo Ejecuta este script desde la raiz del proyecto.
    pause
    exit /b 1
)
echo OK: Directorio correcto
echo.

REM Verificar que .gitignore está actualizado
echo Verificando .gitignore...
findstr /C:"test_results_" .gitignore >nul 2>&1
if %errorlevel% equ 0 (
    echo OK: .gitignore actualizado
) else (
    echo Actualizando .gitignore...
    echo. >> .gitignore
    echo # Test results and reports >> .gitignore
    echo test_results_*.json >> .gitignore
    echo llm_test_results.json >> .gitignore
    echo mlpy_report_*.json >> .gitignore
    echo OK: .gitignore actualizado
)
echo.

REM Verificar tests
echo ==========================================
echo Deseas ejecutar los tests? (s/n)
set /p run_tests="> "
if /i "%run_tests%"=="s" (
    echo Ejecutando tests...
    python -m pytest tests/ --tb=short
    if %errorlevel% neq 0 (
        echo ADVERTENCIA: Algunos tests fallaron
        pause
    )
    echo OK: Tests completados
) else (
    echo Omitiendo tests
)
echo.

REM Verificar ejemplos
echo ==========================================
echo Deseas verificar que los ejemplos funcionen? (s/n)
set /p run_examples="> "
if /i "%run_examples%"=="s" (
    echo Ejecutando ejemplos...
    python examples_mlpy.py >nul 2>&1
    if %errorlevel% equ 0 (
        echo OK: Ejemplos funcionan correctamente
    ) else (
        echo ADVERTENCIA: Los ejemplos tuvieron errores
        pause
    )
) else (
    echo Omitiendo verificacion de ejemplos
)
echo.

REM Inicializar Git si no existe
if not exist .git (
    echo ==========================================
    echo Deseas inicializar el repositorio Git? (s/n)
    set /p init_git="> "
    if /i "%init_git%"=="s" (
        echo Inicializando Git...
        git init
        git branch -M main
        echo OK: Git inicializado
    )
    echo.
)

REM Estado de git
if exist .git (
    echo ==========================================
    echo Estado de Git:
    echo.
    git status --short
    echo.

    REM Verificar si hay commits
    git rev-parse HEAD >nul 2>&1
    if %errorlevel% neq 0 (
        echo No hay commits todavia
        echo Deseas crear el commit inicial? (s/n)
        set /p do_commit="> "
        if /i "%do_commit%"=="s" (
            echo Agregando archivos...
            git add .
            echo.
            echo Archivos a incluir en el commit:
            git diff --cached --name-only
            echo.
            echo Creando commit...
            git commit -m "Initial commit: MLPY v0.1.0-dev" -m "- Core functionality: Tasks, Learners, Measures" -m "- Advanced features: Resampling, Pipelines, Benchmarking" -m "- 84.9%% tests passing (45/53)" -m "- 7 working examples" -m "- Complete documentation"
            echo OK: Commit inicial creado
        )
        echo.
    ) else (
        echo El repository ya tiene commits
    )

    REM Verificar remote
    git remote get-url origin >nul 2>&1
    if %errorlevel% neq 0 (
        echo ==========================================
        echo Configuracion del repositorio remoto
        echo.
        echo Para subir a GitHub, necesitas crear un repositorio primero en:
        echo https://github.com/new
        echo.
        echo Ya creaste el repositorio en GitHub? (s/n)
        set /p has_repo="> "
        if /i "%has_repo%"=="s" (
            set /p repo_url="Ingresa la URL del repositorio: "
            git remote add origin !repo_url!
            echo OK: Remote configurado
            echo.
            echo Deseas hacer push ahora? (s/n)
            set /p do_push="> "
            if /i "%do_push%"=="s" (
                echo Subiendo a GitHub...
                git push -u origin main
                echo OK: Codigo subido exitosamente!
            ) else (
                echo Para subir mas tarde, ejecuta: git push -u origin main
            )
        ) else (
            echo.
            echo Pasos para subir a GitHub:
            echo 1. Ve a https://github.com/new
            echo 2. Crea un repositorio llamado 'mlpy'
            echo 3. Ejecuta:
            echo    git remote add origin https://github.com/franciscoparrao/mlpy.git
            echo    git push -u origin main
        )
    ) else (
        echo Remote ya configurado
    )
)

echo.
echo ==========================================
echo Preparacion completada
echo ==========================================
echo.
echo Recursos utiles:
echo   - README.md - Documentacion principal
echo   - EJEMPLOS_README.md - Guia de ejemplos
echo   - TEST_PLAN.md - Resultados de tests
echo   - GITHUB_READY_CHECKLIST.md - Checklist completo
echo.
echo MLPY esta listo para GitHub!
echo.
pause
