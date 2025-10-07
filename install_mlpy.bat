@echo off
echo ========================================
echo Instalando MLPY en modo desarrollo
echo ========================================
echo.

REM Cambiar al directorio de MLPY
cd /d "C:\Users\gran_\Documents\Proyectos\MLPY"

REM Instalar en modo editable (-e)
echo Instalando MLPY...
pip install -e .

echo.
echo ========================================
echo Instalacion completada!
echo ========================================
echo.
echo Ahora puedes usar MLPY desde cualquier ubicacion:
echo   - import mlpy
echo   - mlpy --help
echo.
pause