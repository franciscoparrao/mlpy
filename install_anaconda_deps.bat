@echo off
echo ====================================================
echo Instalando dependencias de MLPY en Anaconda
echo ====================================================

REM Activar Anaconda
call C:\Users\gran_\anaconda3\Scripts\activate.bat

REM Instalar dependencias básicas
echo.
echo Instalando NumPy, Pandas, Scikit-learn...
conda install -y numpy pandas scikit-learn matplotlib seaborn

REM Instalar dependencias adicionales con pip
echo.
echo Instalando dependencias adicionales...
pip install click pyyaml

REM Instalar MLPY en modo desarrollo
echo.
echo Instalando MLPY...
cd C:\Users\gran_\Documents\Proyectos\MLPY
pip install -e .

echo.
echo ====================================================
echo Instalación completada!
echo ====================================================
echo.
echo Para verificar, ejecuta en Python:
echo   import mlpy
echo   from mlpy.tasks import TaskClassifSpatial
echo.
pause