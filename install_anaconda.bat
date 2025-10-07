@echo off
echo Installing MLPY in Anaconda environment...

REM Copy MLPY package to Anaconda site-packages
xcopy /E /I /Y mlpy "C:\Users\gran_\anaconda3\Lib\site-packages\mlpy"

REM Create a .pth file to add current directory to Python path
echo C:\Users\gran_\Documents\Proyectos\MLPY > "C:\Users\gran_\anaconda3\Lib\site-packages\mlpy.pth"

echo.
echo MLPY has been installed in your Anaconda environment!
echo.
echo Testing installation...
C:\Users\gran_\anaconda3\python.exe -c "import mlpy; print('MLPY version:', mlpy.__version__)"

pause