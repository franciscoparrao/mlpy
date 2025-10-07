"""
Monitor de progreso para el benchmark H2O.
Ejecuta este script en otra terminal para ver el progreso.
"""

import time
import os
from datetime import datetime

# Archivo a monitorear (si el benchmark guarda logs)
LOG_FILE = "examples/h2o_benchmark.log"  # Ajusta si es necesario

print("="*60)
print("MONITOR DE BENCHMARK H2O")
print("="*60)
print("Presiona Ctrl+C para detener el monitoreo")
print("")

start_time = time.time()
last_size = 0

while True:
    try:
        # Mostrar tiempo transcurrido
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        # Verificar si existe el archivo de resultados
        if os.path.exists("h2o_benchmark_results.md"):
            print(f"\n[{mins:02d}:{secs:02d}] ¡BENCHMARK COMPLETADO!")
            print("Archivo de resultados generado: h2o_benchmark_results.md")
            break
            
        # Verificar procesos Java (H2O)
        java_running = os.system('tasklist | findstr /i "java.exe" > nul 2>&1') == 0
        
        # Estado
        print(f"\r[{mins:02d}:{secs:02d}] H2O: {'Activo' if java_running else 'Inactivo'} | Esperando...", end="", flush=True)
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nMonitoreo detenido por el usuario.")
        break

print("\nMonitoreo finalizado.")