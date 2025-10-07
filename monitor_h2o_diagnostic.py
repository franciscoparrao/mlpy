"""
Monitor para el diagnóstico de H2O en ejecución.
"""

import time
import os
from datetime import datetime

print("="*60)
print("MONITOR DE DIAGNÓSTICO H2O")
print("="*60)
print("Monitoreando el proceso de diagnóstico...")
print("Presiona Ctrl+C para detener\n")

start_time = time.time()
phases = [
    "1. INVESTIGANDO PROBLEMA MULTICLASE",
    "2. INVESTIGANDO H2O GLM EN REGRESIÓN", 
    "3. INVESTIGANDO PROBLEMA CON R²",
    "4. VERIFICANDO H2O DEEP LEARNING",
    "RESUMEN DE HALLAZGOS"
]

current_phase = 0

while True:
    try:
        elapsed = time.time() - start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        
        # Verificar proceso Java
        java_running = os.system('tasklist | findstr /i "java.exe" > nul 2>&1') == 0
        python_running = os.system('tasklist | findstr /i "python.exe" > nul 2>&1') == 0
        
        # Estado estimado
        phase_estimate = min(int(elapsed / 120), len(phases) - 1)  # ~2 min por fase
        
        print(f"\r[{mins:02d}:{secs:02d}] Java: {'✓' if java_running else '✗'} | "
              f"Python: {'✓' if python_running else '✗'} | "
              f"Fase estimada: {phases[phase_estimate]}", 
              end="", flush=True)
        
        # Si ningún proceso está activo, probablemente terminó
        if not java_running and elapsed > 30:
            print(f"\n\n¡Parece que el diagnóstico terminó! (Java no está activo)")
            break
            
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nMonitoreo detenido.")
        break

print(f"\nTiempo total: {mins:02d}:{secs:02d}")