"""
Visualización de resultados del benchmark de Gradient Boosting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Leer los resultados
df = pd.read_csv('benchmark_results_20250817_181823.csv')

# Crear figura con subplots
fig = plt.figure(figsize=(16, 10))

# 1. Comparación de tiempos de entrenamiento
ax1 = plt.subplot(2, 3, 1)
train_times = df[['Dataset', 'XGB_Train_Time', 'LGB_Train_Time', 'CB_Train_Time']].set_index('Dataset')
train_times.plot(kind='bar', ax=ax1)
ax1.set_title('Training Time Comparison (seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Dataset')
ax1.legend(['XGBoost', 'LightGBM', 'CatBoost'], loc='upper right')
ax1.tick_params(axis='x', rotation=45)

# 2. Comparación de accuracy
ax2 = plt.subplot(2, 3, 2)
acc_cols = [col for col in df.columns if 'accuracy' in col]
if acc_cols:
    accuracy_data = df[df['Type'] == 'classification'][['Dataset'] + acc_cols].set_index('Dataset')
    accuracy_data.plot(kind='bar', ax=ax2)
    ax2.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Dataset')
    ax2.legend(['XGBoost', 'LightGBM', 'CatBoost'], loc='lower right')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0.85, 1.01])

# 3. RMSE para regresión
ax3 = plt.subplot(2, 3, 3)
rmse_cols = [col for col in df.columns if 'rmse' in col]
if rmse_cols:
    rmse_data = df[df['Type'] == 'regression'][['Dataset'] + rmse_cols].set_index('Dataset')
    if not rmse_data.empty:
        rmse_data.plot(kind='bar', ax=ax3)
        ax3.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RMSE')
        ax3.set_xlabel('Dataset')
        ax3.legend(['XGBoost', 'LightGBM', 'CatBoost'], loc='upper right')

# 4. Tiempos promedio (pie chart)
ax4 = plt.subplot(2, 3, 4)
avg_train_times = {
    'XGBoost': df['XGB_Train_Time'].mean(),
    'LightGBM': df['LGB_Train_Time'].mean(),
    'CatBoost': df['CB_Train_Time'].mean()
}
colors = ['#ff9999', '#66b3ff', '#99ff99']
wedges, texts, autotexts = ax4.pie(avg_train_times.values(), 
                                     labels=avg_train_times.keys(),
                                     autopct='%1.1f%%',
                                     colors=colors,
                                     startangle=90)
ax4.set_title('Average Training Time Distribution', fontsize=12, fontweight='bold')

# 5. Speed vs Accuracy tradeoff
ax5 = plt.subplot(2, 3, 5)
# Calcular promedios
libraries = ['XGBoost', 'LightGBM', 'CatBoost']
avg_times = []
avg_accs = []

for lib, prefix in zip(libraries, ['XGB', 'LGB', 'CB']):
    time_col = f'{prefix}_Train_Time'
    acc_col = f'{prefix}_accuracy'
    
    avg_times.append(df[time_col].mean())
    
    # Solo considerar filas de clasificación para accuracy
    class_df = df[df['Type'] == 'classification']
    if acc_col in class_df.columns:
        avg_accs.append(class_df[acc_col].mean())
    else:
        avg_accs.append(0)

scatter_colors = ['red', 'blue', 'green']
for i, (lib, time, acc) in enumerate(zip(libraries, avg_times, avg_accs)):
    ax5.scatter(time, acc, s=200, c=scatter_colors[i], alpha=0.6, edgecolors='black', linewidth=2)
    ax5.annotate(lib, (time, acc), xytext=(5, 5), textcoords='offset points', fontsize=10)

ax5.set_xlabel('Average Training Time (s)')
ax5.set_ylabel('Average Accuracy')
ax5.set_title('Speed vs Accuracy Tradeoff', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Resumen de rendimiento
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Crear tabla de resumen
summary_data = []
for lib, prefix in zip(libraries, ['XGB', 'LGB', 'CB']):
    time_col = f'{prefix}_Train_Time'
    acc_col = f'{prefix}_accuracy'
    
    avg_time = df[time_col].mean()
    class_df = df[df['Type'] == 'classification']
    avg_acc = class_df[acc_col].mean() if acc_col in class_df.columns else 0
    
    # Determinar ranking
    time_rank = 1 + sum(1 for t in avg_times if t < avg_time)
    acc_rank = 1 + sum(1 for a in avg_accs if a > avg_acc)
    
    summary_data.append([
        lib,
        f'{avg_time:.3f}s',
        f'{avg_acc:.4f}',
        f'#{time_rank}',
        f'#{acc_rank}'
    ])

# Crear tabla
table = ax6.table(cellText=summary_data,
                  colLabels=['Library', 'Avg Time', 'Avg Acc', 'Speed Rank', 'Acc Rank'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Colorear headers
for i in range(5):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Colorear mejor rendimiento
for i in range(1, 4):
    # Tiempo más rápido (verde)
    if f'#{1}' in summary_data[i-1][3]:
        table[(i, 3)].set_facecolor('#90EE90')
    # Mejor accuracy (verde)
    if f'#{1}' in summary_data[i-1][4]:
        table[(i, 4)].set_facecolor('#90EE90')

ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

# Título general
fig.suptitle('Gradient Boosting Libraries Benchmark Results', fontsize=16, fontweight='bold', y=1.02)

# Ajustar layout
plt.tight_layout()

# Guardar figura
plt.savefig('benchmark_visualization.png', dpi=150, bbox_inches='tight')
print("[OK] Visualización guardada en: benchmark_visualization.png")

# Mostrar
plt.show()

# Imprimir conclusiones
print("\n" + "="*60)
print("CONCLUSIONES DEL BENCHMARK")
print("="*60)

# Determinar ganadores
fastest = min(avg_times)
fastest_lib = libraries[avg_times.index(fastest)]

most_accurate = max(avg_accs)
most_accurate_lib = libraries[avg_accs.index(most_accurate)]

print(f"\n[RESULTADOS CLAVE]:")
print(f"   - Más rápido: {fastest_lib} ({fastest:.3f}s promedio)")
print(f"   - Más preciso: {most_accurate_lib} ({most_accurate:.4f} accuracy promedio)")

# Análisis por tipo de dataset
print("\n[RENDIMIENTO POR TIPO DE DATASET]:")

# Dataset pequeño
small_data = df[df['Dataset'] == 'Binary_Small']
if not small_data.empty:
    times = [small_data['XGB_Train_Time'].values[0], 
             small_data['LGB_Train_Time'].values[0],
             small_data['CB_Train_Time'].values[0]]
    fastest_small = libraries[times.index(min(times))]
    print(f"   - Dataset pequeño: {fastest_small} fue más rápido")

# Dataset con categóricas
cat_data = df[df['Dataset'] == 'Mixed_Categorical']
if not cat_data.empty:
    times = [cat_data['XGB_Train_Time'].values[0],
             cat_data['LGB_Train_Time'].values[0],
             cat_data['CB_Train_Time'].values[0]]
    fastest_cat = libraries[times.index(min(times))]
    print(f"   - Dataset con categóricas: {fastest_cat} fue más rápido")
    
    # Nota especial sobre CatBoost con categóricas
    cb_time = cat_data['CB_Train_Time'].values[0]
    if cb_time > 3:
        print(f"     [NOTA]: CatBoost tardó {cb_time:.1f}s (manejo nativo de categóricas tiene overhead)")

# Dataset grande
large_data = df[df['Dataset'] == 'Binary_Large']
if not large_data.empty:
    times = [large_data['XGB_Train_Time'].values[0],
             large_data['LGB_Train_Time'].values[0],
             large_data['CB_Train_Time'].values[0]]
    fastest_large = libraries[times.index(min(times))]
    print(f"   - Dataset grande: {fastest_large} fue más rápido")

print("\n[RECOMENDACIONES]:")
print("   - Para velocidad máxima: LightGBM")
print("   - Para accuracy máxima: CatBoost o LightGBM")
print("   - Para balance velocidad/accuracy: LightGBM")
print("   - Para features categóricas nativas: CatBoost (pero con overhead)")
print("   - Para compatibilidad y estabilidad: XGBoost")

print("\n" + "="*60)