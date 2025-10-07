"""
Dashboards de visualización para MLPY.

Proporciona dashboards completos para análisis de modelos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, Any, List, Tuple
import warnings

from .plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_residuals,
    plot_prediction_error,
    plot_feature_importance
)


def create_model_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    feature_importance: Optional[Dict] = None,
    model_type: str = 'auto',
    title: str = "Dashboard del Modelo",
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Crea un dashboard completo para análisis del modelo.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valores verdaderos.
    y_pred : np.ndarray
        Predicciones.
    y_proba : np.ndarray, optional
        Probabilidades predichas (para clasificación).
    feature_importance : dict, optional
        Importancia de características.
    model_type : str
        'classification', 'regression' o 'auto'.
    title : str
        Título del dashboard.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura del dashboard.
    """
    # Detectar tipo de modelo
    if model_type == 'auto':
        unique_vals = np.unique(y_true)
        if len(unique_vals) <= 20 and np.all(unique_vals == unique_vals.astype(int)):
            model_type = 'classification'
        else:
            model_type = 'regression'
    
    # Crear figura con subplots
    fig = plt.figure(figsize=figsize)
    
    if model_type == 'classification':
        # Dashboard de clasificación
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Matriz de confusión
        ax1 = fig.add_subplot(gs[0, 0])
        cm_fig = plot_confusion_matrix(y_true, y_pred, normalize=False, title='')
        plt.close(cm_fig)
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Verdadero')
        
        # Matriz normalizada
        ax2 = fig.add_subplot(gs[0, 1])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
        ax2.set_title('Matriz Normalizada', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicción')
        ax2.set_ylabel('Verdadero')
        
        # Métricas
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Calcular métricas
        accuracy = accuracy_score(y_true, y_pred)
        
        # Manejar multiclase
        avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
        
        metrics_text = f"""Métricas del Modelo
        
Accuracy:   {accuracy:.3f}
Precision:  {precision:.3f}
Recall:     {recall:.3f}
F1-Score:   {f1:.3f}

Muestras:   {len(y_true)}
Clases:     {len(np.unique(y_true))}"""
        
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax3.axis('off')
        ax3.set_title('Métricas', fontsize=12, fontweight='bold')
        
        # Curvas ROC y PR si hay probabilidades
        if y_proba is not None:
            # ROC
            ax4 = fig.add_subplot(gs[1, :2])
            from sklearn.metrics import roc_curve, auc
            
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            roc_auc = auc(fpr, tpr)
            
            ax4.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
            ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            ax4.set_xlabel('Tasa de Falsos Positivos')
            ax4.set_ylabel('Tasa de Verdaderos Positivos')
            ax4.set_title('Curva ROC', fontsize=12, fontweight='bold')
            ax4.legend(loc='lower right')
            ax4.grid(True, alpha=0.3)
            
            # Precision-Recall
            ax5 = fig.add_subplot(gs[1, 2])
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba_pos)
            avg_precision = average_precision_score(y_true, y_proba_pos)
            
            ax5.plot(recall_curve, precision_curve, 'b-', 
                    label=f'AP = {avg_precision:.3f}', linewidth=2)
            ax5.fill_between(recall_curve, precision_curve, alpha=0.2, color='steelblue')
            ax5.set_xlabel('Recall')
            ax5.set_ylabel('Precision')
            ax5.set_title('Curva Precision-Recall', fontsize=12, fontweight='bold')
            ax5.legend(loc='lower left')
            ax5.grid(True, alpha=0.3)
        
        # Importancia de características
        if feature_importance is not None:
            ax6 = fig.add_subplot(gs[2, :])
            importance_series = pd.Series(feature_importance)
            top_features = importance_series.nlargest(15)
            
            positions = np.arange(len(top_features))
            ax6.barh(positions, top_features.values, color='steelblue')
            ax6.set_yticks(positions)
            ax6.set_yticklabels(top_features.index, fontsize=9)
            ax6.set_xlabel('Importancia')
            ax6.set_title('Top 15 Características', fontsize=12, fontweight='bold')
            ax6.invert_yaxis()
            ax6.grid(True, alpha=0.3, axis='x')
            
    else:
        # Dashboard de regresión
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Error de predicción
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.6, color='steelblue', s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        ax1.set_xlabel('Valores Reales')
        ax1.set_ylabel('Valores Predichos')
        ax1.set_title('Predicción vs Real', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residuos
        ax2 = fig.add_subplot(gs[0, 1])
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='steelblue', s=20)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Valores Predichos')
        ax2.set_ylabel('Residuos')
        ax2.set_title('Residuos vs Predicciones', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Métricas
        ax3 = fig.add_subplot(gs[0, 2])
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Calcular MAPE si es posible
        if np.all(y_true != 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            mape_text = f"MAPE:      {mape:.2f}%"
        else:
            mape_text = "MAPE:      N/A"
        
        metrics_text = f"""Métricas del Modelo
        
R²:        {r2:.3f}
MAE:       {mae:.3f}
MSE:       {mse:.3f}
RMSE:      {rmse:.3f}
{mape_text}

Muestras:  {len(y_true)}"""
        
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax3.axis('off')
        ax3.set_title('Métricas', fontsize=12, fontweight='bold')
        
        # Distribución de residuos
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(residuals, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Residuos')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Q-Q plot
        ax5 = fig.add_subplot(gs[1, 1])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Error por percentil
        ax6 = fig.add_subplot(gs[1, 2])
        percentiles = np.percentile(y_true, np.arange(0, 101, 10))
        errors_by_percentile = []
        
        for i in range(len(percentiles) - 1):
            mask = (y_true >= percentiles[i]) & (y_true < percentiles[i+1])
            if np.any(mask):
                errors_by_percentile.append(np.mean(np.abs(residuals[mask])))
            else:
                errors_by_percentile.append(0)
        
        ax6.bar(range(len(errors_by_percentile)), errors_by_percentile, 
                color='steelblue', alpha=0.7)
        ax6.set_xlabel('Percentil')
        ax6.set_ylabel('Error Absoluto Medio')
        ax6.set_title('Error por Percentil', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(errors_by_percentile)))
        ax6.set_xticklabels([f'{i*10}-{(i+1)*10}' for i in range(len(errors_by_percentile))],
                           rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Importancia de características
        if feature_importance is not None:
            ax7 = fig.add_subplot(gs[2, :])
            importance_series = pd.Series(feature_importance)
            top_features = importance_series.nlargest(15)
            
            positions = np.arange(len(top_features))
            ax7.barh(positions, top_features.values, color='steelblue')
            ax7.set_yticks(positions)
            ax7.set_yticklabels(top_features.index, fontsize=9)
            ax7.set_xlabel('Importancia')
            ax7.set_title('Top 15 Características', fontsize=12, fontweight='bold')
            ax7.invert_yaxis()
            ax7.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comparison_dashboard(
    models_results: Dict[str, Dict],
    title: str = "Comparación de Modelos",
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Crea un dashboard para comparar múltiples modelos.
    
    Parameters
    ----------
    models_results : dict
        Diccionario con resultados de modelos.
        Formato: {
            'model_name': {
                'y_true': array,
                'y_pred': array,
                'y_proba': array (opcional),
                'train_time': float (opcional),
                'predict_time': float (opcional)
            }
        }
    title : str
        Título del dashboard.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura del dashboard.
    """
    n_models = len(models_results)
    
    # Detectar tipo de problema
    first_model = list(models_results.values())[0]
    y_true = first_model['y_true']
    unique_vals = np.unique(y_true)
    
    if len(unique_vals) <= 20 and np.all(unique_vals == unique_vals.astype(int)):
        model_type = 'classification'
    else:
        model_type = 'regression'
    
    # Crear figura
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Recopilar métricas
    metrics_df = []
    
    for model_name, results in models_results.items():
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        metrics_row = {'Modelo': model_name}
        
        if model_type == 'classification':
            from sklearn.metrics import (accuracy_score, precision_score, 
                                       recall_score, f1_score)
            
            avg_method = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            
            metrics_row['Accuracy'] = accuracy_score(y_true, y_pred)
            metrics_row['Precision'] = precision_score(y_true, y_pred, 
                                                     average=avg_method, zero_division=0)
            metrics_row['Recall'] = recall_score(y_true, y_pred, 
                                               average=avg_method, zero_division=0)
            metrics_row['F1-Score'] = f1_score(y_true, y_pred, 
                                             average=avg_method, zero_division=0)
            
            if 'y_proba' in results and results['y_proba'] is not None:
                from sklearn.metrics import roc_auc_score
                y_proba = results['y_proba']
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                    y_proba = y_proba[:, 1]
                try:
                    metrics_row['AUC'] = roc_auc_score(y_true, y_proba)
                except:
                    metrics_row['AUC'] = np.nan
                    
        else:
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            
            metrics_row['R²'] = r2_score(y_true, y_pred)
            metrics_row['MAE'] = mean_absolute_error(y_true, y_pred)
            metrics_row['MSE'] = mean_squared_error(y_true, y_pred)
            metrics_row['RMSE'] = np.sqrt(metrics_row['MSE'])
        
        if 'train_time' in results:
            metrics_row['Tiempo Entrenamiento (s)'] = results['train_time']
        if 'predict_time' in results:
            metrics_row['Tiempo Predicción (s)'] = results['predict_time']
            
        metrics_df.append(metrics_row)
    
    metrics_df = pd.DataFrame(metrics_df)
    
    # Tabla de métricas
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('tight')
    ax1.axis('off')
    
    # Formatear valores numéricos
    formatted_data = []
    for _, row in metrics_df.iterrows():
        formatted_row = [row['Modelo']]
        for col in metrics_df.columns[1:]:
            if col in row:
                val = row[col]
                if pd.notna(val):
                    if 'Tiempo' in col:
                        formatted_row.append(f'{val:.3f}')
                    else:
                        formatted_row.append(f'{val:.3f}')
                else:
                    formatted_row.append('N/A')
            else:
                formatted_row.append('N/A')
        formatted_data.append(formatted_row)
    
    table = ax1.table(cellText=formatted_data,
                     colLabels=metrics_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] + [0.1] * (len(metrics_df.columns) - 1))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colorear mejor modelo
    if model_type == 'classification':
        best_col = metrics_df['Accuracy'].idxmax() + 1 if 'Accuracy' in metrics_df else 0
    else:
        best_col = metrics_df['R²'].idxmax() + 1 if 'R²' in metrics_df else 0
    
    if best_col > 0:
        for i in range(len(metrics_df.columns)):
            table[(best_col, i)].set_facecolor('#90EE90')
    
    ax1.set_title('Tabla Comparativa de Métricas', fontsize=14, fontweight='bold', pad=20)
    
    # Gráficos de barras para métricas principales
    if model_type == 'classification':
        # Accuracy
        ax2 = fig.add_subplot(gs[1, 0])
        if 'Accuracy' in metrics_df:
            ax2.bar(metrics_df['Modelo'], metrics_df['Accuracy'], color='steelblue')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Comparación de Accuracy', fontsize=12)
            ax2.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # F1-Score
        ax3 = fig.add_subplot(gs[1, 1])
        if 'F1-Score' in metrics_df:
            ax3.bar(metrics_df['Modelo'], metrics_df['F1-Score'], color='darkorange')
            ax3.set_ylabel('F1-Score')
            ax3.set_title('Comparación de F1-Score', fontsize=12)
            ax3.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # AUC si está disponible
        ax4 = fig.add_subplot(gs[1, 2])
        if 'AUC' in metrics_df:
            valid_auc = metrics_df[metrics_df['AUC'].notna()]
            if not valid_auc.empty:
                ax4.bar(valid_auc['Modelo'], valid_auc['AUC'], color='green')
                ax4.set_ylabel('AUC')
                ax4.set_title('Comparación de AUC', fontsize=12)
                ax4.set_xticklabels(valid_auc['Modelo'], rotation=45, ha='right')
                ax4.grid(True, alpha=0.3, axis='y')
        
    else:
        # R²
        ax2 = fig.add_subplot(gs[1, 0])
        if 'R²' in metrics_df:
            ax2.bar(metrics_df['Modelo'], metrics_df['R²'], color='steelblue')
            ax2.set_ylabel('R²')
            ax2.set_title('Comparación de R²', fontsize=12)
            ax2.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # RMSE
        ax3 = fig.add_subplot(gs[1, 1])
        if 'RMSE' in metrics_df:
            ax3.bar(metrics_df['Modelo'], metrics_df['RMSE'], color='darkorange')
            ax3.set_ylabel('RMSE')
            ax3.set_title('Comparación de RMSE', fontsize=12)
            ax3.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # MAE
        ax4 = fig.add_subplot(gs[1, 2])
        if 'MAE' in metrics_df:
            ax4.bar(metrics_df['Modelo'], metrics_df['MAE'], color='green')
            ax4.set_ylabel('MAE')
            ax4.set_title('Comparación de MAE', fontsize=12)
            ax4.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
            ax4.grid(True, alpha=0.3, axis='y')
    
    # Tiempos si están disponibles
    if 'Tiempo Entrenamiento (s)' in metrics_df or 'Tiempo Predicción (s)' in metrics_df:
        ax5 = fig.add_subplot(gs[2, :2])
        
        width = 0.35
        x = np.arange(len(metrics_df))
        
        if 'Tiempo Entrenamiento (s)' in metrics_df:
            ax5.bar(x - width/2, metrics_df['Tiempo Entrenamiento (s)'], 
                   width, label='Entrenamiento', color='steelblue')
        
        if 'Tiempo Predicción (s)' in metrics_df:
            ax5.bar(x + width/2, metrics_df['Tiempo Predicción (s)'], 
                   width, label='Predicción', color='darkorange')
        
        ax5.set_ylabel('Tiempo (segundos)')
        ax5.set_title('Comparación de Tiempos', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_df['Modelo'], rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Gráfico de radar para métricas normalizadas
    ax6 = fig.add_subplot(gs[2, 2], projection='polar')
    
    # Seleccionar métricas para el radar
    if model_type == 'classification':
        radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        # Invertir MSE y RMSE para que mayor sea mejor
        radar_metrics = ['R²']
        if 'MAE' in metrics_df:
            metrics_df['1-MAE_norm'] = 1 - (metrics_df['MAE'] / metrics_df['MAE'].max())
            radar_metrics.append('1-MAE_norm')
        if 'RMSE' in metrics_df:
            metrics_df['1-RMSE_norm'] = 1 - (metrics_df['RMSE'] / metrics_df['RMSE'].max())
            radar_metrics.append('1-RMSE_norm')
    
    radar_metrics = [m for m in radar_metrics if m in metrics_df.columns]
    
    if radar_metrics:
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for _, row in metrics_df.iterrows():
            values = [row[m] if pd.notna(row[m]) else 0 for m in radar_metrics]
            values += values[:1]
            ax6.plot(angles, values, 'o-', linewidth=2, label=row['Modelo'])
            ax6.fill(angles, values, alpha=0.25)
        
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels([m.replace('1-', '').replace('_norm', '') for m in radar_metrics])
        ax6.set_ylim(0, 1)
        ax6.set_title('Comparación Multimétrica', fontsize=12, fontweight='bold', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig