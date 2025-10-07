"""
Funciones de visualización para MLPY.

Proporciona funciones para crear gráficos comunes en machine learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, Tuple, List, Dict, Any
import warnings

# Configurar estilo por defecto
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: Optional[np.ndarray] = None,
    title: str = "Curva de Aprendizaje",
    xlabel: str = "Tamaño del conjunto de entrenamiento",
    ylabel: str = "Score",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la curva de aprendizaje mostrando el rendimiento en train y validación.
    
    Parameters
    ----------
    train_scores : np.ndarray
        Scores de entrenamiento para cada tamaño.
    val_scores : np.ndarray
        Scores de validación para cada tamaño.
    train_sizes : np.ndarray, optional
        Tamaños del conjunto de entrenamiento.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje X.
    ylabel : str
        Etiqueta del eje Y.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar la figura.
        
    Returns
    -------
    plt.Figure
        La figura matplotlib creada.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if train_sizes is None:
        train_sizes = np.arange(1, len(train_scores) + 1)
    
    # Calcular medias y desviaciones si son arrays 2D
    if train_scores.ndim > 1:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
    else:
        train_mean = train_scores
        train_std = np.zeros_like(train_scores)
        val_mean = val_scores
        val_std = np.zeros_like(val_scores)
    
    # Graficar curvas
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Score de Entrenamiento')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='steelblue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Score de Validación')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='darkorange')
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance: Union[pd.Series, Dict, np.ndarray],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Importancia de Características",
    figsize: Tuple[int, int] = (10, 8),
    orientation: str = 'horizontal',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la importancia de las características.
    
    Parameters
    ----------
    importance : pd.Series, dict, or np.ndarray
        Importancia de las características.
    feature_names : list, optional
        Nombres de las características.
    top_n : int
        Número de características top a mostrar.
    title : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.
    orientation : str
        'horizontal' o 'vertical'.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    # Convertir a Series de pandas
    if isinstance(importance, dict):
        importance = pd.Series(importance)
    elif isinstance(importance, np.ndarray):
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        importance = pd.Series(importance, index=feature_names)
    
    # Seleccionar top N
    top_features = importance.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if orientation == 'horizontal':
        positions = np.arange(len(top_features))
        ax.barh(positions, top_features.values, color='steelblue')
        ax.set_yticks(positions)
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Importancia', fontsize=12)
        ax.invert_yaxis()
    else:
        positions = np.arange(len(top_features))
        ax.bar(positions, top_features.values, color='steelblue')
        ax.set_xticks(positions)
        ax.set_xticklabels(top_features.index, rotation=45, ha='right')
        ax.set_ylabel('Importancia', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x' if orientation == 'horizontal' else 'y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    normalize: bool = False,
    title: str = "Matriz de Confusión",
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la matriz de confusión.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_pred : np.ndarray
        Predicciones.
    labels : list, optional
        Nombres de las clases.
    normalize : bool
        Si normalizar la matriz.
    title : str
        Título.
    cmap : str
        Mapa de colores.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Frecuencia' if not normalize else 'Proporción'},
                ax=ax)
    
    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Verdadero', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Análisis de Residuos",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica análisis de residuos para regresión.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valores verdaderos.
    y_pred : np.ndarray
        Predicciones.
    title : str
        Título.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Residuos vs Predicciones
    axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Valores Predichos')
    axes[0].set_ylabel('Residuos')
    axes[0].set_title('Residuos vs Predicciones')
    axes[0].grid(True, alpha=0.3)
    
    # Histograma de residuos
    axes[1].hist(residuals, bins=30, edgecolor='black', color='steelblue', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Residuos')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Residuos')
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Error de Predicción",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica los valores predichos vs reales.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valores verdaderos.
    y_pred : np.ndarray
        Predicciones.
    title : str
        Título.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Línea perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Predicción Perfecta', alpha=0.7)
    
    # Calcular métricas
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Añadir texto con métricas
    textstr = f'R² = {r2:.3f}\nMAE = {mae:.3f}\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Valores Predichos', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Curva ROC",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la curva ROC.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_proba : np.ndarray
        Probabilidades predichas.
    title : str
        Título.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Manejar multiclase
    if y_proba.ndim > 1 and y_proba.shape[1] > 2:
        # Multiclase
        from sklearn.preprocessing import label_binarize
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        
        for i, class_name in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'Clase {class_name} (AUC = {roc_auc:.3f})')
    else:
        # Binario
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Línea diagonal
    ax.plot([0, 1], [0, 1], 'r--', label='Random', alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos', fontsize=12)
    ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Curva Precision-Recall",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la curva precision-recall.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_proba : np.ndarray
        Probabilidades predichas.
    title : str
        Título.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if y_proba.ndim > 1 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    ax.plot(recall, precision, 'b-', label=f'AP = {avg_precision:.3f}', linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2, color='steelblue')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Curva de Calibración",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la curva de calibración.
    
    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas verdaderas.
    y_proba : np.ndarray
        Probabilidades predichas.
    n_bins : int
        Número de bins.
    title : str
        Título.
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if y_proba.ndim > 1 and y_proba.shape[1] == 2:
        y_proba = y_proba[:, 1]
    
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    
    ax.plot(mean_pred, fraction_pos, 's-', label='Modelo', color='steelblue', markersize=8)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfectamente calibrado', alpha=0.5)
    
    # Histograma de predicciones
    ax2 = ax.twinx()
    ax2.hist(y_proba, bins=30, alpha=0.3, color='gray', edgecolor='none')
    ax2.set_ylabel('Frecuencia', fontsize=12, color='gray')
    ax2.tick_params(axis='y', colors='gray')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Probabilidad Media Predicha', fontsize=12)
    ax.set_ylabel('Fracción de Positivos', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_validation_curve(
    param_range: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    param_name: str = "Parámetro",
    title: str = "Curva de Validación",
    xlabel: Optional[str] = None,
    ylabel: str = "Score",
    xscale: str = 'linear',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grafica la curva de validación para un hiperparámetro.
    
    Parameters
    ----------
    param_range : np.ndarray
        Rango de valores del parámetro.
    train_scores : np.ndarray
        Scores de entrenamiento.
    val_scores : np.ndarray
        Scores de validación.
    param_name : str
        Nombre del parámetro.
    title : str
        Título.
    xlabel : str, optional
        Etiqueta del eje X.
    ylabel : str
        Etiqueta del eje Y.
    xscale : str
        Escala del eje X ('linear' o 'log').
    figsize : tuple
        Tamaño de la figura.
    save_path : str, optional
        Ruta para guardar.
        
    Returns
    -------
    plt.Figure
        La figura creada.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calcular medias y desviaciones
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Graficar
    ax.plot(param_range, train_mean, 'o-', color='steelblue', 
            label='Score de Entrenamiento', linewidth=2, markersize=8)
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color='steelblue')
    
    ax.plot(param_range, val_mean, 'o-', color='darkorange',
            label='Score de Validación', linewidth=2, markersize=8)
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='darkorange')
    
    # Marcar el mejor valor
    best_idx = np.argmax(val_mean)
    ax.axvline(x=param_range[best_idx], color='red', linestyle='--', 
               alpha=0.5, label=f'Mejor: {param_range[best_idx]:.3g}')
    
    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel or param_name, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig