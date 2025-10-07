"""
Dashboard interactivo para MLPY.

Visualización en tiempo real del entrenamiento,
métricas y análisis de modelos.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import time
from datetime import datetime
from pathlib import Path
import json
import webbrowser
import threading
from queue import Queue

# Imports opcionales para visualización
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


@dataclass
class TrainingMetrics:
    """Métricas durante el entrenamiento."""
    epoch: int
    timestamp: float
    train_loss: float
    val_loss: Optional[float] = None
    train_metric: Optional[float] = None
    val_metric: Optional[float] = None
    learning_rate: Optional[float] = None
    duration: Optional[float] = None
    

class MLPYDashboard:
    """
    Dashboard interactivo para monitoreo de ML en MLPY.
    
    Características:
    - Visualización en tiempo real
    - Métricas de entrenamiento
    - Comparación de modelos
    - Análisis de features
    - Exportación de reportes
    """
    
    def __init__(self, 
                 title: str = "MLPY Dashboard",
                 update_interval: float = 1.0,
                 port: int = 8050,
                 auto_open: bool = True):
        """
        Inicializa el dashboard.
        
        Parameters
        ----------
        title : str
            Título del dashboard
        update_interval : float
            Intervalo de actualización en segundos
        port : int
            Puerto para el servidor web
        auto_open : bool
            Abrir automáticamente en el navegador
        """
        self.title = title
        self.update_interval = update_interval
        self.port = port
        self.auto_open = auto_open
        
        # Almacenamiento de métricas
        self.metrics_history = []
        self.models_comparison = {}
        self.feature_importance = {}
        
        # Cola para actualizaciones en tiempo real
        self.update_queue = Queue()
        
        # Estado del dashboard
        self.is_running = False
        self.start_time = None
        
    def start(self):
        """Inicia el dashboard."""
        if not HAS_PLOTLY:
            print("Dashboard requiere plotly. Instalar con: pip install plotly")
            return self._start_simple_dashboard()
        
        self.is_running = True
        self.start_time = time.time()
        
        # Crear aplicación Dash (si está disponible)
        try:
            import dash
            from dash import dcc, html, Input, Output
            import dash_bootstrap_components as dbc
            
            self._create_dash_app()
            
        except ImportError:
            print("Para dashboard completo instalar: pip install dash dash-bootstrap-components")
            self._create_static_dashboard()
    
    def _create_static_dashboard(self):
        """Crea dashboard estático con plotly."""
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Training Progress',
                'Model Comparison',
                'Feature Importance',
                'Confusion Matrix'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'heatmap'}]
            ]
        )
        
        # 1. Training Progress
        if self.metrics_history:
            df_metrics = pd.DataFrame(self.metrics_history)
            fig.add_trace(
                go.Scatter(
                    x=df_metrics['epoch'],
                    y=df_metrics['train_loss'],
                    name='Train Loss',
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            if 'val_loss' in df_metrics.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_metrics['epoch'],
                        y=df_metrics['val_loss'],
                        name='Val Loss',
                        mode='lines+markers'
                    ),
                    row=1, col=1
                )
        
        # 2. Model Comparison
        if self.models_comparison:
            models = list(self.models_comparison.keys())
            scores = [self.models_comparison[m].get('score', 0) for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=scores, name='Model Scores'),
                row=1, col=2
            )
        
        # 3. Feature Importance
        if self.feature_importance:
            features = list(self.feature_importance.keys())[:10]
            importance = [self.feature_importance[f] for f in features]
            
            fig.add_trace(
                go.Bar(x=importance, y=features, orientation='h', name='Importance'),
                row=2, col=1
            )
        
        # 4. Confusion Matrix (ejemplo)
        confusion_matrix = np.random.rand(3, 3)
        fig.add_trace(
            go.Heatmap(
                z=confusion_matrix,
                colorscale='Blues',
                showscale=True
            ),
            row=2, col=2
        )
        
        # Actualizar layout
        fig.update_layout(
            title_text=self.title,
            height=800,
            showlegend=True
        )
        
        # Guardar y abrir
        output_path = f"mlpy_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(output_path)
        
        if self.auto_open:
            webbrowser.open(f"file://{Path(output_path).absolute()}")
        
        print(f"Dashboard guardado en: {output_path}")
        return output_path
    
    def _start_simple_dashboard(self):
        """Dashboard simple con matplotlib."""
        if not HAS_MATPLOTLIB:
            print("Visualización requiere matplotlib. Instalar con: pip install matplotlib seaborn")
            return self._create_text_dashboard()
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(self.title, fontsize=16)
        
        # 1. Training Progress
        ax1 = axes[0, 0]
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
            if 'val_loss' in df.columns:
                ax1.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Model Comparison
        ax2 = axes[0, 1]
        if self.models_comparison:
            models = list(self.models_comparison.keys())
            scores = [self.models_comparison[m].get('score', 0) for m in models]
            ax2.bar(models, scores)
            ax2.set_title('Model Comparison')
            ax2.set_ylabel('Score')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Feature Importance
        ax3 = axes[1, 0]
        if self.feature_importance:
            features = list(self.feature_importance.keys())[:10]
            importance = [self.feature_importance[f] for f in features]
            ax3.barh(features, importance)
            ax3.set_title('Top 10 Features')
            ax3.set_xlabel('Importance')
        
        # 4. Metrics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary_text = self._generate_summary()
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Guardar y mostrar
        output_path = f"mlpy_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.show()
        
        print(f"Dashboard guardado en: {output_path}")
        return output_path
    
    def _create_text_dashboard(self):
        """Dashboard de texto cuando no hay librerías de visualización."""
        
        output = []
        output.append("=" * 60)
        output.append(f" {self.title} ")
        output.append("=" * 60)
        
        # Training Progress
        output.append("\nTRAINING PROGRESS:")
        output.append("-" * 40)
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            output.append(df.to_string())
        else:
            output.append("No training data available")
        
        # Model Comparison
        output.append("\nMODEL COMPARISON:")
        output.append("-" * 40)
        if self.models_comparison:
            for model, metrics in self.models_comparison.items():
                output.append(f"{model}: {metrics}")
        else:
            output.append("No models to compare")
        
        # Feature Importance
        output.append("\nFEATURE IMPORTANCE:")
        output.append("-" * 40)
        if self.feature_importance:
            sorted_features = sorted(
                self.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            for feat, imp in sorted_features:
                output.append(f"{feat:20s}: {'#' * int(imp * 50)}")
        else:
            output.append("No feature importance data")
        
        # Summary
        output.append("\nSUMMARY:")
        output.append("-" * 40)
        output.append(self._generate_summary())
        
        dashboard_text = "\n".join(output)
        
        # Guardar en archivo
        output_path = f"mlpy_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_path, 'w') as f:
            f.write(dashboard_text)
        
        print(dashboard_text)
        print(f"\nDashboard guardado en: {output_path}")
        
        return output_path
    
    def log_metrics(self, metrics: Union[TrainingMetrics, Dict]):
        """Registra métricas de entrenamiento."""
        if isinstance(metrics, dict):
            metrics = TrainingMetrics(**metrics)
        
        self.metrics_history.append(metrics.__dict__)
        
        if self.is_running:
            self.update_queue.put(('metrics', metrics))
    
    def log_model(self, model_name: str, metrics: Dict[str, float]):
        """Registra un modelo para comparación."""
        self.models_comparison[model_name] = metrics
        
        if self.is_running:
            self.update_queue.put(('model', (model_name, metrics)))
    
    def log_feature_importance(self, importance: Dict[str, float]):
        """Registra importancia de features."""
        self.feature_importance = importance
        
        if self.is_running:
            self.update_queue.put(('features', importance))
    
    def _generate_summary(self) -> str:
        """Genera resumen de métricas."""
        lines = []
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            lines.append(f"Runtime: {elapsed:.2f}s")
        
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            if 'train_loss' in df.columns:
                lines.append(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
            if 'val_loss' in df.columns:
                lines.append(f"Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")
                lines.append(f"Best Val Loss: {df['val_loss'].min():.4f}")
        
        if self.models_comparison:
            best_model = max(
                self.models_comparison.items(),
                key=lambda x: x[1].get('score', 0)
            )
            lines.append(f"Best Model: {best_model[0]}")
            lines.append(f"Best Score: {best_model[1].get('score', 0):.4f}")
        
        lines.append(f"Total Models: {len(self.models_comparison)}")
        lines.append(f"Total Epochs: {len(self.metrics_history)}")
        
        return "\n".join(lines)
    
    def export_report(self, output_path: Optional[str] = None) -> str:
        """Exporta reporte completo."""
        if output_path is None:
            output_path = f"mlpy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'title': self.title,
            'timestamp': datetime.now().isoformat(),
            'metrics_history': self.metrics_history,
            'models_comparison': self.models_comparison,
            'feature_importance': self.feature_importance,
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Reporte exportado a: {output_path}")
        return output_path
    
    def stop(self):
        """Detiene el dashboard."""
        self.is_running = False
        print("Dashboard detenido")


# Función de conveniencia para crear dashboard rápidamente
def create_dashboard(title="MLPY Training Dashboard", **kwargs):
    """
    Crea un dashboard para monitoreo de entrenamiento.
    
    Parameters
    ----------
    title : str
        Título del dashboard
    **kwargs
        Argumentos adicionales para MLPYDashboard
    
    Returns
    -------
    MLPYDashboard
        Instancia del dashboard
    """
    dashboard = MLPYDashboard(title=title, **kwargs)
    return dashboard