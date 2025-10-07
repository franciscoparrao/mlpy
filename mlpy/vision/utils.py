"""
Utilidades para Computer Vision.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def visualize_predictions(
    images: List[np.ndarray],
    predictions: List[Any],
    ground_truths: Optional[List[Any]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None
):
    """Visualiza predicciones de modelos.
    
    Parameters
    ----------
    images : List[np.ndarray]
        Lista de imágenes.
    predictions : List[Any]
        Predicciones del modelo.
    ground_truths : Optional[List[Any]]
        Valores verdaderos.
    class_names : Optional[List[str]]
        Nombres de clases.
    figsize : Tuple[int, int]
        Tamaño de la figura.
    save_path : Optional[Union[str, Path]]
        Ruta para guardar.
    """
    try:
        import matplotlib.pyplot as plt
        
        n_images = len(images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_images == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        idx = 0
        for row in axes:
            for ax in row:
                if idx < n_images:
                    # Mostrar imagen
                    ax.imshow(images[idx])
                    
                    # Título con predicción
                    title = ""
                    if predictions and idx < len(predictions):
                        pred = predictions[idx]
                        if class_names and isinstance(pred, int):
                            title = f"Pred: {class_names[pred]}"
                        else:
                            title = f"Pred: {pred}"
                    
                    if ground_truths and idx < len(ground_truths):
                        gt = ground_truths[idx]
                        if class_names and isinstance(gt, int):
                            title += f"\nTrue: {class_names[gt]}"
                        else:
                            title += f"\nTrue: {gt}"
                    
                    ax.set_title(title, fontsize=10)
                    ax.axis('off')
                else:
                    ax.axis('off')
                
                idx += 1
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.error("matplotlib required for visualization")


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """Dibuja cajas delimitadoras en imagen.
    
    Parameters
    ----------
    image : np.ndarray
        Imagen base.
    boxes : np.ndarray
        Cajas en formato [x1, y1, x2, y2].
    labels : Optional[np.ndarray]
        Etiquetas de las cajas.
    scores : Optional[np.ndarray]
        Scores de confianza.
    class_names : Optional[List[str]]
        Nombres de clases.
    colors : Optional[List[Tuple[int, int, int]]]
        Colores para cada clase.
    thickness : int
        Grosor de las líneas.
    font_scale : float
        Escala de la fuente.
        
    Returns
    -------
    np.ndarray
        Imagen con cajas dibujadas.
    """
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python required for drawing boxes")
        return image
    
    # Copiar imagen
    result = image.copy()
    
    # Colores predeterminados
    if colors is None:
        colors = generate_colors(max(labels) + 1 if labels is not None else len(boxes))
    
    # Dibujar cada caja
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        
        # Color de la caja
        if labels is not None and i < len(labels):
            color = colors[labels[i] % len(colors)]
        else:
            color = colors[i % len(colors)]
        
        # Dibujar rectángulo
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # Preparar texto
        text_parts = []
        if labels is not None and i < len(labels):
            label = labels[i]
            if class_names and label < len(class_names):
                text_parts.append(class_names[label])
            else:
                text_parts.append(f"Class {label}")
        
        if scores is not None and i < len(scores):
            text_parts.append(f"{scores[i]:.2f}")
        
        if text_parts:
            text = " ".join(text_parts)
            
            # Calcular tamaño del texto
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Fondo para el texto
            cv2.rectangle(
                result,
                (x1, y1 - text_height - 4),
                (x1 + text_width + 4, y1),
                color,
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                result,
                text,
                (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness - 1
            )
    
    return result


def draw_segmentation_masks(
    image: np.ndarray,
    mask: np.ndarray,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """Dibuja máscaras de segmentación en imagen.
    
    Parameters
    ----------
    image : np.ndarray
        Imagen base.
    mask : np.ndarray
        Máscara de segmentación.
    class_names : Optional[List[str]]
        Nombres de clases.
    colors : Optional[List[Tuple[int, int, int]]]
        Colores para cada clase.
    alpha : float
        Transparencia de la máscara.
        
    Returns
    -------
    np.ndarray
        Imagen con máscara superpuesta.
    """
    # Copiar imagen
    result = image.copy().astype(np.float32)
    
    # Obtener clases únicas
    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 0]  # Ignorar fondo
    
    # Colores predeterminados
    if colors is None:
        colors = generate_colors(max(unique_classes) + 1)
    
    # Crear overlay
    overlay = np.zeros_like(result)
    
    for cls in unique_classes:
        # Máscara para esta clase
        class_mask = (mask == cls)
        
        # Color para esta clase
        color = colors[cls % len(colors)]
        
        # Aplicar color
        overlay[class_mask] = color
    
    # Combinar con imagen original
    mask_indices = mask > 0
    result[mask_indices] = (1 - alpha) * result[mask_indices] + alpha * overlay[mask_indices]
    
    return result.astype(np.uint8)


def plot_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    n_cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
):
    """Muestra grid de imágenes.
    
    Parameters
    ----------
    images : List[np.ndarray]
        Lista de imágenes.
    titles : Optional[List[str]]
        Títulos para cada imagen.
    n_cols : int
        Número de columnas.
    figsize : Optional[Tuple[int, int]]
        Tamaño de la figura.
    save_path : Optional[Union[str, Path]]
        Ruta para guardar.
    """
    try:
        import matplotlib.pyplot as plt
        
        n_images = len(images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 3, n_rows * 3)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Asegurar que axes sea 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (row_axes) in enumerate(axes):
            for col_idx, ax in enumerate(row_axes if n_cols > 1 else [row_axes]):
                img_idx = idx * n_cols + col_idx
                
                if img_idx < n_images:
                    ax.imshow(images[img_idx])
                    if titles and img_idx < len(titles):
                        ax.set_title(titles[img_idx], fontsize=10)
                    ax.axis('off')
                else:
                    ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grid saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        logger.error("matplotlib required for plotting")


def save_predictions(
    predictions: Dict[str, Any],
    save_path: Union[str, Path],
    format: str = 'json'
):
    """Guarda predicciones en archivo.
    
    Parameters
    ----------
    predictions : Dict[str, Any]
        Predicciones a guardar.
    save_path : Union[str, Path]
        Ruta de guardado.
    format : str
        Formato ('json', 'csv', 'pickle').
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        import json
        
        # Convertir arrays numpy a listas
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        predictions_json = convert_numpy(predictions)
        
        with open(save_path, 'w') as f:
            json.dump(predictions_json, f, indent=2)
    
    elif format == 'csv':
        import pandas as pd
        
        # Convertir a DataFrame
        df = pd.DataFrame(predictions)
        df.to_csv(save_path, index=False)
    
    elif format == 'pickle':
        import pickle
        
        with open(save_path, 'wb') as f:
            pickle.dump(predictions, f)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Predictions saved to {save_path}")


def calculate_iou(
    box1: np.ndarray,
    box2: np.ndarray
) -> float:
    """Calcula Intersection over Union entre dos cajas.
    
    Parameters
    ----------
    box1 : np.ndarray
        Primera caja [x1, y1, x2, y2].
    box2 : np.ndarray
        Segunda caja [x1, y1, x2, y2].
        
    Returns
    -------
    float
        Valor IoU.
    """
    # Calcular intersección
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calcular áreas
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calcular unión
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


def calculate_map(
    predictions: List[Dict[str, np.ndarray]],
    ground_truths: List[Dict[str, np.ndarray]],
    iou_thresholds: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """Calcula mean Average Precision.
    
    Parameters
    ----------
    predictions : List[Dict[str, np.ndarray]]
        Predicciones con 'boxes', 'labels' y 'scores'.
    ground_truths : List[Dict[str, np.ndarray]]
        Ground truth con 'boxes' y 'labels'.
    iou_thresholds : Optional[List[float]]
        Umbrales IoU para evaluación.
    class_names : Optional[List[str]]
        Nombres de clases.
        
    Returns
    -------
    Dict[str, float]
        mAP y métricas relacionadas.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    # Implementación simplificada
    # Una implementación completa requeriría el cálculo de precisión-recall
    results = {}
    
    for threshold in iou_thresholds:
        results[f'map_{int(threshold*100)}'] = 0.0
    
    results['map'] = 0.0
    
    logger.warning("Full mAP calculation not implemented yet")
    return results


def generate_colors(n_colors: int) -> List[Tuple[int, int, int]]:
    """Genera colores distintivos.
    
    Parameters
    ----------
    n_colors : int
        Número de colores a generar.
        
    Returns
    -------
    List[Tuple[int, int, int]]
        Lista de colores RGB.
    """
    colors = []
    
    # Colores base distintivos
    base_colors = [
        (255, 0, 0),     # Rojo
        (0, 255, 0),     # Verde
        (0, 0, 255),     # Azul
        (255, 255, 0),   # Amarillo
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Cian
        (255, 128, 0),   # Naranja
        (128, 0, 255),   # Violeta
        (0, 128, 255),   # Azul claro
        (255, 0, 128),   # Rosa
    ]
    
    # Usar colores base primero
    colors.extend(base_colors[:min(n_colors, len(base_colors))])
    
    # Generar más colores si es necesario
    if n_colors > len(base_colors):
        import colorsys
        
        for i in range(n_colors - len(base_colors)):
            hue = i / (n_colors - len(base_colors))
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb))
    
    return colors[:n_colors]


def augment_image(
    image: np.ndarray,
    augmentation_params: Dict[str, Any]
) -> np.ndarray:
    """Aplica aumentación a imagen.
    
    Parameters
    ----------
    image : np.ndarray
        Imagen a aumentar.
    augmentation_params : Dict[str, Any]
        Parámetros de aumentación.
        
    Returns
    -------
    np.ndarray
        Imagen aumentada.
    """
    result = image.copy()
    
    # Aplicar aumentaciones según parámetros
    if 'brightness' in augmentation_params:
        factor = augmentation_params['brightness']
        result = np.clip(result * factor, 0, 255).astype(np.uint8)
    
    if 'contrast' in augmentation_params:
        factor = augmentation_params['contrast']
        mean = result.mean()
        result = np.clip((result - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    if 'rotation' in augmentation_params:
        try:
            import cv2
            angle = augmentation_params['rotation']
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            result = cv2.warpAffine(result, matrix, (w, h))
        except ImportError:
            logger.warning("opencv-python required for rotation")
    
    if 'flip_horizontal' in augmentation_params and augmentation_params['flip_horizontal']:
        result = np.fliplr(result)
    
    if 'flip_vertical' in augmentation_params and augmentation_params['flip_vertical']:
        result = np.flipud(result)
    
    if 'noise' in augmentation_params:
        noise_level = augmentation_params['noise']
        noise = np.random.randn(*result.shape) * noise_level
        result = np.clip(result + noise, 0, 255).astype(np.uint8)
    
    return result