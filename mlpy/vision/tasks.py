"""
Tareas de Computer Vision para MLPY.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from ..tasks.base import Task
from ..data.data import Data
from ..measures.base import Measure

logger = logging.getLogger(__name__)


class TaskVision(Task):
    """Tarea base de visión.
    
    Parameters
    ----------
    id : str
        Identificador de la tarea.
    dataset : Any
        Dataset de imágenes.
    model : Any
        Modelo de visión.
    transform : Optional[Any]
        Transformaciones a aplicar.
    metrics : Optional[List[str]]
        Métricas a evaluar.
    """
    
    def __init__(
        self,
        id: str,
        dataset: Any = None,
        model: Any = None,
        transform: Optional[Any] = None,
        metrics: Optional[List[str]] = None
    ):
        super().__init__(id=id)
        self.dataset = dataset
        self.model = model
        self.transform = transform
        self.metrics = metrics or []
        self.results = {}
    
    def get_data(self) -> Data:
        """Obtiene datos de la tarea."""
        if self.dataset is None:
            raise ValueError("Dataset not set")
        
        # Convertir dataset a formato MLPY Data
        # Por ahora retornamos un placeholder
        return Data(id="vision_data")
    
    def evaluate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evalúa predicciones.
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicciones del modelo.
        targets : np.ndarray
            Valores verdaderos.
        metrics : Optional[List[str]]
            Métricas a calcular.
            
        Returns
        -------
        Dict[str, float]
            Resultados de evaluación.
        """
        metrics = metrics or self.metrics
        results = {}
        
        for metric in metrics:
            if metric == 'accuracy':
                results['accuracy'] = np.mean(predictions == targets)
            elif metric == 'top5_accuracy':
                # Para clasificación multi-clase
                if len(predictions.shape) > 1:
                    top5 = np.argsort(predictions, axis=1)[:, -5:]
                    correct = np.any(top5 == targets[:, None], axis=1)
                    results['top5_accuracy'] = np.mean(correct)
            elif metric == 'iou':
                # Para segmentación
                intersection = np.logical_and(predictions, targets).sum()
                union = np.logical_or(predictions, targets).sum()
                results['iou'] = intersection / (union + 1e-6)
            elif metric == 'map':
                # Para detección de objetos
                # Placeholder - implementación completa requiere más lógica
                results['map'] = 0.0
        
        self.results = results
        return results
    
    def visualize(
        self,
        images: List[np.ndarray],
        predictions: Optional[Any] = None,
        save_path: Optional[Path] = None
    ):
        """Visualiza imágenes y predicciones.
        
        Parameters
        ----------
        images : List[np.ndarray]
            Imágenes a visualizar.
        predictions : Optional[Any]
            Predicciones a mostrar.
        save_path : Optional[Path]
            Ruta para guardar visualización.
        """
        try:
            import matplotlib.pyplot as plt
            
            n_images = len(images)
            fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
            
            if n_images == 1:
                axes = [axes]
            
            for i, (img, ax) in enumerate(zip(images, axes)):
                ax.imshow(img)
                ax.axis('off')
                
                if predictions is not None and i < len(predictions):
                    ax.set_title(f"Pred: {predictions[i]}")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib required for visualization")


class TaskImageClassification(TaskVision):
    """Tarea de clasificación de imágenes.
    
    Parameters
    ----------
    id : str
        Identificador.
    dataset : Any
        Dataset de imágenes.
    model : Any
        Modelo clasificador.
    num_classes : int
        Número de clases.
    class_names : Optional[List[str]]
        Nombres de las clases.
    """
    
    def __init__(
        self,
        id: str,
        dataset: Any = None,
        model: Any = None,
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        transform: Optional[Any] = None
    ):
        super().__init__(
            id=id,
            dataset=dataset,
            model=model,
            transform=transform,
            metrics=['accuracy', 'top5_accuracy']
        )
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.confusion_matrix = None
    
    def train(
        self,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        device: str = 'auto'
    ) -> Dict[str, List[float]]:
        """Entrena el modelo de clasificación.
        
        Parameters
        ----------
        epochs : int
            Número de épocas.
        batch_size : int
            Tamaño del batch.
        learning_rate : float
            Tasa de aprendizaje.
        optimizer : str
            Optimizador a usar.
        device : str
            Dispositivo de entrenamiento.
            
        Returns
        -------
        Dict[str, List[float]]
            Historial de entrenamiento.
        """
        if self.model is None:
            raise ValueError("Model not set")
        if self.dataset is None:
            raise ValueError("Dataset not set")
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader
            
            # Configurar dispositivo
            if device == 'auto':
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device(device)
            
            self.model.to(device)
            
            # Crear DataLoader
            train_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Configurar pérdida y optimizador
            criterion = nn.CrossEntropyLoss()
            
            if optimizer == 'adam':
                opt = optim.Adam(self.model.model.parameters(), lr=learning_rate)
            elif optimizer == 'sgd':
                opt = optim.SGD(self.model.model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            
            # Historial
            history = {
                'loss': [],
                'accuracy': []
            }
            
            # Entrenar
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward
                    opt.zero_grad()
                    outputs = self.model.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward
                    loss.backward()
                    opt.step()
                    
                    # Estadísticas
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # Guardar métricas
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100 * correct / total
                
                history['loss'].append(epoch_loss)
                history['accuracy'].append(epoch_acc)
                
                logger.info(f"Epoch [{epoch+1}/{epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")
            
            return history
            
        except ImportError:
            logger.error("PyTorch required for training")
            return {}
    
    def compute_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """Calcula matriz de confusión.
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicciones.
        targets : np.ndarray
            Valores verdaderos.
            
        Returns
        -------
        np.ndarray
            Matriz de confusión.
        """
        from sklearn.metrics import confusion_matrix
        
        self.confusion_matrix = confusion_matrix(targets, predictions)
        return self.confusion_matrix
    
    def get_misclassified_samples(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        images: np.ndarray,
        n_samples: int = 10
    ) -> List[Dict[str, Any]]:
        """Obtiene muestras mal clasificadas.
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicciones.
        targets : np.ndarray
            Valores verdaderos.
        images : np.ndarray
            Imágenes correspondientes.
        n_samples : int
            Número de muestras a retornar.
            
        Returns
        -------
        List[Dict[str, Any]]
            Muestras mal clasificadas.
        """
        misclassified = []
        incorrect_idx = np.where(predictions != targets)[0]
        
        for idx in incorrect_idx[:n_samples]:
            misclassified.append({
                'image': images[idx],
                'predicted': self.class_names[predictions[idx]],
                'actual': self.class_names[targets[idx]],
                'index': idx
            })
        
        return misclassified


class TaskObjectDetection(TaskVision):
    """Tarea de detección de objetos.
    
    Parameters
    ----------
    id : str
        Identificador.
    dataset : Any
        Dataset con imágenes y anotaciones.
    model : Any
        Modelo detector.
    num_classes : int
        Número de clases de objetos.
    class_names : Optional[List[str]]
        Nombres de las clases.
    iou_threshold : float
        Umbral IoU para evaluación.
    """
    
    def __init__(
        self,
        id: str,
        dataset: Any = None,
        model: Any = None,
        num_classes: int = 80,  # COCO classes
        class_names: Optional[List[str]] = None,
        iou_threshold: float = 0.5,
        transform: Optional[Any] = None
    ):
        super().__init__(
            id=id,
            dataset=dataset,
            model=model,
            transform=transform,
            metrics=['map', 'map_50', 'map_75']
        )
        self.num_classes = num_classes
        self.class_names = class_names or self._get_coco_classes()
        self.iou_threshold = iou_threshold
    
    def _get_coco_classes(self) -> List[str]:
        """Obtiene nombres de clases COCO."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def calculate_iou(
        self,
        box1: np.ndarray,
        box2: np.ndarray
    ) -> float:
        """Calcula Intersection over Union.
        
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
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calcular áreas
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calcular unión
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def calculate_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """Calcula Average Precision para una clase.
        
        Parameters
        ----------
        predictions : List[Dict]
            Predicciones con boxes, scores y labels.
        ground_truths : List[Dict]
            Ground truth con boxes y labels.
        iou_threshold : float
            Umbral IoU.
            
        Returns
        -------
        float
            Average Precision.
        """
        # Implementación simplificada de AP
        # Una implementación completa requeriría más lógica
        return 0.0
    
    def non_max_suppression(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """Aplica Non-Maximum Suppression.
        
        Parameters
        ----------
        boxes : np.ndarray
            Cajas detectadas.
        scores : np.ndarray
            Scores de las cajas.
        threshold : float
            Umbral IoU para suprimir.
            
        Returns
        -------
        np.ndarray
            Índices de cajas a mantener.
        """
        if len(boxes) == 0:
            return np.array([])
        
        # Ordenar por score
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            # Mantener el de mayor score
            keep.append(indices[0])
            
            if len(indices) == 1:
                break
            
            # Calcular IoU con el resto
            current_box = boxes[indices[0]]
            other_boxes = boxes[indices[1:]]
            
            ious = np.array([
                self.calculate_iou(current_box, box)
                for box in other_boxes
            ])
            
            # Mantener solo los que no se solapan mucho
            indices = indices[1:][ious < threshold]
        
        return np.array(keep)


class TaskSegmentation(TaskVision):
    """Tarea de segmentación semántica.
    
    Parameters
    ----------
    id : str
        Identificador.
    dataset : Any
        Dataset con imágenes y máscaras.
    model : Any
        Modelo de segmentación.
    num_classes : int
        Número de clases.
    class_names : Optional[List[str]]
        Nombres de las clases.
    ignore_index : int
        Índice a ignorar en evaluación.
    """
    
    def __init__(
        self,
        id: str,
        dataset: Any = None,
        model: Any = None,
        num_classes: int = 21,
        class_names: Optional[List[str]] = None,
        ignore_index: int = 255,
        transform: Optional[Any] = None
    ):
        super().__init__(
            id=id,
            dataset=dataset,
            model=model,
            transform=transform,
            metrics=['iou', 'dice', 'pixel_accuracy']
        )
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
    
    def calculate_iou_per_class(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Calcula IoU por clase.
        
        Parameters
        ----------
        prediction : np.ndarray
            Máscara predicha.
        target : np.ndarray
            Máscara verdadera.
            
        Returns
        -------
        np.ndarray
            IoU por clase.
        """
        ious = []
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
            
            pred_mask = (prediction == cls)
            target_mask = (target == cls)
            
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            
            if union == 0:
                ious.append(1.0)  # No hay píxeles de esta clase
            else:
                ious.append(intersection / union)
        
        return np.array(ious)
    
    def calculate_dice(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Calcula coeficiente Dice.
        
        Parameters
        ----------
        prediction : np.ndarray
            Máscara predicha.
        target : np.ndarray
            Máscara verdadera.
            
        Returns
        -------
        float
            Coeficiente Dice.
        """
        intersection = np.logical_and(prediction, target).sum()
        dice = 2 * intersection / (prediction.sum() + target.sum() + 1e-6)
        return dice
    
    def calculate_pixel_accuracy(
        self,
        prediction: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Calcula precisión por píxel.
        
        Parameters
        ----------
        prediction : np.ndarray
            Máscara predicha.
        target : np.ndarray
            Máscara verdadera.
            
        Returns
        -------
        float
            Precisión por píxel.
        """
        valid = (target != self.ignore_index)
        correct = (prediction == target) & valid
        return correct.sum() / valid.sum()


class TaskVideoAnalysis(TaskVision):
    """Tarea de análisis de video.
    
    Parameters
    ----------
    id : str
        Identificador.
    dataset : Any
        Dataset de videos.
    model : Any
        Modelo para análisis.
    task_type : str
        Tipo de tarea ('classification', 'detection', 'tracking').
    fps : int
        Frames por segundo para procesamiento.
    """
    
    def __init__(
        self,
        id: str,
        dataset: Any = None,
        model: Any = None,
        task_type: str = 'classification',
        fps: int = 30,
        transform: Optional[Any] = None
    ):
        super().__init__(
            id=id,
            dataset=dataset,
            model=model,
            transform=transform
        )
        self.task_type = task_type
        self.fps = fps
        self.video_results = []
    
    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """Procesa un video completo.
        
        Parameters
        ----------
        video_path : Union[str, Path]
            Ruta del video.
        output_path : Optional[Union[str, Path]]
            Ruta para guardar resultado.
        show_progress : bool
            Si mostrar progreso.
            
        Returns
        -------
        Dict[str, Any]
            Resultados del procesamiento.
        """
        try:
            import cv2
            from tqdm import tqdm
        except ImportError:
            logger.error("opencv-python and tqdm required for video processing")
            return {}
        
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configurar writer si se especifica output
        writer = None
        if output_path:
            output_path = Path(output_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.fps,
                (width, height)
            )
        
        # Procesar frames
        frame_results = []
        frame_interval = int(original_fps / self.fps)
        
        iterator = tqdm(range(total_frames)) if show_progress else range(total_frames)
        
        for frame_idx in iterator:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar solo frames según fps objetivo
            if frame_idx % frame_interval == 0:
                # Aplicar modelo
                if self.model:
                    # Convertir BGR a RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    if self.transform:
                        frame_rgb = self.transform(frame_rgb)
                    
                    # Predicción según tipo de tarea
                    if self.task_type == 'classification':
                        prediction = self.model.predict(frame_rgb)
                    elif self.task_type == 'detection':
                        prediction = self.model.predict(frame_rgb)
                    else:
                        prediction = None
                    
                    frame_results.append({
                        'frame': frame_idx,
                        'prediction': prediction
                    })
                
                # Escribir frame si hay output
                if writer:
                    writer.write(frame)
        
        # Limpiar
        cap.release()
        if writer:
            writer.release()
        
        results = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'processed_frames': len(frame_results),
            'fps': self.fps,
            'frame_results': frame_results
        }
        
        self.video_results.append(results)
        return results