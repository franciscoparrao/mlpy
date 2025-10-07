"""
Modelos de Computer Vision.
"""

from typing import Optional, List, Tuple, Union, Dict, Any
import numpy as np
from pathlib import Path
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VisionModel(ABC):
    """Modelo base de visión."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: Optional[int] = None,
        pretrained: bool = True,
        device: str = 'auto'
    ):
        """Inicializa modelo de visión.
        
        Parameters
        ----------
        model_name : str
            Nombre del modelo.
        num_classes : Optional[int]
            Número de clases de salida.
        pretrained : bool
            Si usar pesos preentrenados.
        device : str
            Dispositivo ('cpu', 'cuda', 'auto').
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.device = self._get_device(device)
        self.model = None
        self._setup()
    
    def _get_device(self, device: str):
        """Obtiene dispositivo de cómputo."""
        try:
            import torch
            
            if device == 'auto':
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(device)
        except ImportError:
            return 'cpu'
    
    @abstractmethod
    def _setup(self):
        """Configura el modelo."""
        pass
    
    @abstractmethod
    def predict(self, image: Any) -> Any:
        """Realiza predicción."""
        pass
    
    def train(self):
        """Pone modelo en modo entrenamiento."""
        if self.model:
            self.model.train()
    
    def eval(self):
        """Pone modelo en modo evaluación."""
        if self.model:
            self.model.eval()
    
    def to(self, device):
        """Mueve modelo a dispositivo."""
        if self.model:
            self.model = self.model.to(device)
            self.device = device
        return self
    
    def save(self, path: Union[str, Path]):
        """Guarda el modelo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            
            checkpoint = {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'state_dict': self.model.state_dict() if self.model else None,
                'model_config': self._get_config()
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
        except ImportError:
            logger.error("PyTorch required to save model")
    
    def load(self, path: Union[str, Path]):
        """Carga el modelo."""
        path = Path(path)
        
        try:
            import torch
            
            checkpoint = torch.load(path, map_location=self.device)
            
            if self.model and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Model loaded from {path}")
        except ImportError:
            logger.error("PyTorch required to load model")
    
    def _get_config(self) -> Dict:
        """Obtiene configuración del modelo."""
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained
        }


class ImageClassifier(VisionModel):
    """Clasificador de imágenes.
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo ('resnet18', 'resnet50', 'vgg16', 'mobilenet_v2', etc.).
    num_classes : int
        Número de clases.
    pretrained : bool
        Si usar pesos preentrenados.
    freeze_backbone : bool
        Si congelar el backbone.
    dropout : float
        Dropout para la capa final.
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        device: str = 'auto'
    ):
        self.freeze_backbone = freeze_backbone
        self.dropout = dropout
        super().__init__(model_name, num_classes, pretrained, device)
    
    def _setup(self):
        """Configura el clasificador."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            
            # Cargar modelo base
            if self.model_name == 'resnet18':
                self.model = models.resnet18(pretrained=self.pretrained)
                in_features = self.model.fc.in_features
                
            elif self.model_name == 'resnet50':
                self.model = models.resnet50(pretrained=self.pretrained)
                in_features = self.model.fc.in_features
                
            elif self.model_name == 'vgg16':
                self.model = models.vgg16(pretrained=self.pretrained)
                in_features = self.model.classifier[-1].in_features
                
            elif self.model_name == 'mobilenet_v2':
                self.model = models.mobilenet_v2(pretrained=self.pretrained)
                in_features = self.model.classifier[-1].in_features
                
            elif self.model_name == 'efficientnet_b0':
                self.model = models.efficientnet_b0(pretrained=self.pretrained)
                in_features = self.model.classifier[-1].in_features
                
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            # Modificar capa final
            if self.num_classes != 1000:
                if 'resnet' in self.model_name:
                    # ResNet
                    if self.dropout > 0:
                        self.model.fc = nn.Sequential(
                            nn.Dropout(self.dropout),
                            nn.Linear(in_features, self.num_classes)
                        )
                    else:
                        self.model.fc = nn.Linear(in_features, self.num_classes)
                        
                elif 'vgg' in self.model_name:
                    # VGG
                    self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
                    
                elif 'mobilenet' in self.model_name or 'efficientnet' in self.model_name:
                    # MobileNet/EfficientNet
                    self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            
            # Congelar backbone si se especifica
            if self.freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # Descongelar capa final
                if 'resnet' in self.model_name:
                    for param in self.model.fc.parameters():
                        param.requires_grad = True
                elif 'vgg' in self.model_name:
                    for param in self.model.classifier[-1].parameters():
                        param.requires_grad = True
                else:
                    for param in self.model.classifier.parameters():
                        param.requires_grad = True
            
            # Mover a dispositivo
            self.model = self.model.to(self.device)
            
        except ImportError:
            logger.error("torchvision required for image classifier")
            self.model = None
    
    def predict(
        self,
        image: Union[np.ndarray, Any],
        return_probs: bool = False
    ) -> Union[int, np.ndarray]:
        """Predice clase de imagen.
        
        Parameters
        ----------
        image : Union[np.ndarray, Any]
            Imagen a clasificar.
        return_probs : bool
            Si retornar probabilidades.
            
        Returns
        -------
        Union[int, np.ndarray]
            Clase predicha o probabilidades.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            import torch
            import torch.nn.functional as F
            
            self.model.eval()
            
            with torch.no_grad():
                # Preparar imagen
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                
                # Añadir batch dimension si es necesario
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                # Mover a dispositivo
                image = image.to(self.device)
                
                # Predicción
                outputs = self.model(image)
                
                if return_probs:
                    probs = F.softmax(outputs, dim=1)
                    return probs.cpu().numpy()
                else:
                    _, predicted = torch.max(outputs, 1)
                    return predicted.cpu().item()
                    
        except ImportError:
            logger.error("PyTorch required for prediction")
            return None


class ObjectDetector(VisionModel):
    """Detector de objetos.
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo ('fasterrcnn_resnet50_fpn', 'retinanet', 'ssd', 'yolo').
    num_classes : int
        Número de clases.
    pretrained : bool
        Si usar pesos preentrenados.
    min_score : float
        Score mínimo para detecciones.
    """
    
    def __init__(
        self,
        model_name: str = 'fasterrcnn_resnet50_fpn',
        num_classes: int = 91,  # COCO classes
        pretrained: bool = True,
        min_score: float = 0.5,
        device: str = 'auto'
    ):
        self.min_score = min_score
        super().__init__(model_name, num_classes, pretrained, device)
    
    def _setup(self):
        """Configura el detector."""
        try:
            import torch
            from torchvision import models
            from torchvision.models.detection import FasterRCNN
            from torchvision.models.detection.rpn import AnchorGenerator
            
            if self.model_name == 'fasterrcnn_resnet50_fpn':
                if self.pretrained:
                    self.model = models.detection.fasterrcnn_resnet50_fpn(
                        pretrained=True
                    )
                else:
                    self.model = models.detection.fasterrcnn_resnet50_fpn(
                        pretrained=False,
                        num_classes=self.num_classes
                    )
                
                # Modificar número de clases si es necesario
                if not self.pretrained and self.num_classes != 91:
                    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
                    self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
                        in_features,
                        self.num_classes
                    )
            
            elif self.model_name == 'retinanet_resnet50_fpn':
                self.model = models.detection.retinanet_resnet50_fpn(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 91
                )
            
            elif self.model_name == 'ssd300_vgg16':
                self.model = models.detection.ssd300_vgg16(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 91
                )
            
            else:
                raise ValueError(f"Unknown detector: {self.model_name}")
            
            # Mover a dispositivo
            self.model = self.model.to(self.device)
            
        except ImportError:
            logger.error("torchvision required for object detection")
            self.model = None
    
    def predict(
        self,
        image: Union[np.ndarray, Any]
    ) -> List[Dict[str, Any]]:
        """Detecta objetos en imagen.
        
        Parameters
        ----------
        image : Union[np.ndarray, Any]
            Imagen para detección.
            
        Returns
        -------
        List[Dict[str, Any]]
            Lista de detecciones con boxes, labels y scores.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            import torch
            
            self.model.eval()
            
            with torch.no_grad():
                # Preparar imagen
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                
                # Normalizar si es necesario
                if image.max() > 1:
                    image = image / 255.0
                
                # Añadir batch dimension
                if len(image.shape) == 3:
                    image = [image]
                
                # Mover a dispositivo
                image = [img.to(self.device) for img in image]
                
                # Predicción
                predictions = self.model(image)
                
                # Procesar predicciones
                detections = []
                for pred in predictions:
                    # Filtrar por score
                    keep = pred['scores'] > self.min_score
                    
                    detection = {
                        'boxes': pred['boxes'][keep].cpu().numpy(),
                        'labels': pred['labels'][keep].cpu().numpy(),
                        'scores': pred['scores'][keep].cpu().numpy()
                    }
                    
                    detections.append(detection)
                
                return detections[0] if len(detections) == 1 else detections
                
        except ImportError:
            logger.error("PyTorch required for prediction")
            return []


class SemanticSegmentation(VisionModel):
    """Segmentación semántica.
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo ('deeplabv3_resnet50', 'fcn_resnet50', 'lraspp_mobilenet_v3').
    num_classes : int
        Número de clases.
    pretrained : bool
        Si usar pesos preentrenados.
    """
    
    def __init__(
        self,
        model_name: str = 'deeplabv3_resnet50',
        num_classes: int = 21,  # PASCAL VOC classes
        pretrained: bool = True,
        device: str = 'auto'
    ):
        super().__init__(model_name, num_classes, pretrained, device)
    
    def _setup(self):
        """Configura el modelo de segmentación."""
        try:
            import torch
            from torchvision import models
            
            if self.model_name == 'deeplabv3_resnet50':
                self.model = models.segmentation.deeplabv3_resnet50(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 21
                )
            
            elif self.model_name == 'deeplabv3_resnet101':
                self.model = models.segmentation.deeplabv3_resnet101(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 21
                )
            
            elif self.model_name == 'fcn_resnet50':
                self.model = models.segmentation.fcn_resnet50(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 21
                )
            
            elif self.model_name == 'lraspp_mobilenet_v3_large':
                self.model = models.segmentation.lraspp_mobilenet_v3_large(
                    pretrained=self.pretrained,
                    num_classes=self.num_classes if not self.pretrained else 21
                )
            
            else:
                raise ValueError(f"Unknown segmentation model: {self.model_name}")
            
            # Mover a dispositivo
            self.model = self.model.to(self.device)
            
        except ImportError:
            logger.error("torchvision required for semantic segmentation")
            self.model = None
    
    def predict(
        self,
        image: Union[np.ndarray, Any],
        return_probs: bool = False
    ) -> np.ndarray:
        """Realiza segmentación semántica.
        
        Parameters
        ----------
        image : Union[np.ndarray, Any]
            Imagen a segmentar.
        return_probs : bool
            Si retornar probabilidades por clase.
            
        Returns
        -------
        np.ndarray
            Máscara de segmentación o probabilidades.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            import torch
            import torch.nn.functional as F
            
            self.model.eval()
            
            with torch.no_grad():
                # Preparar imagen
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()
                
                # Normalizar si es necesario
                if image.max() > 1:
                    image = image / 255.0
                
                # Añadir batch dimension
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                # Mover a dispositivo
                image = image.to(self.device)
                
                # Predicción
                output = self.model(image)['out']
                
                if return_probs:
                    # Retornar probabilidades
                    probs = F.softmax(output, dim=1)
                    return probs.cpu().numpy()
                else:
                    # Retornar máscara
                    _, predicted = torch.max(output, 1)
                    return predicted.cpu().numpy()
                    
        except ImportError:
            logger.error("PyTorch required for prediction")
            return None


def load_pretrained_model(
    model_type: str,
    model_name: str,
    **kwargs
) -> VisionModel:
    """Carga modelo preentrenado.
    
    Parameters
    ----------
    model_type : str
        Tipo de modelo ('classifier', 'detector', 'segmentation').
    model_name : str
        Nombre del modelo.
    **kwargs
        Argumentos adicionales para el modelo.
        
    Returns
    -------
    VisionModel
        Modelo cargado.
    """
    if model_type == 'classifier':
        return ImageClassifier(model_name=model_name, pretrained=True, **kwargs)
    elif model_type == 'detector':
        return ObjectDetector(model_name=model_name, pretrained=True, **kwargs)
    elif model_type == 'segmentation':
        return SemanticSegmentation(model_name=model_name, pretrained=True, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model(
    model_type: str,
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    **kwargs
) -> VisionModel:
    """Crea modelo de visión.
    
    Parameters
    ----------
    model_type : str
        Tipo de modelo.
    model_name : str
        Nombre del modelo.
    num_classes : int
        Número de clases.
    pretrained : bool
        Si usar pesos preentrenados.
    **kwargs
        Argumentos adicionales.
        
    Returns
    -------
    VisionModel
        Modelo creado.
    """
    if model_type == 'classifier':
        return ImageClassifier(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == 'detector':
        return ObjectDetector(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == 'segmentation':
        return SemanticSegmentation(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")