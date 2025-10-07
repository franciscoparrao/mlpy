"""
Soporte para modelos pre-entrenados en PyTorch.

Facilita el uso de modelos pre-entrenados para transfer learning.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_pretrained_model(
    model_name: str,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: Optional[float] = None
) -> nn.Module:
    """Carga un modelo pre-entrenado.
    
    Parameters
    ----------
    model_name : str
        Nombre del modelo (e.g., 'resnet50', 'efficientnet_b0', 'bert').
    num_classes : Optional[int]
        Número de clases para la tarea. Si None, retorna el modelo sin modificar.
    pretrained : bool
        Si cargar pesos pre-entrenados.
    freeze_backbone : bool
        Si congelar las capas del backbone.
    dropout : Optional[float]
        Dropout para la capa de clasificación.
        
    Returns
    -------
    nn.Module
        Modelo cargado.
    """
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError("torchvision required. Install with: pip install torchvision")
    
    # Modelos de visión disponibles
    vision_models = {
        # ResNet family
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        
        # DenseNet family
        'densenet121': models.densenet121,
        'densenet161': models.densenet161,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        
        # VGG family
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'vgg11_bn': models.vgg11_bn,
        'vgg13_bn': models.vgg13_bn,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
        
        # MobileNet family
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        
        # EfficientNet family
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
        'efficientnet_b5': models.efficientnet_b5,
        'efficientnet_b6': models.efficientnet_b6,
        'efficientnet_b7': models.efficientnet_b7,
        
        # Vision Transformer
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'vit_l_16': models.vit_l_16,
        'vit_l_32': models.vit_l_32,
        
        # Other architectures
        'alexnet': models.alexnet,
        'googlenet': models.googlenet,
        'inception_v3': models.inception_v3,
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
    }
    
    # Verificar si es un modelo de visión
    if model_name in vision_models:
        model = _load_vision_model(
            vision_models[model_name],
            model_name,
            num_classes,
            pretrained,
            freeze_backbone,
            dropout
        )
    else:
        # Intentar cargar modelo de Hugging Face
        model = _load_huggingface_model(
            model_name,
            num_classes,
            pretrained,
            freeze_backbone
        )
    
    logger.info(f"Loaded {model_name} model (pretrained={pretrained}, frozen={freeze_backbone})")
    return model


def _load_vision_model(
    model_fn,
    model_name: str,
    num_classes: Optional[int],
    pretrained: bool,
    freeze_backbone: bool,
    dropout: Optional[float]
) -> nn.Module:
    """Carga un modelo de visión de torchvision."""
    
    # Cargar modelo
    if pretrained:
        model = model_fn(weights='DEFAULT')
    else:
        model = model_fn(weights=None)
    
    # Congelar backbone si se solicita
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Modificar capa de salida si se especifica num_classes
    if num_classes is not None:
        # Encontrar y reemplazar la capa de clasificación
        if hasattr(model, 'fc'):  # ResNet, DenseNet
            in_features = model.fc.in_features
            if dropout:
                model.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
            else:
                model.fc = nn.Linear(in_features, num_classes)
        
        elif hasattr(model, 'classifier'):  # VGG, MobileNet, EfficientNet
            if isinstance(model.classifier, nn.Sequential):
                # Obtener dimensión de entrada
                in_features = model.classifier[-1].in_features
                if dropout:
                    model.classifier[-1] = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(in_features, num_classes)
                    )
                else:
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                if dropout:
                    model.classifier = nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(in_features, num_classes)
                    )
                else:
                    model.classifier = nn.Linear(in_features, num_classes)
        
        elif hasattr(model, 'heads'):  # Vision Transformer
            in_features = model.heads.head.in_features
            if dropout:
                model.heads.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_features, num_classes)
                )
            else:
                model.heads.head = nn.Linear(in_features, num_classes)
    
    return model


def _load_huggingface_model(
    model_name: str,
    num_classes: Optional[int],
    pretrained: bool,
    freeze_backbone: bool
) -> nn.Module:
    """Carga un modelo de Hugging Face."""
    try:
        from transformers import AutoModel, AutoModelForSequenceClassification
    except ImportError:
        raise ImportError(
            f"Model '{model_name}' requires transformers. "
            "Install with: pip install transformers"
        )
    
    # Cargar modelo
    if num_classes is not None:
        # Modelo para clasificación
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        # Modelo base
        model = AutoModel.from_pretrained(model_name)
    
    # Congelar backbone si se solicita
    if freeze_backbone:
        # Congelar todo excepto la capa de clasificación
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'pooler' not in name:
                param.requires_grad = False
    
    return model


def get_available_models() -> Dict[str, List[str]]:
    """Obtiene lista de modelos pre-entrenados disponibles.
    
    Returns
    -------
    Dict[str, List[str]]
        Diccionario con categorías y modelos disponibles.
    """
    models = {
        'vision_classification': [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn',
            'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large',
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
            'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
            'alexnet', 'googlenet', 'inception_v3', 'shufflenet_v2_x1_0'
        ],
        'nlp_classification': [
            'bert-base-uncased', 'bert-large-uncased',
            'roberta-base', 'roberta-large',
            'distilbert-base-uncased',
            'albert-base-v2', 'albert-large-v2',
            'xlnet-base-cased', 'xlnet-large-cased'
        ],
        'object_detection': [
            'fasterrcnn_resnet50_fpn', 'fasterrcnn_mobilenet_v3_large_fpn',
            'maskrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn',
            'retinanet_resnet50_fpn'
        ],
        'segmentation': [
            'fcn_resnet50', 'fcn_resnet101',
            'deeplabv3_resnet50', 'deeplabv3_resnet101',
            'deeplabv3_mobilenet_v3_large', 'lraspp_mobilenet_v3_large'
        ]
    }
    
    return models


def finetune_model(
    model: nn.Module,
    layers_to_unfreeze: Optional[Union[int, List[str]]] = None,
    learning_rates: Optional[Dict[str, float]] = None
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """Configura un modelo para fine-tuning.
    
    Parameters
    ----------
    model : nn.Module
        Modelo a ajustar.
    layers_to_unfreeze : Optional[Union[int, List[str]]]
        Número de capas a descongelar desde el final, o lista de nombres.
    learning_rates : Optional[Dict[str, float]]
        Learning rates diferenciados por capa.
        
    Returns
    -------
    Tuple[nn.Module, List[Dict[str, Any]]]
        Modelo configurado y grupos de parámetros para el optimizador.
    """
    # Descongelar capas específicas
    if layers_to_unfreeze is not None:
        if isinstance(layers_to_unfreeze, int):
            # Descongelar las últimas N capas
            all_layers = list(model.named_parameters())
            total_layers = len(all_layers)
            
            for i, (name, param) in enumerate(all_layers):
                if i >= total_layers - layers_to_unfreeze:
                    param.requires_grad = True
                    logger.info(f"Unfroze layer: {name}")
        else:
            # Descongelar capas por nombre
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
                    logger.info(f"Unfroze layer: {name}")
    
    # Configurar grupos de parámetros con diferentes learning rates
    param_groups = []
    
    if learning_rates:
        # Agrupar parámetros por learning rate
        for lr_name, lr_value in learning_rates.items():
            params = []
            for name, param in model.named_parameters():
                if param.requires_grad and lr_name in name:
                    params.append(param)
            
            if params:
                param_groups.append({
                    'params': params,
                    'lr': lr_value,
                    'name': lr_name
                })
        
        # Agregar parámetros restantes con learning rate por defecto
        remaining_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Verificar si ya está en algún grupo
                in_group = False
                for group_name in learning_rates.keys():
                    if group_name in name:
                        in_group = True
                        break
                
                if not in_group:
                    remaining_params.append(param)
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'name': 'default'
            })
    else:
        # Un solo grupo con todos los parámetros entrenables
        param_groups = [{
            'params': [p for p in model.parameters() if p.requires_grad],
            'name': 'all'
        }]
    
    # Log resumen
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Fine-tuning configuration:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Parameter groups: {len(param_groups)}")
    
    for i, group in enumerate(param_groups):
        group_params = sum(p.numel() for p in group['params'])
        lr = group.get('lr', 'default')
        logger.info(f"    Group {i} ({group['name']}): {group_params:,} params, lr={lr}")
    
    return model, param_groups


def extract_features(
    model: nn.Module,
    layer_name: str,
    input_tensor: torch.Tensor
) -> torch.Tensor:
    """Extrae features de una capa intermedia.
    
    Parameters
    ----------
    model : nn.Module
        Modelo del que extraer features.
    layer_name : str
        Nombre de la capa.
    input_tensor : torch.Tensor
        Tensor de entrada.
        
    Returns
    -------
    torch.Tensor
        Features extraídas.
    """
    features = None
    
    def hook_fn(module, input, output):
        nonlocal features
        features = output
    
    # Registrar hook
    layer = dict(model.named_modules())[layer_name]
    hook = layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        _ = model(input_tensor)
    
    # Remover hook
    hook.remove()
    
    return features


def create_feature_extractor(
    model: nn.Module,
    layer_names: List[str]
) -> nn.Module:
    """Crea un extractor de features de múltiples capas.
    
    Parameters
    ----------
    model : nn.Module
        Modelo base.
    layer_names : List[str]
        Nombres de las capas de las que extraer features.
        
    Returns
    -------
    nn.Module
        Extractor de features.
    """
    
    class FeatureExtractor(nn.Module):
        def __init__(self, model, layer_names):
            super().__init__()
            self.model = model
            self.layer_names = layer_names
            self.features = {}
            
            # Registrar hooks
            for name in layer_names:
                layer = dict(model.named_modules())[name]
                layer.register_forward_hook(self._get_hook(name))
        
        def _get_hook(self, name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        def forward(self, x):
            self.features = {}
            _ = self.model(x)
            return self.features
    
    return FeatureExtractor(model, layer_names)