"""
Módulo de Computer Vision para MLPY.

Integración con torchvision para tareas de visión por computadora.
"""

from .transforms import (
    ImageTransform,
    Compose,
    Resize,
    CenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ColorJitter,
    Normalize,
    ToTensor,
    ToPILImage,
    create_augmentation_pipeline
)

from .datasets import (
    ImageDataset,
    ImageFolder,
    VideoDataset,
    create_data_loader,
    split_dataset
)

from .models import (
    VisionModel,
    ImageClassifier,
    ObjectDetector,
    SemanticSegmentation,
    load_pretrained_model,
    create_model
)

from .tasks import (
    TaskVision,
    TaskImageClassification,
    TaskObjectDetection,
    TaskSegmentation,
    TaskVideoAnalysis
)

from .utils import (
    visualize_predictions,
    draw_bounding_boxes,
    draw_segmentation_masks,
    plot_image_grid,
    save_predictions,
    calculate_iou,
    calculate_map
)

from .video import (
    VideoProcessor,
    FrameExtractor,
    VideoWriter,
    apply_model_to_video,
    track_objects
)

__all__ = [
    # Transforms
    'ImageTransform',
    'Compose',
    'Resize',
    'CenterCrop',
    'RandomCrop',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'ColorJitter',
    'Normalize',
    'ToTensor',
    'ToPILImage',
    'create_augmentation_pipeline',
    
    # Datasets
    'ImageDataset',
    'ImageFolder',
    'VideoDataset',
    'create_data_loader',
    'split_dataset',
    
    # Models
    'VisionModel',
    'ImageClassifier',
    'ObjectDetector',
    'SemanticSegmentation',
    'load_pretrained_model',
    'create_model',
    
    # Tasks
    'TaskVision',
    'TaskImageClassification',
    'TaskObjectDetection',
    'TaskSegmentation',
    'TaskVideoAnalysis',
    
    # Utils
    'visualize_predictions',
    'draw_bounding_boxes',
    'draw_segmentation_masks',
    'plot_image_grid',
    'save_predictions',
    'calculate_iou',
    'calculate_map',
    
    # Video
    'VideoProcessor',
    'FrameExtractor',
    'VideoWriter',
    'apply_model_to_video',
    'track_objects'
]