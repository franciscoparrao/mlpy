"""
Datasets y DataLoaders para PyTorch.

Convierte tasks de MLPY a datasets de PyTorch.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import warnings

from ...tasks import Task, TaskClassif, TaskRegr


class MLPYDataset(Dataset):
    """Dataset de PyTorch para tasks de MLPY.
    
    Parameters
    ----------
    task : Task
        Tarea de MLPY.
    training : bool
        Si es para entrenamiento (incluye targets).
    transform : Optional[callable]
        Transformación a aplicar a los datos.
    target_transform : Optional[callable]
        Transformación a aplicar a los targets.
    """
    
    def __init__(
        self,
        task: Task,
        training: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None
    ):
        self.task = task
        self.training = training
        self.transform = transform
        self.target_transform = target_transform
        
        # Obtener datos
        self.X = task.X()
        
        # Convertir a numpy si es necesario
        if isinstance(self.X, pd.DataFrame):
            self.X = self.X.values
        
        # Normalizar tipos de datos
        if self.X.dtype != np.float32:
            self.X = self.X.astype(np.float32)
        
        # Obtener targets si es entrenamiento
        self.y = None
        if training and task.has_target():
            self.y = task.truth()
            
            # Convertir a numpy si es necesario
            if isinstance(self.y, pd.Series):
                self.y = self.y.values
            
            # Para clasificación, asegurar que son enteros
            if isinstance(task, TaskClassif):
                if self.y.dtype != np.int64:
                    self.y = self.y.astype(np.int64)
            else:
                # Para regresión, usar float32
                if self.y.dtype != np.float32:
                    self.y = self.y.astype(np.float32)
    
    def __len__(self) -> int:
        """Retorna el tamaño del dataset."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Obtiene un item del dataset.
        
        Parameters
        ----------
        idx : int
            Índice del item.
            
        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            (X, y) si es entrenamiento, solo X si es predicción.
        """
        x = self.X[idx]
        
        # Aplicar transformación si existe
        if self.transform:
            x = self.transform(x)
        
        # Convertir a tensor
        x_tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else torch.tensor(x)
        
        # Si es entrenamiento, retornar también el target
        if self.training and self.y is not None:
            y = self.y[idx]
            
            # Aplicar transformación al target si existe
            if self.target_transform:
                y = self.target_transform(y)
            
            # Convertir a tensor
            y_tensor = torch.from_numpy(np.array(y)) if isinstance(y, (np.ndarray, np.generic)) else torch.tensor(y)
            
            return x_tensor, y_tensor
        
        return x_tensor


def create_data_loaders(
    task: Task,
    batch_size: int = 32,
    validation_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    transform: Optional[callable] = None,
    target_transform: Optional[callable] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Crea DataLoaders para entrenamiento y validación.
    
    Parameters
    ----------
    task : Task
        Tarea de MLPY.
    batch_size : int
        Tamaño del batch.
    validation_split : float
        Proporción de datos para validación.
    shuffle : bool
        Si mezclar los datos.
    num_workers : int
        Número de workers para carga de datos.
    pin_memory : bool
        Si fijar memoria (útil para GPU).
    transform : Optional[callable]
        Transformación para los datos.
    target_transform : Optional[callable]
        Transformación para los targets.
        
    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader]]
        DataLoaders de entrenamiento y validación.
    """
    # Crear dataset completo
    dataset = MLPYDataset(
        task,
        training=True,
        transform=transform,
        target_transform=target_transform
    )
    
    # Si no hay split de validación, retornar solo train loader
    if validation_split <= 0 or validation_split >= 1:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return train_loader, None
    
    # Calcular tamaños
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    # Split del dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Para reproducibilidad
    )
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No mezclar validación
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


class MLPYDataLoader(DataLoader):
    """DataLoader especializado para MLPY.
    
    Extiende DataLoader de PyTorch con funcionalidad específica de MLPY.
    """
    
    def __init__(
        self,
        task: Task,
        batch_size: int = 32,
        shuffle: bool = True,
        training: bool = True,
        **kwargs
    ):
        """Inicializa el DataLoader.
        
        Parameters
        ----------
        task : Task
            Tarea de MLPY.
        batch_size : int
            Tamaño del batch.
        shuffle : bool
            Si mezclar los datos.
        training : bool
            Si es para entrenamiento.
        **kwargs
            Argumentos adicionales para DataLoader.
        """
        dataset = MLPYDataset(task, training=training)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
        self.task = task
        self.training = training


def create_image_data_loaders(
    task: Task,
    image_column: str,
    batch_size: int = 32,
    validation_split: float = 0.2,
    image_size: Tuple[int, int] = (224, 224),
    augmentation: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Crea DataLoaders para datos de imagen.
    
    Parameters
    ----------
    task : Task
        Tarea con rutas de imágenes.
    image_column : str
        Columna con rutas de imágenes.
    batch_size : int
        Tamaño del batch.
    validation_split : float
        Proporción para validación.
    image_size : Tuple[int, int]
        Tamaño de las imágenes.
    augmentation : bool
        Si aplicar augmentación de datos.
    num_workers : int
        Workers para carga de datos.
        
    Returns
    -------
    Tuple[DataLoader, Optional[DataLoader]]
        DataLoaders de entrenamiento y validación.
    """
    try:
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        raise ImportError("torchvision required for image data. Install with: pip install torchvision")
    
    # Definir transformaciones
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Crear dataset personalizado para imágenes
    class ImageDataset(Dataset):
        def __init__(self, task, image_col, transform=None):
            self.task = task
            self.image_paths = task.data[image_col].values
            self.transform = transform
            self.targets = None
            
            if task.has_target():
                self.targets = task.truth()
                if isinstance(self.targets, pd.Series):
                    self.targets = self.targets.values
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # Cargar imagen
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            # Aplicar transformaciones
            if self.transform:
                image = self.transform(image)
            
            # Retornar con o sin target
            if self.targets is not None:
                target = self.targets[idx]
                if isinstance(task, TaskClassif):
                    target = int(target)
                else:
                    target = float(target)
                return image, torch.tensor(target)
            
            return image
    
    # Crear datasets
    full_dataset = ImageDataset(task, image_column, train_transform)
    
    # Split para validación
    if validation_split > 0 and validation_split < 1:
        total_size = len(full_dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Cambiar transform para validación
        val_dataset.dataset.transform = val_transform
        
        # Crear loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    # Solo train loader
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, None