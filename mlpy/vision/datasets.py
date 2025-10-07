"""
Datasets para Computer Vision.
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class ImageDataset:
    """Dataset de imágenes base.
    
    Parameters
    ----------
    root : Union[str, Path]
        Directorio raíz del dataset.
    transform : Optional[Callable]
        Transformaciones a aplicar.
    target_transform : Optional[Callable]
        Transformaciones para targets.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self._load_dataset()
    
    def _load_dataset(self):
        """Carga el dataset."""
        raise NotImplementedError
    
    def __len__(self):
        """Número de muestras."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Obtiene una muestra.
        
        Parameters
        ----------
        idx : int
            Índice de la muestra.
            
        Returns
        -------
        Tuple[Any, Any]
            Imagen y etiqueta.
        """
        path, target = self.samples[idx]
        sample = self._load_image(path)
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sample, target
    
    def _load_image(self, path: Path):
        """Carga una imagen."""
        try:
            from PIL import Image
            return Image.open(path).convert('RGB')
        except ImportError:
            # Fallback con opencv
            try:
                import cv2
                img = cv2.imread(str(path))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except ImportError:
                raise ImportError("PIL or opencv-python required to load images")
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Obtiene distribución de clases."""
        distribution = defaultdict(int)
        for _, target in self.samples:
            if isinstance(target, int):
                class_name = self.classes[target] if self.classes else str(target)
            else:
                class_name = str(target)
            distribution[class_name] += 1
        return dict(distribution)


class ImageFolder(ImageDataset):
    """Dataset de carpetas de imágenes.
    
    Estructura esperada:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img3.jpg
            img4.jpg
    
    Parameters
    ----------
    root : Union[str, Path]
        Directorio raíz.
    transform : Optional[Callable]
        Transformaciones.
    target_transform : Optional[Callable]
        Transformaciones de targets.
    extensions : Optional[List[str]]
        Extensiones de archivo válidas.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[List[str]] = None
    ):
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        super().__init__(root, transform, target_transform)
    
    def _load_dataset(self):
        """Carga dataset desde estructura de carpetas."""
        if not self.root.exists():
            raise ValueError(f"Root directory not found: {self.root}")
        
        # Encontrar clases
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Cargar muestras
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    self.samples.append((file_path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")


class VideoDataset:
    """Dataset de videos.
    
    Parameters
    ----------
    root : Union[str, Path]
        Directorio raíz.
    frame_transform : Optional[Callable]
        Transformaciones para frames.
    video_transform : Optional[Callable]
        Transformaciones para videos completos.
    extensions : Optional[List[str]]
        Extensiones de video válidas.
    frames_per_clip : int
        Frames por clip.
    frame_rate : Optional[int]
        Frame rate objetivo.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        frame_transform: Optional[Callable] = None,
        video_transform: Optional[Callable] = None,
        extensions: Optional[List[str]] = None,
        frames_per_clip: int = 16,
        frame_rate: Optional[int] = None
    ):
        self.root = Path(root)
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.extensions = extensions or ['.mp4', '.avi', '.mov', '.mkv']
        self.frames_per_clip = frames_per_clip
        self.frame_rate = frame_rate
        self.videos = []
        self._load_videos()
    
    def _load_videos(self):
        """Carga lista de videos."""
        if not self.root.exists():
            raise ValueError(f"Root directory not found: {self.root}")
        
        # Buscar videos
        for ext in self.extensions:
            self.videos.extend(self.root.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(self.videos)} videos")
    
    def __len__(self):
        """Número de videos."""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Obtiene un clip de video.
        
        Parameters
        ----------
        idx : int
            Índice del video.
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            Frames y metadata.
        """
        video_path = self.videos[idx]
        frames = self._load_video_frames(video_path)
        
        if self.frame_transform:
            frames = [self.frame_transform(f) for f in frames]
        
        if self.video_transform:
            frames = self.video_transform(frames)
        
        metadata = {
            'path': str(video_path),
            'name': video_path.stem,
            'n_frames': len(frames)
        }
        
        return np.array(frames), metadata
    
    def _load_video_frames(self, path: Path) -> List[np.ndarray]:
        """Carga frames de un video."""
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python required for video loading")
        
        cap = cv2.VideoCapture(str(path))
        frames = []
        
        # Configurar frame rate si se especifica
        if self.frame_rate:
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(original_fps / self.frame_rate)
        else:
            frame_interval = 1
        
        frame_count = 0
        while len(frames) < self.frames_per_clip:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convertir BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        # Pad si es necesario
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.frames_per_clip]


def create_data_loader(
    dataset: Union[ImageDataset, VideoDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
):
    """Crea un DataLoader para el dataset.
    
    Parameters
    ----------
    dataset : Union[ImageDataset, VideoDataset]
        Dataset a cargar.
    batch_size : int
        Tamaño del batch.
    shuffle : bool
        Si mezclar datos.
    num_workers : int
        Número de workers para carga.
    pin_memory : bool
        Si usar pinned memory.
    drop_last : bool
        Si descartar último batch incompleto.
        
    Returns
    -------
    DataLoader
        DataLoader configurado.
    """
    try:
        from torch.utils.data import DataLoader
        
        # Crear wrapper si es necesario
        if not hasattr(dataset, '__getitem__'):
            raise ValueError("Dataset must implement __getitem__")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        return loader
    except ImportError:
        raise ImportError("PyTorch required for DataLoader")


def split_dataset(
    dataset: ImageDataset,
    split_ratio: Union[float, Tuple[float, float, float]] = 0.8,
    seed: Optional[int] = None
) -> Union[Tuple[ImageDataset, ImageDataset], Tuple[ImageDataset, ImageDataset, ImageDataset]]:
    """Divide dataset en train/val o train/val/test.
    
    Parameters
    ----------
    dataset : ImageDataset
        Dataset a dividir.
    split_ratio : Union[float, Tuple[float, float, float]]
        Ratio de división. Float para train/val, tupla para train/val/test.
    seed : Optional[int]
        Semilla aleatoria.
        
    Returns
    -------
    Union[Tuple[ImageDataset, ImageDataset], Tuple[ImageDataset, ImageDataset, ImageDataset]]
        Datasets divididos.
    """
    try:
        from torch.utils.data import random_split
        import torch
        
        if seed is not None:
            torch.manual_seed(seed)
        
        n_samples = len(dataset)
        
        if isinstance(split_ratio, float):
            # Train/Val split
            n_train = int(n_samples * split_ratio)
            n_val = n_samples - n_train
            
            train_dataset, val_dataset = random_split(
                dataset,
                [n_train, n_val]
            )
            
            return train_dataset, val_dataset
        else:
            # Train/Val/Test split
            train_ratio, val_ratio, test_ratio = split_ratio
            
            # Normalizar ratios
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total
            
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            n_test = n_samples - n_train - n_val
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset,
                [n_train, n_val, n_test]
            )
            
            return train_dataset, val_dataset, test_dataset
    except ImportError:
        # Fallback manual
        import random
        
        if seed is not None:
            random.seed(seed)
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        if isinstance(split_ratio, float):
            split_point = int(len(indices) * split_ratio)
            train_indices = indices[:split_point]
            val_indices = indices[split_point:]
            
            # Crear subsets
            train_dataset = DatasetSubset(dataset, train_indices)
            val_dataset = DatasetSubset(dataset, val_indices)
            
            return train_dataset, val_dataset
        else:
            train_ratio, val_ratio, _ = split_ratio
            total = sum(split_ratio)
            train_ratio /= total
            val_ratio /= total
            
            train_end = int(len(indices) * train_ratio)
            val_end = train_end + int(len(indices) * val_ratio)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            train_dataset = DatasetSubset(dataset, train_indices)
            val_dataset = DatasetSubset(dataset, val_indices)
            test_dataset = DatasetSubset(dataset, test_indices)
            
            return train_dataset, val_dataset, test_dataset


class DatasetSubset:
    """Subset de un dataset.
    
    Parameters
    ----------
    dataset : ImageDataset
        Dataset original.
    indices : List[int]
        Índices del subset.
    """
    
    def __init__(self, dataset: ImageDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        """Tamaño del subset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        """Obtiene elemento del subset."""
        return self.dataset[self.indices[idx]]


class AugmentedDataset:
    """Dataset con aumentación de datos.
    
    Parameters
    ----------
    dataset : ImageDataset
        Dataset base.
    augmentation_transform : Callable
        Transformaciones de aumentación.
    n_augmentations : int
        Número de aumentaciones por imagen.
    """
    
    def __init__(
        self,
        dataset: ImageDataset,
        augmentation_transform: Callable,
        n_augmentations: int = 1
    ):
        self.dataset = dataset
        self.augmentation_transform = augmentation_transform
        self.n_augmentations = n_augmentations
    
    def __len__(self):
        """Tamaño del dataset aumentado."""
        return len(self.dataset) * (1 + self.n_augmentations)
    
    def __getitem__(self, idx: int):
        """Obtiene elemento aumentado."""
        # Determinar si es original o aumentado
        base_idx = idx // (1 + self.n_augmentations)
        aug_idx = idx % (1 + self.n_augmentations)
        
        image, label = self.dataset[base_idx]
        
        if aug_idx > 0:
            # Aplicar aumentación
            image = self.augmentation_transform(image)
        
        return image, label