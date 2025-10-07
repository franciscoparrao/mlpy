"""
Transformaciones de imágenes para Computer Vision.
"""

from typing import Optional, List, Tuple, Union, Dict, Any, Callable
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageTransform:
    """Clase base para transformaciones de imágenes."""
    
    def __call__(self, image):
        """Aplica la transformación.
        
        Parameters
        ----------
        image : Any
            Imagen a transformar (PIL, numpy, tensor).
            
        Returns
        -------
        Any
            Imagen transformada.
        """
        return self.transform(image)
    
    def transform(self, image):
        """Implementación de la transformación."""
        raise NotImplementedError


class Compose(ImageTransform):
    """Compone múltiples transformaciones.
    
    Parameters
    ----------
    transforms : List[ImageTransform]
        Lista de transformaciones a aplicar.
    """
    
    def __init__(self, transforms: List[ImageTransform]):
        self.transforms = transforms
    
    def transform(self, image):
        """Aplica todas las transformaciones en secuencia."""
        for t in self.transforms:
            image = t(image)
        return image


class Resize(ImageTransform):
    """Redimensiona imagen.
    
    Parameters
    ----------
    size : Union[int, Tuple[int, int]]
        Tamaño objetivo (height, width) o tamaño mínimo.
    interpolation : str
        Método de interpolación.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = 'bilinear'
    ):
        self.size = size
        self.interpolation = interpolation
        self._setup()
    
    def _setup(self):
        """Configura el transform de torchvision."""
        try:
            from torchvision import transforms as T
            
            if isinstance(self.size, int):
                self._transform = T.Resize(self.size)
            else:
                self._transform = T.Resize(self.size)
        except ImportError:
            logger.warning("torchvision not installed, using fallback resize")
            self._transform = None
    
    def transform(self, image):
        """Aplica redimensionado."""
        if self._transform:
            return self._transform(image)
        
        # Fallback con PIL
        try:
            from PIL import Image
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if isinstance(self.size, int):
                # Mantener aspect ratio
                w, h = image.size
                if h < w:
                    new_h = self.size
                    new_w = int(w * self.size / h)
                else:
                    new_w = self.size
                    new_h = int(h * self.size / w)
                return image.resize((new_w, new_h))
            else:
                return image.resize(self.size[::-1])  # PIL usa (width, height)
        except ImportError:
            raise ImportError("PIL or torchvision required for image resizing")


class CenterCrop(ImageTransform):
    """Recorte central de imagen.
    
    Parameters
    ----------
    size : Union[int, Tuple[int, int]]
        Tamaño del recorte.
    """
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        self.size = size
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.CenterCrop(self.size)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica recorte central."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        try:
            from PIL import Image
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            w, h = image.size
            if isinstance(self.size, int):
                crop_h = crop_w = self.size
            else:
                crop_h, crop_w = self.size
            
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            right = left + crop_w
            bottom = top + crop_h
            
            return image.crop((left, top, right, bottom))
        except ImportError:
            raise ImportError("PIL or torchvision required")


class RandomCrop(ImageTransform):
    """Recorte aleatorio de imagen.
    
    Parameters
    ----------
    size : Union[int, Tuple[int, int]]
        Tamaño del recorte.
    padding : Optional[int]
        Padding antes del recorte.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[int] = None
    ):
        self.size = size
        self.padding = padding
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.RandomCrop(self.size, padding=self.padding)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica recorte aleatorio."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        import random
        try:
            from PIL import Image
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            w, h = image.size
            if isinstance(self.size, int):
                crop_h = crop_w = self.size
            else:
                crop_h, crop_w = self.size
            
            if self.padding:
                # Añadir padding
                new_w = w + 2 * self.padding
                new_h = h + 2 * self.padding
                padded = Image.new(image.mode, (new_w, new_h))
                padded.paste(image, (self.padding, self.padding))
                image = padded
                w, h = new_w, new_h
            
            # Posición aleatoria
            left = random.randint(0, max(0, w - crop_w))
            top = random.randint(0, max(0, h - crop_h))
            right = left + crop_w
            bottom = top + crop_h
            
            return image.crop((left, top, right, bottom))
        except ImportError:
            raise ImportError("PIL or torchvision required")


class RandomHorizontalFlip(ImageTransform):
    """Volteo horizontal aleatorio.
    
    Parameters
    ----------
    p : float
        Probabilidad de volteo.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.RandomHorizontalFlip(self.p)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica volteo horizontal aleatorio."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        import random
        if random.random() < self.p:
            try:
                from PIL import Image, ImageOps
                
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                return ImageOps.mirror(image)
            except ImportError:
                # Numpy fallback
                if isinstance(image, np.ndarray):
                    return np.fliplr(image)
        
        return image


class RandomVerticalFlip(ImageTransform):
    """Volteo vertical aleatorio.
    
    Parameters
    ----------
    p : float
        Probabilidad de volteo.
    """
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.RandomVerticalFlip(self.p)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica volteo vertical aleatorio."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        import random
        if random.random() < self.p:
            try:
                from PIL import Image, ImageOps
                
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                return ImageOps.flip(image)
            except ImportError:
                # Numpy fallback
                if isinstance(image, np.ndarray):
                    return np.flipud(image)
        
        return image


class RandomRotation(ImageTransform):
    """Rotación aleatoria.
    
    Parameters
    ----------
    degrees : Union[float, Tuple[float, float]]
        Rango de grados de rotación.
    interpolation : str
        Método de interpolación.
    fill : Optional[Union[int, Tuple[int, int, int]]]
        Color de relleno.
    """
    
    def __init__(
        self,
        degrees: Union[float, Tuple[float, float]],
        interpolation: str = 'bilinear',
        fill: Optional[Union[int, Tuple[int, int, int]]] = None
    ):
        self.degrees = degrees
        self.interpolation = interpolation
        self.fill = fill
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.RandomRotation(
                self.degrees,
                fill=self.fill
            )
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica rotación aleatoria."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        import random
        try:
            from PIL import Image
            
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if isinstance(self.degrees, (int, float)):
                angle = random.uniform(-self.degrees, self.degrees)
            else:
                angle = random.uniform(self.degrees[0], self.degrees[1])
            
            return image.rotate(angle, fillcolor=self.fill)
        except ImportError:
            raise ImportError("PIL or torchvision required")


class ColorJitter(ImageTransform):
    """Ajuste aleatorio de color.
    
    Parameters
    ----------
    brightness : float
        Factor de brillo.
    contrast : float
        Factor de contraste.
    saturation : float
        Factor de saturación.
    hue : float
        Factor de tono.
    """
    
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica ajuste de color."""
        if self._transform:
            return self._transform(image)
        
        logger.warning("ColorJitter requires torchvision")
        return image


class Normalize(ImageTransform):
    """Normaliza tensor de imagen.
    
    Parameters
    ----------
    mean : Union[float, List[float]]
        Media para normalización.
    std : Union[float, List[float]]
        Desviación estándar.
    """
    
    def __init__(
        self,
        mean: Union[float, List[float]],
        std: Union[float, List[float]]
    ):
        self.mean = mean
        self.std = std
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.Normalize(mean=self.mean, std=self.std)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Aplica normalización."""
        if self._transform:
            return self._transform(image)
        
        # Fallback numpy
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            if isinstance(self.mean, (list, tuple)):
                for c in range(min(image.shape[-1], len(self.mean))):
                    image[..., c] = (image[..., c] - self.mean[c]) / self.std[c]
            else:
                image = (image - self.mean) / self.std
            return image
        
        return image


class ToTensor(ImageTransform):
    """Convierte imagen a tensor."""
    
    def __init__(self):
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.ToTensor()
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Convierte a tensor."""
        if self._transform:
            return self._transform(image)
        
        # Fallback con torch
        try:
            import torch
            
            if isinstance(image, np.ndarray):
                # HWC -> CHW
                if len(image.shape) == 3:
                    image = image.transpose(2, 0, 1)
                # Normalizar a [0, 1]
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                return torch.from_numpy(image)
            
            # PIL Image
            from PIL import Image
            if isinstance(image, Image.Image):
                image = np.array(image)
                return self.transform(image)
        except ImportError:
            logger.warning("torch or torchvision required for ToTensor")
        
        return image


class ToPILImage(ImageTransform):
    """Convierte tensor a imagen PIL."""
    
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode
        self._setup()
    
    def _setup(self):
        """Configura el transform."""
        try:
            from torchvision import transforms as T
            self._transform = T.ToPILImage(mode=self.mode)
        except ImportError:
            self._transform = None
    
    def transform(self, image):
        """Convierte a PIL Image."""
        if self._transform:
            return self._transform(image)
        
        # Fallback manual
        try:
            from PIL import Image
            import torch
            
            if torch.is_tensor(image):
                # Convertir a numpy
                image = image.cpu().numpy()
                
                # CHW -> HWC
                if len(image.shape) == 3 and image.shape[0] in [1, 3, 4]:
                    image = image.transpose(1, 2, 0)
                
                # Desnormalizar si está en [0, 1]
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            
            if isinstance(image, np.ndarray):
                return Image.fromarray(image, mode=self.mode)
        except ImportError:
            logger.warning("PIL required for ToPILImage")
        
        return image


def create_augmentation_pipeline(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_level: str = 'medium',
    normalize: bool = True,
    to_tensor: bool = True
) -> Compose:
    """Crea pipeline de aumentación de datos.
    
    Parameters
    ----------
    image_size : Tuple[int, int]
        Tamaño de imagen objetivo.
    augmentation_level : str
        Nivel de aumentación ('none', 'light', 'medium', 'heavy').
    normalize : bool
        Si normalizar con ImageNet stats.
    to_tensor : bool
        Si convertir a tensor.
        
    Returns
    -------
    Compose
        Pipeline de transformaciones.
    """
    transforms = []
    
    # Redimensionar
    transforms.append(Resize(image_size))
    
    # Aumentación según nivel
    if augmentation_level == 'light':
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomCrop(image_size, padding=4)
        ])
    elif augmentation_level == 'medium':
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomCrop(image_size, padding=8),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2)
        ])
    elif augmentation_level == 'heavy':
        transforms.extend([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.3),
            RandomCrop(image_size, padding=16),
            RandomRotation(degrees=30),
            ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        ])
    
    # Convertir a tensor
    if to_tensor:
        transforms.append(ToTensor())
    
    # Normalizar con ImageNet stats
    if normalize:
        transforms.append(
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    return Compose(transforms)