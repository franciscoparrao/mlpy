"""
Modelos de redes neuronales para PyTorch en MLPY.

Arquitecturas predefinidas y wrappers para modelos comunes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class MLPNet(nn.Module):
    """Red neuronal multicapa (MLP).
    
    Parameters
    ----------
    input_dim : int
        Dimensión de entrada.
    hidden_dims : List[int]
        Dimensiones de capas ocultas.
    output_dim : int
        Dimensión de salida.
    activation : str
        Función de activación ('relu', 'tanh', 'sigmoid', 'elu', 'leaky_relu').
    dropout : float
        Tasa de dropout.
    batch_norm : bool
        Si usar batch normalization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        super().__init__()
        
        # Seleccionar función de activación
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(0.2)
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.activation_fn = activations[activation]
        
        # Construir capas
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Capa lineal
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activación
            layers.append(self.activation_fn)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Capa de salida
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.layers(x)


class CNNClassifier(nn.Module):
    """Red convolucional para clasificación.
    
    Parameters
    ----------
    input_channels : int
        Número de canales de entrada.
    num_classes : int
        Número de clases.
    conv_channels : List[int]
        Canales de las capas convolucionales.
    kernel_sizes : List[int]
        Tamaños de kernel.
    pool_sizes : List[int]
        Tamaños de pooling.
    fc_dims : List[int]
        Dimensiones de capas fully connected.
    dropout : float
        Tasa de dropout.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        conv_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        pool_sizes: Optional[List[int]] = None,
        fc_dims: Optional[List[int]] = None,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Valores por defecto
        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3] * len(conv_channels)
        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)
        if fc_dims is None:
            fc_dims = [256, 128]
        
        # Capas convolucionales
        conv_layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel, pool) in enumerate(zip(conv_channels, kernel_sizes, pool_sizes)):
            # Conv block
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            
            # Pooling
            if pool > 1:
                conv_layers.append(nn.MaxPool2d(pool))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculamos el tamaño después de las convoluciones
        # Asumiendo entrada de 224x224 (ajustar según necesidad)
        self.flatten_size = self._calculate_flatten_size(input_channels)
        
        # Capas fully connected
        fc_layers = []
        prev_dim = self.flatten_size
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = fc_dim
        
        fc_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def _calculate_flatten_size(self, input_channels: int, img_size: int = 224) -> int:
        """Calcula el tamaño después de flatten."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_size, img_size)
            output = self.conv_layers(dummy_input)
            return output.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, channels, height, width].
            
        Returns
        -------
        torch.Tensor
            Output tensor [batch, num_classes].
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class ResNetWrapper(nn.Module):
    """Wrapper para modelos ResNet pre-entrenados.
    
    Parameters
    ----------
    num_classes : int
        Número de clases.
    pretrained : bool
        Si usar pesos pre-entrenados.
    model_name : str
        Nombre del modelo ('resnet18', 'resnet34', 'resnet50', etc.).
    freeze_backbone : bool
        Si congelar el backbone.
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        model_name: str = 'resnet18',
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        try:
            import torchvision.models as models
        except ImportError:
            raise ImportError("torchvision required. Install with: pip install torchvision")
        
        # Cargar modelo
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if model_name not in model_dict:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = model_dict[model_name](pretrained=pretrained)
        
        # Congelar backbone si se solicita
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Reemplazar capa final
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        logger.info(f"Loaded {model_name} with {num_classes} output classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self.model(x)


class TransformerModel(nn.Module):
    """Modelo Transformer para secuencias.
    
    Parameters
    ----------
    input_dim : int
        Dimensión de entrada.
    d_model : int
        Dimensión del modelo.
    nhead : int
        Número de attention heads.
    num_layers : int
        Número de capas transformer.
    dim_feedforward : int
        Dimensión de feedforward.
    dropout : float
        Tasa de dropout.
    max_seq_length : int
        Longitud máxima de secuencia.
    num_classes : Optional[int]
        Número de clases (para clasificación).
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 1000,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        if num_classes is not None:
            self.classifier = nn.Linear(d_model, num_classes)
        else:
            self.classifier = None
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, seq_len, input_dim].
        mask : Optional[torch.Tensor]
            Attention mask.
            
        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Classification head
        if self.classifier is not None:
            # Pool over sequence dimension
            x = x.mean(dim=1)  # Global average pooling
            x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding para Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class AutoEncoder(nn.Module):
    """Autoencoder para reducción de dimensionalidad.
    
    Parameters
    ----------
    input_dim : int
        Dimensión de entrada.
    encoding_dims : List[int]
        Dimensiones del encoder.
    latent_dim : int
        Dimensión del espacio latente.
    activation : str
        Función de activación.
    tied_weights : bool
        Si compartir pesos entre encoder y decoder.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int],
        latent_dim: int,
        activation: str = 'relu',
        tied_weights: bool = False
    ):
        super().__init__()
        
        self.tied_weights = tied_weights
        
        # Activación
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU()
        }
        
        activation_fn = activations.get(activation, nn.ReLU())
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(activation_fn)
            prev_dim = dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        if tied_weights:
            # Compartir pesos (transpuestos)
            self.decoder = self._create_tied_decoder()
        else:
            # Decoder independiente
            decoder_layers = []
            decoder_dims = [latent_dim] + encoding_dims[::-1] + [input_dim]
            
            for i in range(len(decoder_dims) - 1):
                decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
                if i < len(decoder_dims) - 2:  # No activación en última capa
                    decoder_layers.append(activation_fn)
            
            self.decoder = nn.Sequential(*decoder_layers)
    
    def _create_tied_decoder(self):
        """Crea decoder con pesos compartidos."""
        # Implementación simplificada - en práctica necesitaría más trabajo
        logger.warning("Tied weights not fully implemented, using regular decoder")
        return self.decoder
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Codifica la entrada.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        torch.Tensor
            Representación latente.
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica desde el espacio latente.
        
        Parameters
        ----------
        z : torch.Tensor
            Representación latente.
            
        Returns
        -------
        torch.Tensor
            Reconstrucción.
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass completo.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (reconstrucción, representación latente).
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class LSTMClassifier(nn.Module):
    """LSTM para clasificación de secuencias.
    
    Parameters
    ----------
    input_dim : int
        Dimensión de entrada.
    hidden_dim : int
        Dimensión oculta del LSTM.
    num_layers : int
        Número de capas LSTM.
    num_classes : int
        Número de clases.
    dropout : float
        Tasa de dropout.
    bidirectional : bool
        Si usar LSTM bidireccional.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, seq_len, input_dim].
        lengths : Optional[torch.Tensor]
            Longitudes reales de las secuencias.
            
        Returns
        -------
        torch.Tensor
            Output tensor [batch, num_classes].
        """
        # Pack sequences si se proporcionan longitudes
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Unpack si fue packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        
        # Usar última salida del LSTM
        if self.bidirectional:
            # Concatenar últimas salidas forward y backward
            last_forward = hidden[-2]
            last_backward = hidden[-1]
            last_hidden = torch.cat([last_forward, last_backward], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Clasificación
        output = self.classifier(last_hidden)
        
        return output