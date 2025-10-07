"""
Transformaciones para features.

Define transformaciones comunes y personalizadas para features.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class FeatureTransformation(ABC):
    """Clase base para transformaciones de features."""
    
    def __init__(self, name: str):
        """Inicializa la transformación.
        
        Parameters
        ----------
        name : str
            Nombre de la transformación.
        """
        self.name = name
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la transformación.
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos de entrada.
            
        Returns
        -------
        pd.Series
            Feature transformada.
        """
        pass
    
    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Obtiene las features requeridas.
        
        Returns
        -------
        List[str]
            Lista de features requeridas.
        """
        pass


@dataclass
class AggregationTransform(FeatureTransformation):
    """Transformación de agregación.
    
    Aplica una función de agregación sobre múltiples features.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    features : List[str]
        Features a agregar.
    method : str
        Método de agregación (sum, mean, max, min, std, count).
    """
    
    def __init__(self, name: str, features: List[str], method: str = "mean"):
        super().__init__(name)
        self.features = features
        self.method = method
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la agregación."""
        if self.method == "sum":
            return data[self.features].sum(axis=1)
        elif self.method == "mean":
            return data[self.features].mean(axis=1)
        elif self.method == "max":
            return data[self.features].max(axis=1)
        elif self.method == "min":
            return data[self.features].min(axis=1)
        elif self.method == "std":
            return data[self.features].std(axis=1)
        elif self.method == "count":
            return data[self.features].notna().sum(axis=1)
        elif self.method == "median":
            return data[self.features].median(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return self.features


@dataclass
class WindowTransform(FeatureTransformation):
    """Transformación de ventana temporal.
    
    Calcula estadísticas sobre ventanas de tiempo.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    feature : str
        Feature a transformar.
    window_size : int
        Tamaño de la ventana.
    method : str
        Método de agregación en la ventana.
    timestamp_col : str
        Columna de timestamp.
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        window_size: int,
        method: str = "mean",
        timestamp_col: str = "_timestamp"
    ):
        super().__init__(name)
        self.feature = feature
        self.window_size = window_size
        self.method = method
        self.timestamp_col = timestamp_col
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la transformación de ventana."""
        # Ordenar por timestamp
        if self.timestamp_col in data.columns:
            data = data.sort_values(self.timestamp_col)
        
        # Aplicar función de ventana
        if self.method == "mean":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).mean()
        elif self.method == "sum":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).sum()
        elif self.method == "max":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).max()
        elif self.method == "min":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).min()
        elif self.method == "std":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).std()
        elif self.method == "count":
            return data[self.feature].rolling(window=self.window_size, min_periods=1).count()
        else:
            raise ValueError(f"Unknown window method: {self.method}")
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        features = [self.feature]
        if self.timestamp_col != "_timestamp":
            features.append(self.timestamp_col)
        return features


@dataclass
class CustomTransform(FeatureTransformation):
    """Transformación personalizada.
    
    Permite definir transformaciones custom con funciones.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    features : List[str]
        Features de entrada.
    func : Callable
        Función de transformación.
    """
    
    def __init__(self, name: str, features: List[str], func: Callable):
        super().__init__(name)
        self.features = features
        self.func = func
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la transformación custom."""
        return self.func(data[self.features])
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return self.features


class RatioTransform(FeatureTransformation):
    """Transformación de ratio entre dos features.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    numerator : str
        Feature numerador.
    denominator : str
        Feature denominador.
    epsilon : float
        Valor pequeño para evitar división por cero.
    """
    
    def __init__(
        self,
        name: str,
        numerator: str,
        denominator: str,
        epsilon: float = 1e-10
    ):
        super().__init__(name)
        self.numerator = numerator
        self.denominator = denominator
        self.epsilon = epsilon
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Calcula el ratio."""
        return data[self.numerator] / (data[self.denominator] + self.epsilon)
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.numerator, self.denominator]


class DifferenceTransform(FeatureTransformation):
    """Transformación de diferencia entre features.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    minuend : str
        Feature minuendo.
    subtrahend : str
        Feature sustraendo.
    """
    
    def __init__(self, name: str, minuend: str, subtrahend: str):
        super().__init__(name)
        self.minuend = minuend
        self.subtrahend = subtrahend
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la diferencia."""
        return data[self.minuend] - data[self.subtrahend]
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.minuend, self.subtrahend]


class LagTransform(FeatureTransformation):
    """Transformación de lag temporal.
    
    Crea features con valores anteriores.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    feature : str
        Feature a retrasar.
    lag : int
        Número de períodos a retrasar.
    """
    
    def __init__(self, name: str, feature: str, lag: int):
        super().__init__(name)
        self.feature = feature
        self.lag = lag
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica el lag."""
        return data[self.feature].shift(self.lag)
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.feature]


class BinningTransform(FeatureTransformation):
    """Transformación de binning/discretización.
    
    Convierte features continuas en categorías.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    feature : str
        Feature a discretizar.
    bins : Union[int, List[float]]
        Número de bins o límites específicos.
    labels : Optional[List[str]]
        Etiquetas para los bins.
    """
    
    def __init__(
        self,
        name: str,
        feature: str,
        bins: Union[int, List[float]],
        labels: Optional[List[str]] = None
    ):
        super().__init__(name)
        self.feature = feature
        self.bins = bins
        self.labels = labels
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica el binning."""
        return pd.cut(data[self.feature], bins=self.bins, labels=self.labels)
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.feature]


class NormalizeTransform(FeatureTransformation):
    """Transformación de normalización.
    
    Normaliza features a un rango específico.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    feature : str
        Feature a normalizar.
    method : str
        Método de normalización (minmax, zscore, robust).
    """
    
    def __init__(self, name: str, feature: str, method: str = "zscore"):
        super().__init__(name)
        self.feature = feature
        self.method = method
        self.params = {}
    
    def fit(self, data: pd.DataFrame):
        """Ajusta los parámetros de normalización.
        
        Parameters
        ----------
        data : pd.DataFrame
            Datos para ajustar.
        """
        if self.method == "minmax":
            self.params["min"] = data[self.feature].min()
            self.params["max"] = data[self.feature].max()
        elif self.method == "zscore":
            self.params["mean"] = data[self.feature].mean()
            self.params["std"] = data[self.feature].std()
        elif self.method == "robust":
            self.params["median"] = data[self.feature].median()
            self.params["mad"] = (data[self.feature] - data[self.feature].median()).abs().median()
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la normalización."""
        if not self.params:
            self.fit(data)
        
        if self.method == "minmax":
            min_val = self.params["min"]
            max_val = self.params["max"]
            return (data[self.feature] - min_val) / (max_val - min_val + 1e-10)
        elif self.method == "zscore":
            mean = self.params["mean"]
            std = self.params["std"]
            return (data[self.feature] - mean) / (std + 1e-10)
        elif self.method == "robust":
            median = self.params["median"]
            mad = self.params["mad"]
            return (data[self.feature] - median) / (mad + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.feature]


class InteractionTransform(FeatureTransformation):
    """Transformación de interacción entre features.
    
    Crea features de interacción multiplicativa.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    features : List[str]
        Features a multiplicar.
    """
    
    def __init__(self, name: str, features: List[str]):
        super().__init__(name)
        self.features = features
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Calcula la interacción."""
        result = data[self.features[0]].copy()
        for feature in self.features[1:]:
            result *= data[feature]
        return result
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return self.features


class PolynomialTransform(FeatureTransformation):
    """Transformación polinomial.
    
    Crea features polinomiales de una feature.
    
    Attributes
    ----------
    name : str
        Nombre de la transformación.
    feature : str
        Feature base.
    degree : int
        Grado del polinomio.
    """
    
    def __init__(self, name: str, feature: str, degree: int):
        super().__init__(name)
        self.feature = feature
        self.degree = degree
    
    def transform(self, data: pd.DataFrame) -> pd.Series:
        """Aplica la transformación polinomial."""
        return data[self.feature] ** self.degree
    
    def get_required_features(self) -> List[str]:
        """Obtiene features requeridas."""
        return [self.feature]