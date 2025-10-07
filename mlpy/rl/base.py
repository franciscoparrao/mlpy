"""
Clases base para Reinforcement Learning en MLPY.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from dataclasses import dataclass, field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuración para agentes de RL.
    
    Attributes
    ----------
    learning_rate : float
        Tasa de aprendizaje.
    gamma : float
        Factor de descuento.
    batch_size : int
        Tamaño del batch.
    n_steps : int
        Número de pasos para actualización.
    verbose : int
        Nivel de verbosidad.
    tensorboard_log : Optional[str]
        Directorio para logs de TensorBoard.
    seed : Optional[int]
        Semilla aleatoria.
    device : str
        Dispositivo ('cuda', 'cpu', 'auto').
    """
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    n_steps: int = 2048
    verbose: int = 0
    tensorboard_log: Optional[str] = None
    seed: Optional[int] = None
    device: str = 'auto'


class RLEnvironment(ABC):
    """Clase base para entornos de RL."""
    
    def __init__(self, env_id: str):
        """Inicializa el entorno.
        
        Parameters
        ----------
        env_id : str
            ID del entorno.
        """
        self.env_id = env_id
        self._env = None
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea el entorno.
        
        Parameters
        ----------
        seed : Optional[int]
            Semilla aleatoria.
            
        Returns
        -------
        Tuple[np.ndarray, Dict]
            Observación inicial e info.
        """
        pass
    
    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta una acción en el entorno.
        
        Parameters
        ----------
        action : Union[int, np.ndarray]
            Acción a ejecutar.
            
        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict]
            observation, reward, terminated, truncated, info
        """
        pass
    
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno.
        
        Returns
        -------
        Optional[np.ndarray]
            Frame renderizado o None.
        """
        pass
    
    @abstractmethod
    def close(self):
        """Cierra el entorno."""
        pass
    
    @property
    @abstractmethod
    def observation_space(self):
        """Espacio de observaciones."""
        pass
    
    @property
    @abstractmethod
    def action_space(self):
        """Espacio de acciones."""
        pass
    
    @property
    def is_discrete(self) -> bool:
        """Verifica si el espacio de acciones es discreto."""
        return hasattr(self.action_space, 'n')
    
    @property
    def n_actions(self) -> Optional[int]:
        """Número de acciones (si es discreto)."""
        if self.is_discrete:
            return self.action_space.n
        return None
    
    @property
    def action_dim(self) -> Optional[int]:
        """Dimensión del espacio de acciones (si es continuo)."""
        if not self.is_discrete:
            return self.action_space.shape[0]
        return None


class RLAgent(ABC):
    """Clase base para agentes de RL."""
    
    def __init__(self, env: RLEnvironment, config: Optional[RLConfig] = None):
        """Inicializa el agente.
        
        Parameters
        ----------
        env : RLEnvironment
            Entorno de RL.
        config : Optional[RLConfig]
            Configuración del agente.
        """
        self.env = env
        self.config = config or RLConfig()
        self.model = None
        self.is_trained = False
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Configura el agente."""
        pass
    
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        callback: Optional['RLCallback'] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Entrena el agente.
        
        Parameters
        ----------
        total_timesteps : int
            Número total de timesteps.
        callback : Optional[RLCallback]
            Callback para el entrenamiento.
        **kwargs
            Argumentos adicionales.
            
        Returns
        -------
        Dict[str, Any]
            Resultados del entrenamiento.
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[Union[int, np.ndarray], Optional[np.ndarray]]:
        """Predice una acción.
        
        Parameters
        ----------
        observation : np.ndarray
            Observación actual.
        deterministic : bool
            Si usar política determinística.
            
        Returns
        -------
        Tuple[Union[int, np.ndarray], Optional[np.ndarray]]
            Acción y estado (si aplica).
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Guarda el modelo.
        
        Parameters
        ----------
        path : Union[str, Path]
            Ruta donde guardar.
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]):
        """Carga el modelo.
        
        Parameters
        ----------
        path : Union[str, Path]
            Ruta del modelo.
        """
        pass
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """Evalúa el agente.
        
        Parameters
        ----------
        n_episodes : int
            Número de episodios.
        deterministic : bool
            Si usar política determinística.
        render : bool
            Si renderizar.
            
        Returns
        -------
        Dict[str, float]
            Métricas de evaluación.
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Agent must be trained before evaluation")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    self.env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def play(
        self,
        n_episodes: int = 1,
        deterministic: bool = True,
        render: bool = True,
        fps: int = 30
    ) -> List[float]:
        """Juega episodios con el agente.
        
        Parameters
        ----------
        n_episodes : int
            Número de episodios.
        deterministic : bool
            Si usar política determinística.
        render : bool
            Si renderizar.
        fps : int
            Frames por segundo para renderizado.
            
        Returns
        -------
        List[float]
            Recompensas por episodio.
        """
        import time
        
        episode_rewards = []
        frame_time = 1.0 / fps if render else 0
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            logger.info(f"Episode {episode + 1}/{n_episodes}")
            
            while not done:
                if render:
                    self.env.render()
                    time.sleep(frame_time)
                
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            logger.info(f"Episode reward: {episode_reward:.2f}")
        
        return episode_rewards


class RLCallback(ABC):
    """Clase base para callbacks de RL."""
    
    def __init__(self):
        """Inicializa el callback."""
        self.model = None
        self.num_timesteps = 0
        self.locals = {}
        self.globals = {}
    
    def init_callback(self, model: RLAgent) -> None:
        """Inicializa el callback con el modelo.
        
        Parameters
        ----------
        model : RLAgent
            Agente de RL.
        """
        self.model = model
    
    def on_training_start(self) -> None:
        """Llamado al inicio del entrenamiento."""
        pass
    
    def on_training_end(self) -> None:
        """Llamado al final del entrenamiento."""
        pass
    
    def on_rollout_start(self) -> None:
        """Llamado al inicio de un rollout."""
        pass
    
    def on_rollout_end(self) -> None:
        """Llamado al final de un rollout."""
        pass
    
    @abstractmethod
    def on_step(self) -> bool:
        """Llamado en cada step.
        
        Returns
        -------
        bool
            True para continuar, False para detener.
        """
        return True
    
    def update_locals(self, locals_: Dict[str, Any]) -> None:
        """Actualiza variables locales.
        
        Parameters
        ----------
        locals_ : Dict[str, Any]
            Variables locales.
        """
        self.locals.update(locals_)
        self.num_timesteps = self.locals.get("num_timesteps", 0)


class EpisodeLogger(RLCallback):
    """Callback para logging de episodios."""
    
    def __init__(self, log_frequency: int = 100):
        """Inicializa el logger.
        
        Parameters
        ----------
        log_frequency : int
            Frecuencia de logging en episodios.
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
    
    def on_step(self) -> bool:
        """Registra información del step."""
        # Obtener info del step
        done = self.locals.get("done", False)
        reward = self.locals.get("reward", 0)
        
        self.current_reward += reward
        self.current_length += 1
        
        if done:
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            self.episode_count += 1
            
            if self.episode_count % self.log_frequency == 0:
                mean_reward = np.mean(self.episode_rewards[-self.log_frequency:])
                mean_length = np.mean(self.episode_lengths[-self.log_frequency:])
                
                logger.info(
                    f"Episode {self.episode_count} | "
                    f"Mean Reward: {mean_reward:.2f} | "
                    f"Mean Length: {mean_length:.1f}"
                )
            
            self.current_reward = 0
            self.current_length = 0
        
        return True


class CheckpointCallback(RLCallback):
    """Callback para guardar checkpoints."""
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./rl_checkpoints",
        name_prefix: str = "rl_model"
    ):
        """Inicializa el callback.
        
        Parameters
        ----------
        save_freq : int
            Frecuencia de guardado en timesteps.
        save_path : str
            Directorio para checkpoints.
        name_prefix : str
            Prefijo para nombres de archivo.
        """
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def on_step(self) -> bool:
        """Guarda checkpoint si corresponde."""
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(path)
            logger.info(f"Saved checkpoint at {path}")
        
        return True