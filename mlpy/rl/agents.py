"""
Agentes de Reinforcement Learning usando Stable Baselines3.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
from pathlib import Path
import logging

from .base import RLAgent, RLEnvironment, RLConfig, RLCallback

logger = logging.getLogger(__name__)


class StableBaselinesAgent(RLAgent):
    """Agente base usando Stable Baselines3.
    
    Wrapper para algoritmos de SB3.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        algorithm_class: Any,
        config: Optional[RLConfig] = None,
        policy: str = 'MlpPolicy',
        **kwargs
    ):
        """Inicializa el agente SB3.
        
        Parameters
        ----------
        env : RLEnvironment
            Entorno de RL.
        algorithm_class : Any
            Clase del algoritmo SB3.
        config : Optional[RLConfig]
            Configuración del agente.
        policy : str
            Tipo de política ('MlpPolicy', 'CnnPolicy', etc.).
        **kwargs
            Argumentos adicionales para el algoritmo.
        """
        self.algorithm_class = algorithm_class
        self.policy = policy
        self.kwargs = kwargs
        super().__init__(env, config)
    
    def _setup(self):
        """Configura el agente SB3."""
        try:
            # Preparar argumentos del algoritmo
            algo_kwargs = {
                'learning_rate': self.config.learning_rate,
                'gamma': self.config.gamma,
                'batch_size': self.config.batch_size,
                'verbose': self.config.verbose,
                'tensorboard_log': self.config.tensorboard_log,
                'seed': self.config.seed,
                'device': self.config.device
            }
            
            # Añadir kwargs específicos
            algo_kwargs.update(self.kwargs)
            
            # Filtrar parámetros no soportados
            import inspect
            valid_params = inspect.signature(self.algorithm_class.__init__).parameters
            algo_kwargs = {k: v for k, v in algo_kwargs.items() 
                          if k in valid_params and v is not None}
            
            # Crear modelo
            self.model = self.algorithm_class(
                self.policy,
                self.env._env if hasattr(self.env, '_env') else self.env,
                **algo_kwargs
            )
            
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed. "
                "Install with: pip install stable-baselines3"
            )
    
    def train(
        self,
        total_timesteps: int,
        callback: Optional[RLCallback] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Entrena el agente."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Convertir callback de MLPY a SB3
        sb3_callback = None
        if callback:
            sb3_callback = self._convert_callback(callback)
        
        # Entrenar
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=sb3_callback,
            **kwargs
        )
        
        self.is_trained = True
        
        # Obtener estadísticas de entrenamiento
        stats = {
            'total_timesteps': total_timesteps,
            'n_updates': self.model.num_timesteps
        }
        
        return stats
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[Union[int, np.ndarray], Optional[np.ndarray]]:
        """Predice una acción."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        action, states = self.model.predict(
            observation,
            deterministic=deterministic
        )
        
        return action, states
    
    def save(self, path: Union[str, Path]):
        """Guarda el modelo."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Union[str, Path]):
        """Carga el modelo."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = self.algorithm_class.load(
            str(path),
            env=self.env._env if hasattr(self.env, '_env') else self.env
        )
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    def _convert_callback(self, callback: RLCallback):
        """Convierte callback de MLPY a SB3."""
        from stable_baselines3.common.callbacks import BaseCallback
        
        class SB3CallbackWrapper(BaseCallback):
            def __init__(self, mlpy_callback: RLCallback):
                super().__init__()
                self.mlpy_callback = mlpy_callback
            
            def _on_training_start(self) -> None:
                self.mlpy_callback.on_training_start()
            
            def _on_training_end(self) -> None:
                self.mlpy_callback.on_training_end()
            
            def _on_rollout_start(self) -> None:
                self.mlpy_callback.on_rollout_start()
            
            def _on_rollout_end(self) -> None:
                self.mlpy_callback.on_rollout_end()
            
            def _on_step(self) -> bool:
                self.mlpy_callback.update_locals(self.locals)
                return self.mlpy_callback.on_step()
        
        return SB3CallbackWrapper(callback)


class PPOAgent(StableBaselinesAgent):
    """Agente Proximal Policy Optimization (PPO).
    
    PPO es un algoritmo on-policy que funciona bien en una variedad de tareas.
    Adecuado para espacios de acción discretos y continuos.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno de RL.
    config : Optional[RLConfig]
        Configuración del agente.
    n_steps : int
        Número de pasos para actualización.
    n_epochs : int
        Número de épocas de optimización.
    clip_range : float
        Rango de clipping para PPO.
    ent_coef : float
        Coeficiente de entropía.
    vf_coef : float
        Coeficiente de value function.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        config: Optional[RLConfig] = None,
        n_steps: int = 2048,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        **kwargs
    ):
        try:
            from stable_baselines3 import PPO
            algorithm_class = PPO
        except ImportError:
            raise ImportError("stable-baselines3 required for PPO")
        
        # Configurar n_steps en config si está presente
        if config:
            config.n_steps = n_steps
        
        super().__init__(
            env=env,
            algorithm_class=algorithm_class,
            config=config,
            n_steps=n_steps,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            **kwargs
        )


class DQNAgent(StableBaselinesAgent):
    """Agente Deep Q-Network (DQN).
    
    DQN es un algoritmo off-policy para espacios de acción discretos.
    Utiliza experience replay y target network.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno de RL (debe tener acciones discretas).
    config : Optional[RLConfig]
        Configuración del agente.
    buffer_size : int
        Tamaño del replay buffer.
    learning_starts : int
        Pasos antes de empezar a aprender.
    tau : float
        Soft update coefficient.
    target_update_interval : int
        Intervalo de actualización del target network.
    exploration_fraction : float
        Fracción del entrenamiento para exploración.
    exploration_initial_eps : float
        Epsilon inicial.
    exploration_final_eps : float
        Epsilon final.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        config: Optional[RLConfig] = None,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        tau: float = 1.0,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        **kwargs
    ):
        # Verificar que el entorno tiene acciones discretas
        if not env.is_discrete:
            raise ValueError("DQN requires discrete action space")
        
        try:
            from stable_baselines3 import DQN
            algorithm_class = DQN
        except ImportError:
            raise ImportError("stable-baselines3 required for DQN")
        
        super().__init__(
            env=env,
            algorithm_class=algorithm_class,
            config=config,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            tau=tau,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            **kwargs
        )


class SACAgent(StableBaselinesAgent):
    """Agente Soft Actor-Critic (SAC).
    
    SAC es un algoritmo off-policy para espacios de acción continuos.
    Maximiza entropía además de recompensa.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno de RL (debe tener acciones continuas).
    config : Optional[RLConfig]
        Configuración del agente.
    buffer_size : int
        Tamaño del replay buffer.
    learning_starts : int
        Pasos antes de empezar a aprender.
    tau : float
        Soft update coefficient.
    train_freq : int
        Frecuencia de actualización.
    gradient_steps : int
        Gradient steps por actualización.
    ent_coef : Union[str, float]
        Coeficiente de entropía ('auto' para ajuste automático).
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        config: Optional[RLConfig] = None,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        tau: float = 0.005,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = 'auto',
        **kwargs
    ):
        # Verificar que el entorno tiene acciones continuas
        if env.is_discrete:
            raise ValueError("SAC requires continuous action space")
        
        try:
            from stable_baselines3 import SAC
            algorithm_class = SAC
        except ImportError:
            raise ImportError("stable-baselines3 required for SAC")
        
        super().__init__(
            env=env,
            algorithm_class=algorithm_class,
            config=config,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            tau=tau,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            **kwargs
        )


class A2CAgent(StableBaselinesAgent):
    """Agente Advantage Actor-Critic (A2C).
    
    A2C es la versión síncrona de A3C, un algoritmo on-policy.
    Funciona para espacios discretos y continuos.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno de RL.
    config : Optional[RLConfig]
        Configuración del agente.
    n_steps : int
        Número de pasos para actualización.
    ent_coef : float
        Coeficiente de entropía.
    vf_coef : float
        Coeficiente de value function.
    max_grad_norm : float
        Norma máxima del gradiente.
    use_rms_prop : bool
        Si usar RMSprop en lugar de Adam.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        config: Optional[RLConfig] = None,
        n_steps: int = 5,
        ent_coef: float = 0.01,
        vf_coef: float = 0.25,
        max_grad_norm: float = 0.5,
        use_rms_prop: bool = True,
        **kwargs
    ):
        try:
            from stable_baselines3 import A2C
            algorithm_class = A2C
        except ImportError:
            raise ImportError("stable-baselines3 required for A2C")
        
        # Configurar n_steps en config si está presente
        if config:
            config.n_steps = n_steps
        
        super().__init__(
            env=env,
            algorithm_class=algorithm_class,
            config=config,
            n_steps=n_steps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_rms_prop=use_rms_prop,
            **kwargs
        )


class TD3Agent(StableBaselinesAgent):
    """Agente Twin Delayed DDPG (TD3).
    
    TD3 es un algoritmo off-policy para espacios de acción continuos.
    Mejora DDPG con twin critics y delayed policy updates.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno de RL (debe tener acciones continuas).
    config : Optional[RLConfig]
        Configuración del agente.
    buffer_size : int
        Tamaño del replay buffer.
    learning_starts : int
        Pasos antes de empezar a aprender.
    tau : float
        Soft update coefficient.
    train_freq : Tuple[int, str]
        Frecuencia de entrenamiento.
    gradient_steps : int
        Gradient steps por actualización.
    policy_delay : int
        Delay para actualización de política.
    action_noise : Optional[Any]
        Ruido para exploración.
    """
    
    def __init__(
        self,
        env: RLEnvironment,
        config: Optional[RLConfig] = None,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        tau: float = 0.005,
        train_freq: Tuple[int, str] = (1, "episode"),
        gradient_steps: int = -1,
        policy_delay: int = 2,
        action_noise: Optional[Any] = None,
        **kwargs
    ):
        # Verificar que el entorno tiene acciones continuas
        if env.is_discrete:
            raise ValueError("TD3 requires continuous action space")
        
        try:
            from stable_baselines3 import TD3
            algorithm_class = TD3
        except ImportError:
            raise ImportError("stable-baselines3 required for TD3")
        
        # Crear ruido si no se proporciona
        if action_noise is None and hasattr(env, 'action_space'):
            try:
                from stable_baselines3.common.noise import NormalActionNoise
                n_actions = env.action_dim
                action_noise = NormalActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.1 * np.ones(n_actions)
                )
            except:
                pass
        
        super().__init__(
            env=env,
            algorithm_class=algorithm_class,
            config=config,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            tau=tau,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_delay=policy_delay,
            action_noise=action_noise,
            **kwargs
        )