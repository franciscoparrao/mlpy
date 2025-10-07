"""
Wrappers de entornos para Reinforcement Learning.
"""

from typing import Optional, Dict, Any, List, Union, Tuple, Callable
import numpy as np
import logging
from pathlib import Path

from .base import RLEnvironment

logger = logging.getLogger(__name__)


class GymEnvironment(RLEnvironment):
    """Wrapper para entornos de Gymnasium/OpenAI Gym.
    
    Parameters
    ----------
    env_id : str
        ID del entorno en Gymnasium.
    render_mode : Optional[str]
        Modo de renderizado ('human', 'rgb_array', etc.).
    max_episode_steps : Optional[int]
        Máximo de pasos por episodio.
    """
    
    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        **kwargs
    ):
        super().__init__(env_id)
        
        try:
            import gymnasium as gym
            self.gym = gym
        except ImportError:
            try:
                import gym
                self.gym = gym
                logger.warning("Using older gym version. Consider upgrading to gymnasium.")
            except ImportError:
                raise ImportError("gymnasium or gym not installed. Install with: pip install gymnasium")
        
        # Crear entorno
        self._env = self.gym.make(
            env_id,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps or self._env.spec.max_episode_steps
        self.episode_step = 0
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea el entorno."""
        self.episode_step = 0
        
        if hasattr(self._env, 'reset'):
            # Gymnasium API
            obs, info = self._env.reset(seed=seed)
        else:
            # Old Gym API
            if seed is not None:
                self._env.seed(seed)
            obs = self._env.reset()
            info = {}
        
        return obs, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta una acción."""
        self.episode_step += 1
        
        if hasattr(self._env, 'step'):
            result = self._env.step(action)
            
            if len(result) == 5:
                # Gymnasium API
                obs, reward, terminated, truncated, info = result
            else:
                # Old Gym API
                obs, reward, done, info = result
                terminated = done
                truncated = False
        else:
            raise RuntimeError("Environment doesn't have step method")
        
        # Añadir información del episodio
        info['episode_step'] = self.episode_step
        
        # Verificar truncation por límite de pasos
        if self.max_episode_steps and self.episode_step >= self.max_episode_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        if hasattr(self._env, 'render'):
            return self._env.render()
        return None
    
    def close(self):
        """Cierra el entorno."""
        if self._env:
            self._env.close()
    
    @property
    def observation_space(self):
        """Espacio de observaciones."""
        return self._env.observation_space
    
    @property
    def action_space(self):
        """Espacio de acciones."""
        return self._env.action_space
    
    @property
    def metadata(self) -> Dict:
        """Metadata del entorno."""
        return self._env.metadata if hasattr(self._env, 'metadata') else {}
    
    @property
    def spec(self):
        """Especificación del entorno."""
        return self._env.spec if hasattr(self._env, 'spec') else None


class CustomEnvironment(RLEnvironment):
    """Entorno personalizado para RL.
    
    Permite crear entornos personalizados con funciones Python.
    
    Parameters
    ----------
    env_id : str
        ID del entorno.
    observation_space : Any
        Espacio de observaciones.
    action_space : Any
        Espacio de acciones.
    reset_fn : Callable
        Función de reset.
    step_fn : Callable
        Función de step.
    render_fn : Optional[Callable]
        Función de render.
    """
    
    def __init__(
        self,
        env_id: str,
        observation_space,
        action_space,
        reset_fn: Callable,
        step_fn: Callable,
        render_fn: Optional[Callable] = None
    ):
        super().__init__(env_id)
        
        self._observation_space = observation_space
        self._action_space = action_space
        self.reset_fn = reset_fn
        self.step_fn = step_fn
        self.render_fn = render_fn
        
        self.state = None
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea el entorno."""
        if seed is not None:
            np.random.seed(seed)
        
        self.state, info = self.reset_fn()
        return self.state, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta una acción."""
        if self.state is None:
            raise RuntimeError("Environment must be reset before step")
        
        self.state, reward, terminated, truncated, info = self.step_fn(self.state, action)
        return self.state, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        if self.render_fn:
            return self.render_fn(self.state)
        return None
    
    def close(self):
        """Cierra el entorno."""
        pass  # No hay recursos que liberar
    
    @property
    def observation_space(self):
        """Espacio de observaciones."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Espacio de acciones."""
        return self._action_space


class VectorizedEnvironment(RLEnvironment):
    """Entorno vectorizado para entrenamiento paralelo.
    
    Parameters
    ----------
    env_id : str
        ID del entorno base.
    n_envs : int
        Número de entornos paralelos.
    start_method : str
        Método de inicio ('fork', 'spawn', 'forkserver').
    """
    
    def __init__(
        self,
        env_id: str,
        n_envs: int = 4,
        start_method: str = 'spawn'
    ):
        super().__init__(f"vec_{env_id}")
        
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.env_util import make_vec_env
            
            self.make_vec_env = make_vec_env
            self.DummyVecEnv = DummyVecEnv
            self.SubprocVecEnv = SubprocVecEnv
            
        except ImportError:
            raise ImportError(
                "stable-baselines3 required for vectorized environments. "
                "Install with: pip install stable-baselines3"
            )
        
        self.base_env_id = env_id
        self.n_envs = n_envs
        self.start_method = start_method
        
        # Crear entornos vectorizados
        if start_method == 'dummy':
            vec_env_cls = self.DummyVecEnv
        else:
            vec_env_cls = self.SubprocVecEnv
        
        self._env = self.make_vec_env(
            env_id,
            n_envs=n_envs,
            vec_env_cls=vec_env_cls
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea todos los entornos."""
        if seed is not None:
            self._env.seed(seed)
        
        obs = self._env.reset()
        info = [{} for _ in range(self.n_envs)]
        return obs, info
    
    def step(self, action: Union[np.ndarray, List]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Ejecuta acciones en todos los entornos.
        
        Returns
        -------
        Tuple
            observations, rewards, dones, infos (todos arrays/listas de tamaño n_envs)
        """
        obs, rewards, dones, infos = self._env.step(action)
        
        # Convertir dones a terminated/truncated
        terminated = dones
        truncated = np.zeros_like(dones)
        
        # Extraer truncated de infos si existe
        for i, info in enumerate(infos):
            if 'TimeLimit.truncated' in info:
                truncated[i] = info['TimeLimit.truncated']
                terminated[i] = dones[i] and not truncated[i]
        
        return obs, rewards, terminated, truncated, infos
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Renderiza el primer entorno."""
        return self._env.render(mode=mode)
    
    def close(self):
        """Cierra todos los entornos."""
        if self._env:
            self._env.close()
    
    @property
    def observation_space(self):
        """Espacio de observaciones."""
        return self._env.observation_space
    
    @property
    def action_space(self):
        """Espacio de acciones."""
        return self._env.action_space
    
    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Obtiene atributo de los entornos.
        
        Parameters
        ----------
        attr_name : str
            Nombre del atributo.
        indices : Optional[List[int]]
            Índices de los entornos.
            
        Returns
        -------
        List[Any]
            Valores del atributo.
        """
        return self._env.get_attr(attr_name, indices)
    
    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None):
        """Establece atributo en los entornos.
        
        Parameters
        ----------
        attr_name : str
            Nombre del atributo.
        value : Any
            Valor a establecer.
        indices : Optional[List[int]]
            Índices de los entornos.
        """
        self._env.set_attr(attr_name, value, indices)
    
    def env_method(self, method_name: str, *args, indices: Optional[List[int]] = None, **kwargs) -> List[Any]:
        """Llama método en los entornos.
        
        Parameters
        ----------
        method_name : str
            Nombre del método.
        *args
            Argumentos posicionales.
        indices : Optional[List[int]]
            Índices de los entornos.
        **kwargs
            Argumentos keyword.
            
        Returns
        -------
        List[Any]
            Resultados del método.
        """
        return self._env.env_method(method_name, *args, indices=indices, **kwargs)


class EnvironmentWrapper(RLEnvironment):
    """Wrapper base para modificar entornos.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno a envolver.
    """
    
    def __init__(self, env: RLEnvironment):
        super().__init__(f"wrapped_{env.env_id}")
        self.env = env
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea el entorno."""
        return self.env.reset(seed)
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta una acción."""
        return self.env.step(action)
    
    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        return self.env.render()
    
    def close(self):
        """Cierra el entorno."""
        self.env.close()
    
    @property
    def observation_space(self):
        """Espacio de observaciones."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Espacio de acciones."""
        return self.env.action_space


class NormalizeObservationWrapper(EnvironmentWrapper):
    """Normaliza observaciones.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno a envolver.
    epsilon : float
        Valor pequeño para evitar división por cero.
    """
    
    def __init__(self, env: RLEnvironment, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        
        # Estadísticas de normalización
        self.obs_mean = np.zeros(env.observation_space.shape)
        self.obs_var = np.ones(env.observation_space.shape)
        self.obs_count = epsilon
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea y normaliza observación."""
        obs, info = self.env.reset(seed)
        return self._normalize_obs(obs), info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta acción y normaliza observación."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._normalize_obs(obs), reward, terminated, truncated, info
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normaliza observación.
        
        Parameters
        ----------
        obs : np.ndarray
            Observación original.
            
        Returns
        -------
        np.ndarray
            Observación normalizada.
        """
        # Actualizar estadísticas
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += delta * (obs - self.obs_mean)
        
        # Normalizar
        std = np.sqrt(self.obs_var / self.obs_count) + self.epsilon
        return (obs - self.obs_mean) / std


class RewardScalingWrapper(EnvironmentWrapper):
    """Escala recompensas.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno a envolver.
    scale : float
        Factor de escala.
    """
    
    def __init__(self, env: RLEnvironment, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta acción y escala recompensa."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = reward * self.scale
        
        # Guardar recompensa original
        info['original_reward'] = reward
        
        return obs, scaled_reward, terminated, truncated, info


class FrameStackWrapper(EnvironmentWrapper):
    """Apila frames consecutivos.
    
    Útil para dar información temporal en entornos parcialmente observables.
    
    Parameters
    ----------
    env : RLEnvironment
        Entorno a envolver.
    n_frames : int
        Número de frames a apilar.
    """
    
    def __init__(self, env: RLEnvironment, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        
        # Buffer de frames
        obs_shape = env.observation_space.shape
        self.frames_buffer = np.zeros((n_frames,) + obs_shape)
        
        # Actualizar espacio de observaciones
        import gymnasium.spaces as spaces
        self._observation_space = spaces.Box(
            low=np.repeat(env.observation_space.low[np.newaxis, ...], n_frames, axis=0),
            high=np.repeat(env.observation_space.high[np.newaxis, ...], n_frames, axis=0),
            dtype=env.observation_space.dtype
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Resetea y llena buffer con observación inicial."""
        obs, info = self.env.reset(seed)
        
        # Llenar buffer con observación inicial
        for i in range(self.n_frames):
            self.frames_buffer[i] = obs
        
        return self.frames_buffer.copy(), info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta acción y actualiza buffer."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Actualizar buffer (FIFO)
        self.frames_buffer[:-1] = self.frames_buffer[1:]
        self.frames_buffer[-1] = obs
        
        return self.frames_buffer.copy(), reward, terminated, truncated, info
    
    @property
    def observation_space(self):
        """Espacio de observaciones apiladas."""
        return self._observation_space


def create_env(
    env_id: str,
    env_type: str = 'gym',
    wrappers: Optional[List[str]] = None,
    **kwargs
) -> RLEnvironment:
    """Crea un entorno con wrappers opcionales.
    
    Parameters
    ----------
    env_id : str
        ID del entorno.
    env_type : str
        Tipo de entorno ('gym', 'custom', 'vectorized').
    wrappers : Optional[List[str]]
        Lista de wrappers a aplicar.
    **kwargs
        Argumentos para el entorno.
        
    Returns
    -------
    RLEnvironment
        Entorno creado.
    """
    # Crear entorno base
    if env_type == 'gym':
        env = GymEnvironment(env_id, **kwargs)
    elif env_type == 'vectorized':
        env = VectorizedEnvironment(env_id, **kwargs)
    elif env_type == 'custom':
        env = CustomEnvironment(env_id, **kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    # Aplicar wrappers
    if wrappers:
        for wrapper_name in wrappers:
            if wrapper_name == 'normalize':
                env = NormalizeObservationWrapper(env)
            elif wrapper_name == 'reward_scale':
                scale = kwargs.get('reward_scale', 1.0)
                env = RewardScalingWrapper(env, scale)
            elif wrapper_name == 'frame_stack':
                n_frames = kwargs.get('n_frames', 4)
                env = FrameStackWrapper(env, n_frames)
            else:
                logger.warning(f"Unknown wrapper: {wrapper_name}")
    
    return env