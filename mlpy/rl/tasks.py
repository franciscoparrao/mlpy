"""
Tareas de Reinforcement Learning para MLPY.
"""

from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import logging

from ..tasks.base import Task
from .base import RLEnvironment

logger = logging.getLogger(__name__)


class TaskRL(Task):
    """Tarea de Reinforcement Learning.
    
    Parameters
    ----------
    id : str
        ID de la tarea.
    env : RLEnvironment
        Entorno de RL.
    max_episode_steps : Optional[int]
        Máximo de pasos por episodio.
    reward_threshold : Optional[float]
        Umbral de recompensa para considerar resuelto.
    """
    
    def __init__(
        self,
        id: str,
        env: RLEnvironment,
        max_episode_steps: Optional[int] = None,
        reward_threshold: Optional[float] = None
    ):
        # Crear DataFrame dummy para compatibilidad con Task base
        dummy_data = pd.DataFrame({'env_id': [env.env_id]})
        super().__init__(id=id, data=dummy_data)
        
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.reward_threshold = reward_threshold
        self._episode_count = 0
        self._total_steps = 0
        self._episode_rewards = []
    
    @property
    def task_type(self) -> str:
        """Tipo de tarea."""
        return "reinforcement_learning"
    
    @property
    def observation_space(self):
        """Espacio de observaciones."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Espacio de acciones."""
        return self.env.action_space
    
    @property
    def is_discrete(self) -> bool:
        """Si el espacio de acciones es discreto."""
        return self.env.is_discrete
    
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
        obs, info = self.env.reset(seed=seed)
        self._episode_count += 1
        return obs, info
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Ejecuta una acción.
        
        Parameters
        ----------
        action : Union[int, np.ndarray]
            Acción a ejecutar.
            
        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, Dict]
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1
        
        # Verificar límite de pasos
        if self.max_episode_steps and info.get('episode_step', 0) >= self.max_episode_steps:
            truncated = True
        
        # Registrar recompensa si el episodio terminó
        if terminated or truncated:
            episode_reward = info.get('episode_reward', 0)
            self._episode_rewards.append(episode_reward)
            
            # Verificar si se alcanzó el umbral
            if self.reward_threshold is not None:
                recent_rewards = self._episode_rewards[-100:]  # Últimos 100 episodios
                if len(recent_rewards) >= 100:
                    mean_reward = np.mean(recent_rewards)
                    if mean_reward >= self.reward_threshold:
                        info['is_solved'] = True
                        logger.info(f"Task solved! Mean reward: {mean_reward:.2f}")
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Renderiza el entorno."""
        return self.env.render()
    
    def close(self):
        """Cierra el entorno."""
        self.env.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la tarea.
        
        Returns
        -------
        Dict[str, Any]
            Estadísticas.
        """
        stats = {
            'episode_count': self._episode_count,
            'total_steps': self._total_steps,
            'episodes_completed': len(self._episode_rewards)
        }
        
        if self._episode_rewards:
            stats.update({
                'mean_reward': np.mean(self._episode_rewards),
                'std_reward': np.std(self._episode_rewards),
                'min_reward': np.min(self._episode_rewards),
                'max_reward': np.max(self._episode_rewards),
                'last_reward': self._episode_rewards[-1]
            })
            
            # Estadísticas de los últimos 100 episodios
            recent = self._episode_rewards[-100:]
            stats.update({
                'mean_reward_100': np.mean(recent),
                'std_reward_100': np.std(recent)
            })
        
        return stats
    
    def is_solved(self) -> bool:
        """Verifica si la tarea está resuelta.
        
        Returns
        -------
        bool
            True si está resuelta.
        """
        if self.reward_threshold is None:
            return False
        
        if len(self._episode_rewards) < 100:
            return False
        
        mean_reward = np.mean(self._episode_rewards[-100:])
        return mean_reward >= self.reward_threshold


class TaskRLDiscrete(TaskRL):
    """Tarea de RL con acciones discretas.
    
    Especialización para entornos con espacio de acciones discreto.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not self.is_discrete:
            raise ValueError(f"Environment {self.env.env_id} has continuous action space")
    
    @property
    def n_actions(self) -> int:
        """Número de acciones posibles."""
        return self.action_space.n
    
    def get_action_meanings(self) -> Optional[List[str]]:
        """Obtiene los significados de las acciones.
        
        Returns
        -------
        Optional[List[str]]
            Lista con significados o None.
        """
        if hasattr(self.env, 'get_action_meanings'):
            return self.env.get_action_meanings()
        
        # Intentar obtener de metadatos del entorno
        if hasattr(self.env, 'metadata'):
            return self.env.metadata.get('action_meanings')
        
        return None
    
    def action_probabilities(self, policy_probs: np.ndarray) -> Dict[str, float]:
        """Convierte probabilidades a diccionario con nombres.
        
        Parameters
        ----------
        policy_probs : np.ndarray
            Probabilidades de la política.
            
        Returns
        -------
        Dict[str, float]
            Diccionario acción->probabilidad.
        """
        meanings = self.get_action_meanings()
        
        if meanings:
            return {meaning: prob for meaning, prob in zip(meanings, policy_probs)}
        else:
            return {f"action_{i}": prob for i, prob in enumerate(policy_probs)}


class TaskRLContinuous(TaskRL):
    """Tarea de RL con acciones continuas.
    
    Especialización para entornos con espacio de acciones continuo.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if self.is_discrete:
            raise ValueError(f"Environment {self.env.env_id} has discrete action space")
    
    @property
    def action_dim(self) -> int:
        """Dimensión del espacio de acciones."""
        return self.action_space.shape[0]
    
    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Límites del espacio de acciones.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (low, high) bounds.
        """
        return self.action_space.low, self.action_space.high
    
    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Recorta acción a los límites válidos.
        
        Parameters
        ----------
        action : np.ndarray
            Acción a recortar.
            
        Returns
        -------
        np.ndarray
            Acción recortada.
        """
        return np.clip(action, self.action_space.low, self.action_space.high)
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normaliza acción al rango [-1, 1].
        
        Parameters
        ----------
        action : np.ndarray
            Acción en espacio original.
            
        Returns
        -------
        np.ndarray
            Acción normalizada.
        """
        low, high = self.action_bounds
        return 2.0 * (action - low) / (high - low) - 1.0
    
    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Desnormaliza acción desde [-1, 1].
        
        Parameters
        ----------
        action : np.ndarray
            Acción normalizada.
            
        Returns
        -------
        np.ndarray
            Acción en espacio original.
        """
        low, high = self.action_bounds
        return low + (action + 1.0) * (high - low) / 2.0


class MultiTaskRL(Task):
    """Tarea multi-entorno para RL.
    
    Permite entrenar en múltiples entornos simultáneamente.
    
    Parameters
    ----------
    id : str
        ID de la tarea.
    tasks : List[TaskRL]
        Lista de tareas RL.
    task_weights : Optional[List[float]]
        Pesos para cada tarea.
    """
    
    def __init__(
        self,
        id: str,
        tasks: List[TaskRL],
        task_weights: Optional[List[float]] = None
    ):
        if not tasks:
            raise ValueError("At least one task required")
        
        # Verificar compatibilidad de espacios
        first_obs_space = tasks[0].observation_space
        first_act_space = tasks[0].action_space
        
        for task in tasks[1:]:
            if task.observation_space != first_obs_space:
                logger.warning(f"Task {task.id} has different observation space")
            if task.action_space != first_act_space:
                logger.warning(f"Task {task.id} has different action space")
        
        # DataFrame dummy
        dummy_data = pd.DataFrame({'task_id': [t.id for t in tasks]})
        super().__init__(id=id, data=dummy_data)
        
        self.tasks = tasks
        self.task_weights = task_weights or [1.0] * len(tasks)
        self.current_task_idx = 0
        self.current_task = tasks[0]
    
    @property
    def task_type(self) -> str:
        """Tipo de tarea."""
        return "multi_reinforcement_learning"
    
    def select_task(self, method: str = 'sequential') -> TaskRL:
        """Selecciona siguiente tarea.
        
        Parameters
        ----------
        method : str
            Método de selección ('sequential', 'random', 'weighted').
            
        Returns
        -------
        TaskRL
            Tarea seleccionada.
        """
        if method == 'sequential':
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        elif method == 'random':
            self.current_task_idx = np.random.randint(len(self.tasks))
        elif method == 'weighted':
            probs = np.array(self.task_weights) / np.sum(self.task_weights)
            self.current_task_idx = np.random.choice(len(self.tasks), p=probs)
        
        self.current_task = self.tasks[self.current_task_idx]
        return self.current_task
    
    def get_combined_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas combinadas.
        
        Returns
        -------
        Dict[str, Any]
            Estadísticas de todas las tareas.
        """
        combined_stats = {}
        
        for i, task in enumerate(self.tasks):
            task_stats = task.get_statistics()
            for key, value in task_stats.items():
                combined_stats[f"task_{i}_{key}"] = value
        
        # Estadísticas agregadas
        all_rewards = []
        for task in self.tasks:
            if task._episode_rewards:
                all_rewards.extend(task._episode_rewards)
        
        if all_rewards:
            combined_stats['overall_mean_reward'] = np.mean(all_rewards)
            combined_stats['overall_std_reward'] = np.std(all_rewards)
        
        return combined_stats