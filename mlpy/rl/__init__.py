"""
Módulo de Reinforcement Learning para MLPY.

Integración con Gymnasium y Stable Baselines3 para aprendizaje por refuerzo.
"""

from .base import RLAgent, RLEnvironment, RLCallback
from .tasks import TaskRL, TaskRLDiscrete, TaskRLContinuous
from .agents import (
    PPOAgent,
    DQNAgent,
    SACAgent,
    A2CAgent,
    TD3Agent
)
from .environments import (
    GymEnvironment,
    CustomEnvironment,
    VectorizedEnvironment,
    create_env
)
from .utils import (
    evaluate_policy,
    record_video,
    plot_rewards,
    save_model,
    load_model
)

__all__ = [
    # Base
    'RLAgent',
    'RLEnvironment', 
    'RLCallback',
    
    # Tasks
    'TaskRL',
    'TaskRLDiscrete',
    'TaskRLContinuous',
    
    # Agents
    'PPOAgent',
    'DQNAgent',
    'SACAgent',
    'A2CAgent',
    'TD3Agent',
    
    # Environments
    'GymEnvironment',
    'CustomEnvironment',
    'VectorizedEnvironment',
    'create_env',
    
    # Utils
    'evaluate_policy',
    'record_video',
    'plot_rewards',
    'save_model',
    'load_model'
]