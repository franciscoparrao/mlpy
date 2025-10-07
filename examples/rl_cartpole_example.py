"""
Ejemplo de Reinforcement Learning con CartPole usando MLPY.

Este ejemplo muestra cómo entrenar un agente PPO para resolver CartPole.
"""

import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar componentes de MLPY RL
from mlpy.rl import (
    GymEnvironment,
    TaskRLDiscrete,
    PPOAgent,
    DQNAgent,
    A2CAgent,
    evaluate_policy,
    plot_rewards,
    save_model
)
from mlpy.rl.base import RLConfig, EpisodeLogger, CheckpointCallback


def train_cartpole_ppo():
    """Entrena PPO en CartPole."""
    
    print("=" * 60)
    print("Entrenando PPO en CartPole-v1")
    print("=" * 60)
    
    # Crear entorno
    env = GymEnvironment(
        env_id='CartPole-v1',
        render_mode=None  # Cambiar a 'human' para ver el entrenamiento
    )
    
    # Crear tarea de RL
    task = TaskRLDiscrete(
        id='cartpole_task',
        env=env,
        max_episode_steps=500,
        reward_threshold=475.0  # CartPole se considera resuelto con 475
    )
    
    # Configuración del agente
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        verbose=1,
        seed=42,
        device='auto'
    )
    
    # Crear agente PPO
    agent = PPOAgent(
        env=task,
        config=config,
        n_steps=2048,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    print("\nConfiguración del agente:")
    print(f"  - Algoritmo: PPO")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - N steps: {config.n_steps}")
    
    # Callbacks
    episode_logger = EpisodeLogger(log_frequency=10)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/cartpole_ppo",
        name_prefix="ppo_model"
    )
    
    # Entrenar
    print("\nEntrenando agente...")
    stats = agent.train(
        total_timesteps=50000,
        callback=episode_logger
    )
    
    print(f"\nEntrenamiento completado. Total timesteps: {stats['total_timesteps']}")
    
    # Evaluar
    print("\nEvaluando agente...")
    eval_results = agent.evaluate(
        n_episodes=10,
        deterministic=True,
        render=False
    )
    
    print(f"\nResultados de evaluación:")
    print(f"  - Recompensa media: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"  - Longitud media: {eval_results['mean_length']:.2f} +/- {eval_results['std_length']:.2f}")
    print(f"  - Recompensa mínima: {eval_results['min_reward']:.2f}")
    print(f"  - Recompensa máxima: {eval_results['max_reward']:.2f}")
    
    # Verificar si la tarea está resuelta
    if task.is_solved():
        print("\n✓ ¡Tarea resuelta!")
    
    # Guardar modelo
    save_path = Path("./models/cartpole_ppo")
    save_model(agent, save_path)
    print(f"\nModelo guardado en: {save_path}")
    
    # Jugar algunos episodios
    print("\nJugando 3 episodios con el agente entrenado...")
    rewards = agent.play(
        n_episodes=3,
        deterministic=True,
        render=False,  # Cambiar a True para ver al agente jugar
        fps=30
    )
    print(f"Recompensas obtenidas: {rewards}")
    
    # Cerrar entorno
    task.close()
    
    return agent, eval_results


def compare_algorithms():
    """Compara diferentes algoritmos en CartPole."""
    
    print("\n" + "=" * 60)
    print("Comparando algoritmos en CartPole-v1")
    print("=" * 60)
    
    # Configuración común
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        verbose=0,
        seed=42
    )
    
    timesteps = 30000
    results = {}
    
    # Algoritmos a comparar
    algorithms = {
        'PPO': PPOAgent,
        'DQN': DQNAgent,
        'A2C': A2CAgent
    }
    
    for name, agent_class in algorithms.items():
        print(f"\nEntrenando {name}...")
        
        # Crear entorno y tarea
        env = GymEnvironment('CartPole-v1')
        task = TaskRLDiscrete('cartpole_task', env)
        
        # Crear y entrenar agente
        if name == 'PPO':
            agent = agent_class(env=task, config=config, n_steps=2048)
        elif name == 'DQN':
            agent = agent_class(env=task, config=config, learning_starts=1000)
        else:  # A2C
            agent = agent_class(env=task, config=config, n_steps=5)
        
        # Entrenar
        agent.train(total_timesteps=timesteps)
        
        # Evaluar
        eval_results = agent.evaluate(n_episodes=10, deterministic=True)
        results[name] = eval_results
        
        print(f"{name} - Recompensa media: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
        
        # Cerrar entorno
        task.close()
    
    # Mostrar comparación
    print("\n" + "=" * 60)
    print("Resumen de resultados:")
    print("=" * 60)
    
    print(f"\n{'Algoritmo':<10} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    
    for name, res in results.items():
        print(f"{name:<10} {res['mean_reward']:<10.2f} {res['std_reward']:<10.2f} "
              f"{res['min_reward']:<10.2f} {res['max_reward']:<10.2f}")
    
    # Determinar ganador
    best_algo = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"\n✓ Mejor algoritmo: {best_algo[0]} con recompensa media de {best_algo[1]['mean_reward']:.2f}")
    
    return results


def demonstrate_wrappers():
    """Demuestra el uso de wrappers de entorno."""
    
    print("\n" + "=" * 60)
    print("Demostrando wrappers de entorno")
    print("=" * 60)
    
    from mlpy.rl.environments import (
        NormalizeObservationWrapper,
        RewardScalingWrapper,
        FrameStackWrapper,
        create_env
    )
    
    # Crear entorno base
    base_env = GymEnvironment('CartPole-v1')
    print(f"\nEntorno base: {base_env.env_id}")
    print(f"  - Espacio de observaciones: {base_env.observation_space}")
    print(f"  - Espacio de acciones: {base_env.action_space}")
    
    # Aplicar wrapper de normalización
    print("\n1. Wrapper de normalización de observaciones:")
    normalized_env = NormalizeObservationWrapper(base_env)
    obs, info = normalized_env.reset()
    print(f"  - Observación normalizada shape: {obs.shape}")
    print(f"  - Observación normalizada (primeros valores): {obs[:2]}")
    
    # Aplicar wrapper de escalado de recompensas
    print("\n2. Wrapper de escalado de recompensas:")
    scaled_env = RewardScalingWrapper(base_env, scale=0.1)
    obs, info = scaled_env.reset()
    action = 0  # Acción dummy
    obs, reward, terminated, truncated, info = scaled_env.step(action)
    print(f"  - Recompensa escalada: {reward}")
    if 'original_reward' in info:
        print(f"  - Recompensa original: {info['original_reward']}")
    
    # Usar función helper para crear entorno con wrappers
    print("\n3. Crear entorno con múltiples wrappers:")
    env_with_wrappers = create_env(
        env_id='CartPole-v1',
        env_type='gym',
        wrappers=['normalize', 'reward_scale'],
        reward_scale=0.01
    )
    print(f"  - Entorno creado con wrappers aplicados")
    
    # Cerrar entornos
    base_env.close()
    env_with_wrappers.close()
    
    print("\n✓ Demostración de wrappers completada")


def custom_environment_example():
    """Ejemplo de entorno personalizado."""
    
    print("\n" + "=" * 60)
    print("Ejemplo de entorno personalizado")
    print("=" * 60)
    
    from mlpy.rl import CustomEnvironment
    import gymnasium.spaces as spaces
    
    # Definir funciones del entorno personalizado
    def reset_fn():
        """Resetea el entorno simple."""
        initial_state = np.array([0.0, 0.0])
        info = {'step': 0}
        return initial_state, info
    
    def step_fn(state, action):
        """Ejecuta un paso en el entorno simple."""
        # Entorno simple: mover en 2D hacia objetivo en (1, 1)
        if action == 0:  # Derecha
            state = state + np.array([0.1, 0.0])
        elif action == 1:  # Arriba
            state = state + np.array([0.0, 0.1])
        elif action == 2:  # Izquierda
            state = state + np.array([-0.1, 0.0])
        else:  # Abajo
            state = state + np.array([0.0, -0.1])
        
        # Calcular recompensa (distancia negativa al objetivo)
        target = np.array([1.0, 1.0])
        distance = np.linalg.norm(state - target)
        reward = -distance
        
        # Terminar si está cerca del objetivo
        terminated = distance < 0.1
        truncated = False
        
        info = {'distance': distance}
        
        return state, reward, terminated, truncated, info
    
    # Crear entorno personalizado
    custom_env = CustomEnvironment(
        env_id='simple_2d_navigation',
        observation_space=spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32),
        action_space=spaces.Discrete(4),
        reset_fn=reset_fn,
        step_fn=step_fn
    )
    
    print(f"Entorno personalizado creado: {custom_env.env_id}")
    print(f"  - Espacio de observaciones: {custom_env.observation_space}")
    print(f"  - Espacio de acciones: {custom_env.action_space}")
    
    # Crear tarea
    task = TaskRLDiscrete(
        id='navigation_task',
        env=custom_env,
        max_episode_steps=100
    )
    
    # Entrenar agente simple
    print("\nEntrenando DQN en entorno personalizado...")
    
    config = RLConfig(
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=32,
        verbose=0
    )
    
    agent = DQNAgent(
        env=task,
        config=config,
        buffer_size=10000,
        learning_starts=100,
        exploration_fraction=0.2
    )
    
    # Entrenar
    agent.train(total_timesteps=5000)
    
    # Evaluar
    print("\nEvaluando agente...")
    eval_results = agent.evaluate(n_episodes=5)
    
    print(f"Recompensa media: {eval_results['mean_reward']:.2f}")
    
    # Probar una trayectoria
    print("\nProbando una trayectoria:")
    obs, _ = task.reset()
    print(f"  Estado inicial: {obs}")
    
    for step in range(10):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = task.step(action)
        print(f"  Paso {step+1}: acción={action}, estado={obs[:2]}, recompensa={reward:.3f}")
        
        if terminated:
            print("  ¡Objetivo alcanzado!")
            break
    
    task.close()
    print("\n✓ Ejemplo de entorno personalizado completado")


if __name__ == "__main__":
    # Ejecutar ejemplos
    
    # 1. Entrenar PPO en CartPole
    agent, results = train_cartpole_ppo()
    
    # 2. Comparar algoritmos
    comparison_results = compare_algorithms()
    
    # 3. Demostrar wrappers
    demonstrate_wrappers()
    
    # 4. Entorno personalizado
    custom_environment_example()
    
    print("\n" + "=" * 60)
    print("¡Todos los ejemplos completados!")
    print("=" * 60)