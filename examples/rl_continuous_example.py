"""
Ejemplo de Reinforcement Learning con acciones continuas usando MLPY.

Este ejemplo muestra cómo entrenar agentes SAC y TD3 en entornos continuos.
"""

import numpy as np
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mlpy.rl import (
    GymEnvironment,
    TaskRLContinuous,
    SACAgent,
    TD3Agent,
    PPOAgent,
    evaluate_policy,
    record_video,
    plot_rewards,
    save_model,
    load_model,
    compare_agents
)
from mlpy.rl.base import RLConfig, EpisodeLogger
from mlpy.rl.environments import VectorizedEnvironment, create_env


def train_pendulum_sac():
    """Entrena SAC en Pendulum (control continuo)."""
    
    print("=" * 60)
    print("Entrenando SAC en Pendulum-v1")
    print("=" * 60)
    
    # Crear entorno
    env = GymEnvironment(
        env_id='Pendulum-v1',
        render_mode=None
    )
    
    # Crear tarea continua
    task = TaskRLContinuous(
        id='pendulum_task',
        env=env,
        max_episode_steps=200,
        reward_threshold=-200.0  # Pendulum se considera bueno con -200
    )
    
    print(f"\nInformación del entorno:")
    print(f"  - Dimensión de acciones: {task.action_dim}")
    print(f"  - Límites de acciones: {task.action_bounds}")
    print(f"  - Espacio de observaciones: {task.observation_space}")
    
    # Configuración del agente
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=256,
        verbose=1,
        seed=42,
        device='auto'
    )
    
    # Crear agente SAC
    agent = SACAgent(
        env=task,
        config=config,
        buffer_size=1000000,
        learning_starts=1000,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto'  # Ajuste automático del coeficiente de entropía
    )
    
    print("\nConfiguración del agente SAC:")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Gamma: {config.gamma}")
    print(f"  - Buffer size: 1000000")
    print(f"  - Entropy coefficient: auto")
    
    # Callback de logging
    episode_logger = EpisodeLogger(log_frequency=10)
    
    # Entrenar
    print("\nEntrenando agente SAC...")
    stats = agent.train(
        total_timesteps=20000,
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
    print(f"  - Longitud media: {eval_results['mean_length']:.2f}")
    
    # Guardar modelo
    save_path = Path("./models/pendulum_sac")
    save_model(agent, save_path, save_replay_buffer=True)
    print(f"\nModelo guardado en: {save_path}")
    
    # Cerrar entorno
    task.close()
    
    return agent, eval_results


def train_with_td3():
    """Entrena TD3 en un entorno continuo."""
    
    print("\n" + "=" * 60)
    print("Entrenando TD3 en Pendulum-v1")
    print("=" * 60)
    
    # Crear entorno con wrapper de normalización
    env = create_env(
        env_id='Pendulum-v1',
        env_type='gym',
        wrappers=['normalize']
    )
    
    # Crear tarea
    task = TaskRLContinuous(
        id='pendulum_td3_task',
        env=env,
        max_episode_steps=200
    )
    
    # Configuración
    config = RLConfig(
        learning_rate=1e-3,
        gamma=0.98,
        batch_size=100,
        verbose=1,
        seed=42
    )
    
    # Crear agente TD3
    agent = TD3Agent(
        env=task,
        config=config,
        buffer_size=200000,
        learning_starts=100,
        tau=0.005,
        train_freq=(1, "episode"),
        gradient_steps=-1,
        policy_delay=2
    )
    
    print("\nConfiguración del agente TD3:")
    print(f"  - Twin Delayed DDPG")
    print(f"  - Policy delay: 2")
    print(f"  - Action noise: Normal(0, 0.1)")
    
    # Entrenar
    print("\nEntrenando agente TD3...")
    stats = agent.train(total_timesteps=10000)
    
    # Evaluar
    eval_results = agent.evaluate(n_episodes=5, deterministic=True)
    
    print(f"\nResultados TD3:")
    print(f"  - Recompensa media: {eval_results['mean_reward']:.2f}")
    
    # Demostrar normalización de acciones
    print("\nDemostrando normalización de acciones:")
    obs, _ = task.reset()
    action, _ = agent.predict(obs, deterministic=True)
    print(f"  - Acción predicha: {action}")
    
    # Normalizar y desnormalizar
    norm_action = task.normalize_action(action)
    denorm_action = task.denormalize_action(norm_action)
    print(f"  - Acción normalizada: {norm_action}")
    print(f"  - Acción desnormalizada: {denorm_action}")
    
    task.close()
    
    return agent


def vectorized_training():
    """Entrenamiento con entornos vectorizados para mayor eficiencia."""
    
    print("\n" + "=" * 60)
    print("Entrenamiento con entornos vectorizados")
    print("=" * 60)
    
    # Crear entorno vectorizado
    vec_env = VectorizedEnvironment(
        env_id='Pendulum-v1',
        n_envs=4,  # 4 entornos en paralelo
        start_method='spawn'
    )
    
    print(f"\nEntorno vectorizado creado:")
    print(f"  - Número de entornos: 4")
    print(f"  - Método de inicio: spawn")
    
    # Configuración
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_steps=2048,
        verbose=1
    )
    
    # PPO funciona bien con entornos vectorizados
    agent = PPOAgent(
        env=vec_env,
        config=config,
        n_steps=2048,
        n_epochs=10,
        clip_range=0.2
    )
    
    print("\nEntrenando PPO con entornos vectorizados...")
    print("(Esto es más eficiente que entrenar en un solo entorno)")
    
    # Entrenar
    stats = agent.train(total_timesteps=20000)
    
    print(f"\nEntrenamiento completado con {stats['n_updates']} actualizaciones")
    
    # Cerrar entornos
    vec_env.close()
    
    return agent


def compare_continuous_algorithms():
    """Compara SAC, TD3 y PPO en entorno continuo."""
    
    print("\n" + "=" * 60)
    print("Comparando algoritmos en entorno continuo")
    print("=" * 60)
    
    # Preparar agentes
    agents = {}
    timesteps = 10000
    
    # Configuración común
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0,
        seed=42
    )
    
    # SAC
    print("\nEntrenando SAC...")
    env_sac = GymEnvironment('Pendulum-v1')
    task_sac = TaskRLContinuous('pendulum_sac', env_sac)
    agents['SAC'] = SACAgent(
        env=task_sac,
        config=config,
        ent_coef='auto'
    )
    agents['SAC'].train(total_timesteps=timesteps)
    
    # TD3
    print("Entrenando TD3...")
    env_td3 = GymEnvironment('Pendulum-v1')
    task_td3 = TaskRLContinuous('pendulum_td3', env_td3)
    agents['TD3'] = TD3Agent(
        env=task_td3,
        config=config
    )
    agents['TD3'].train(total_timesteps=timesteps)
    
    # PPO (también funciona en continuos)
    print("Entrenando PPO...")
    env_ppo = GymEnvironment('Pendulum-v1')
    task_ppo = TaskRLContinuous('pendulum_ppo', env_ppo)
    agents['PPO'] = PPOAgent(
        env=task_ppo,
        config=config,
        n_steps=2048
    )
    agents['PPO'].train(total_timesteps=timesteps)
    
    # Comparar
    print("\nComparando agentes...")
    env_eval = GymEnvironment('Pendulum-v1')
    comparison_df = compare_agents(
        agents=agents,
        env=env_eval,
        n_eval_episodes=10,
        save_path=Path("./results/continuous_comparison.csv")
    )
    
    # Cerrar entornos
    for agent_name in agents:
        agents[agent_name].env.close()
    env_eval.close()
    
    return comparison_df


def advanced_training_example():
    """Ejemplo avanzado con grabación de video y gráficas."""
    
    print("\n" + "=" * 60)
    print("Ejemplo avanzado con visualización")
    print("=" * 60)
    
    # Crear entorno
    env = GymEnvironment('Pendulum-v1', render_mode='rgb_array')
    task = TaskRLContinuous('pendulum_vis', env)
    
    # Configuración
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0
    )
    
    # Entrenar SAC
    print("\nEntrenando SAC para visualización...")
    agent = SACAgent(env=task, config=config)
    
    # Registrar recompensas durante entrenamiento
    episode_rewards = []
    
    for i in range(5):
        print(f"  Iteración {i+1}/5...")
        agent.train(total_timesteps=2000)
        
        # Evaluar y registrar
        rewards, _ = evaluate_policy(
            agent, task, 
            n_eval_episodes=5,
            return_episode_rewards=True
        )
        episode_rewards.extend(rewards)
    
    # Graficar recompensas
    print("\nGenerando gráfica de recompensas...")
    plot_rewards(
        rewards=episode_rewards,
        window=10,
        save_path=Path("./plots/pendulum_training.png"),
        title="SAC Training on Pendulum",
        show=False  # No mostrar, solo guardar
    )
    print("  Gráfica guardada en ./plots/pendulum_training.png")
    
    # Grabar video del agente entrenado
    print("\nGrabando video del agente...")
    try:
        video_rewards = record_video(
            agent=agent,
            env=task,
            video_path=Path("./videos/pendulum_sac.mp4"),
            n_episodes=2,
            deterministic=True,
            fps=30
        )
        print(f"  Video guardado. Recompensas: {video_rewards}")
    except ImportError:
        print("  OpenCV no instalado, saltando grabación de video")
    
    # Estadísticas finales
    print("\nEstadísticas de la tarea:")
    stats = task.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.2f}")
        else:
            print(f"  - {key}: {value}")
    
    task.close()
    
    return agent, episode_rewards


def multi_task_example():
    """Ejemplo de entrenamiento multi-tarea."""
    
    print("\n" + "=" * 60)
    print("Ejemplo de Reinforcement Learning Multi-Tarea")
    print("=" * 60)
    
    from mlpy.rl.tasks import MultiTaskRL
    
    # Crear múltiples tareas con diferentes entornos
    tasks = []
    
    # Tarea 1: Pendulum estándar
    env1 = GymEnvironment('Pendulum-v1')
    task1 = TaskRLContinuous('pendulum_standard', env1)
    tasks.append(task1)
    
    # Tarea 2: Pendulum con recompensas escaladas
    from mlpy.rl.environments import RewardScalingWrapper
    env2 = GymEnvironment('Pendulum-v1')
    env2_scaled = RewardScalingWrapper(env2, scale=0.1)
    task2 = TaskRLContinuous('pendulum_scaled', env2_scaled)
    tasks.append(task2)
    
    # Crear tarea multi-entorno
    multi_task = MultiTaskRL(
        id='multi_pendulum',
        tasks=tasks,
        task_weights=[1.0, 1.0]  # Pesos iguales
    )
    
    print(f"\nTarea multi-entorno creada con {len(tasks)} tareas")
    
    # Configuración
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0
    )
    
    # Crear agente (usar la primera tarea como referencia)
    agent = SACAgent(
        env=multi_task.current_task,
        config=config
    )
    
    print("\nEntrenando en múltiples tareas...")
    
    # Entrenar alternando entre tareas
    for episode in range(10):
        # Seleccionar tarea
        current_task = multi_task.select_task(method='sequential')
        print(f"  Episodio {episode+1}: Entrenando en {current_task.id}")
        
        # Actualizar entorno del agente
        agent.env = current_task
        
        # Entrenar brevemente
        agent.train(total_timesteps=1000)
    
    # Obtener estadísticas combinadas
    print("\nEstadísticas multi-tarea:")
    combined_stats = multi_task.get_combined_statistics()
    
    for key, value in combined_stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.2f}")
        else:
            print(f"  - {key}: {value}")
    
    # Cerrar todas las tareas
    for task in tasks:
        task.close()
    
    return agent


def load_and_continue_training():
    """Ejemplo de cargar un modelo y continuar entrenamiento."""
    
    print("\n" + "=" * 60)
    print("Cargar y continuar entrenamiento")
    print("=" * 60)
    
    # Primero entrenar y guardar un modelo
    print("\nFase 1: Entrenamiento inicial...")
    
    env = GymEnvironment('Pendulum-v1')
    task = TaskRLContinuous('pendulum_continue', env)
    
    config = RLConfig(
        learning_rate=3e-4,
        gamma=0.99,
        verbose=0
    )
    
    agent = SACAgent(env=task, config=config)
    agent.train(total_timesteps=5000)
    
    # Evaluar antes de guardar
    eval1 = agent.evaluate(n_episodes=5)
    print(f"  Recompensa media inicial: {eval1['mean_reward']:.2f}")
    
    # Guardar
    save_path = Path("./models/pendulum_checkpoint")
    save_model(agent, save_path, save_replay_buffer=True)
    print(f"  Modelo guardado en {save_path}")
    
    # Cerrar todo
    task.close()
    del agent
    
    print("\nFase 2: Cargar y continuar...")
    
    # Crear nuevo entorno
    env_new = GymEnvironment('Pendulum-v1')
    task_new = TaskRLContinuous('pendulum_continue_new', env_new)
    
    # Cargar modelo
    agent_loaded = load_model(
        agent_class=SACAgent,
        load_path=save_path,
        env=task_new,
        load_replay_buffer=True
    )
    print("  Modelo cargado exitosamente")
    
    # Evaluar modelo cargado
    eval2 = agent_loaded.evaluate(n_episodes=5)
    print(f"  Recompensa media después de cargar: {eval2['mean_reward']:.2f}")
    
    # Continuar entrenamiento
    print("  Continuando entrenamiento...")
    agent_loaded.train(total_timesteps=5000)
    
    # Evaluar después de más entrenamiento
    eval3 = agent_loaded.evaluate(n_episodes=5)
    print(f"  Recompensa media después de más entrenamiento: {eval3['mean_reward']:.2f}")
    
    # Mostrar mejora
    improvement = eval3['mean_reward'] - eval1['mean_reward']
    print(f"\n✓ Mejora total: {improvement:.2f}")
    
    task_new.close()
    
    return agent_loaded


if __name__ == "__main__":
    # Ejecutar ejemplos de entornos continuos
    
    # 1. Entrenar SAC en Pendulum
    sac_agent, sac_results = train_pendulum_sac()
    
    # 2. Entrenar TD3
    td3_agent = train_with_td3()
    
    # 3. Entrenamiento vectorizado
    vec_agent = vectorized_training()
    
    # 4. Comparar algoritmos continuos
    comparison = compare_continuous_algorithms()
    
    # 5. Ejemplo avanzado con visualización
    vis_agent, rewards = advanced_training_example()
    
    # 6. Multi-tarea
    multi_agent = multi_task_example()
    
    # 7. Cargar y continuar entrenamiento
    continued_agent = load_and_continue_training()
    
    print("\n" + "=" * 60)
    print("¡Todos los ejemplos de RL continuo completados!")
    print("=" * 60)
    print("\nRecuerda instalar las dependencias opcionales:")
    print("  - pip install stable-baselines3")
    print("  - pip install gymnasium")
    print("  - pip install opencv-python  (para grabar videos)")
    print("  - pip install matplotlib  (para gráficas)")
    print("=" * 60)