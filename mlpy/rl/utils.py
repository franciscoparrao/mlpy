"""
Utilidades para Reinforcement Learning.
"""

from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_policy(
    agent,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False,
    callback: Optional[Callable] = None
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """Evalúa una política en un entorno.
    
    Parameters
    ----------
    agent : RLAgent
        Agente a evaluar.
    env : RLEnvironment
        Entorno de evaluación.
    n_eval_episodes : int
        Número de episodios de evaluación.
    deterministic : bool
        Si usar política determinística.
    render : bool
        Si renderizar la evaluación.
    return_episode_rewards : bool
        Si retornar recompensas por episodio.
    callback : Optional[Callable]
        Callback para cada step.
        
    Returns
    -------
    Union[Tuple[float, float], Tuple[List[float], List[int]]]
        Media y desviación de recompensas, o listas de recompensas y longitudes.
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Predecir acción
            action, _ = agent.predict(obs, deterministic=deterministic)
            
            # Ejecutar acción
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
            
            if callback:
                callback(locals(), globals())
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        logger.debug(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    logger.info(f"Evaluation over {n_eval_episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    
    return mean_reward, std_reward


def record_video(
    agent,
    env,
    video_path: Union[str, Path],
    n_episodes: int = 1,
    deterministic: bool = True,
    fps: int = 30,
    video_length: Optional[int] = None
) -> List[float]:
    """Graba video de episodios del agente.
    
    Parameters
    ----------
    agent : RLAgent
        Agente a grabar.
    env : RLEnvironment
        Entorno.
    video_path : Union[str, Path]
        Ruta para guardar el video.
    n_episodes : int
        Número de episodios a grabar.
    deterministic : bool
        Si usar política determinística.
    fps : int
        Frames por segundo.
    video_length : Optional[int]
        Longitud máxima del video en frames.
        
    Returns
    -------
    List[float]
        Recompensas de los episodios grabados.
    """
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for video recording. Install with: pip install opencv-python")
    
    # Configurar video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    episode_rewards = []
    total_frames = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_frames = []
        
        while not done:
            # Renderizar frame
            frame = env.render()
            
            if frame is not None:
                # Inicializar video writer con el primer frame
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        fps,
                        (width, height)
                    )
                
                # Convertir RGB a BGR para OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                episode_frames.append(frame)
            
            # Predecir y ejecutar acción
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            total_frames += 1
            
            # Verificar límite de longitud
            if video_length and total_frames >= video_length:
                done = True
                break
        
        # Escribir frames del episodio
        for frame in episode_frames:
            video_writer.write(frame)
        
        episode_rewards.append(episode_reward)
        logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        if video_length and total_frames >= video_length:
            break
    
    # Cerrar video writer
    if video_writer:
        video_writer.release()
        logger.info(f"Video saved to {video_path}")
    
    return episode_rewards


def plot_rewards(
    rewards: Union[List[float], Dict[str, List[float]]],
    window: int = 100,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Rewards",
    show: bool = True
) -> None:
    """Grafica curva de recompensas.
    
    Parameters
    ----------
    rewards : Union[List[float], Dict[str, List[float]]]
        Recompensas o diccionario de recompensas.
    window : int
        Ventana para media móvil.
    save_path : Optional[Union[str, Path]]
        Ruta para guardar gráfica.
    title : str
        Título de la gráfica.
    show : bool
        Si mostrar la gráfica.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Cannot plot rewards.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convertir a diccionario si es lista
    if isinstance(rewards, list):
        rewards = {"Rewards": rewards}
    
    # Graficar cada serie
    for label, reward_list in rewards.items():
        if not reward_list:
            continue
        
        episodes = range(1, len(reward_list) + 1)
        
        # Recompensas originales
        ax.plot(episodes, reward_list, alpha=0.3, label=f"{label} (raw)")
        
        # Media móvil
        if len(reward_list) >= window:
            moving_avg = pd.Series(reward_list).rolling(window=window).mean()
            ax.plot(episodes, moving_avg, label=f"{label} (avg {window})")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()


def save_model(
    agent,
    save_path: Union[str, Path],
    save_replay_buffer: bool = False,
    save_env: bool = False
) -> None:
    """Guarda modelo y metadatos.
    
    Parameters
    ----------
    agent : RLAgent
        Agente a guardar.
    save_path : Union[str, Path]
        Ruta de guardado.
    save_replay_buffer : bool
        Si guardar replay buffer (para algoritmos off-policy).
    save_env : bool
        Si guardar entorno.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar modelo
    model_path = save_path.with_suffix('.zip')
    agent.save(model_path)
    
    # Guardar metadatos
    metadata = {
        'agent_class': agent.__class__.__name__,
        'env_id': agent.env.env_id if hasattr(agent.env, 'env_id') else 'unknown',
        'timestamp': datetime.now().isoformat(),
        'is_trained': agent.is_trained,
        'config': {}
    }
    
    # Guardar configuración si existe
    if hasattr(agent, 'config') and agent.config:
        metadata['config'] = {
            'learning_rate': agent.config.learning_rate,
            'gamma': agent.config.gamma,
            'batch_size': agent.config.batch_size,
            'device': agent.config.device
        }
    
    # Guardar replay buffer si se solicita
    if save_replay_buffer and hasattr(agent, 'model') and hasattr(agent.model, 'replay_buffer'):
        buffer_path = save_path.with_suffix('.pkl')
        try:
            import pickle
            with open(buffer_path, 'wb') as f:
                pickle.dump(agent.model.replay_buffer, f)
            metadata['replay_buffer_path'] = str(buffer_path)
            logger.info(f"Replay buffer saved to {buffer_path}")
        except Exception as e:
            logger.warning(f"Could not save replay buffer: {e}")
    
    # Guardar entorno si se solicita
    if save_env and hasattr(agent, 'env'):
        env_path = save_path.with_suffix('.env.pkl')
        try:
            import pickle
            with open(env_path, 'wb') as f:
                pickle.dump(agent.env, f)
            metadata['env_path'] = str(env_path)
            logger.info(f"Environment saved to {env_path}")
        except Exception as e:
            logger.warning(f"Could not save environment: {e}")
    
    # Guardar metadatos
    metadata_path = save_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model and metadata saved to {save_path}")


def load_model(
    agent_class,
    load_path: Union[str, Path],
    env: Optional[Any] = None,
    load_replay_buffer: bool = False,
    custom_objects: Optional[Dict] = None
):
    """Carga modelo y metadatos.
    
    Parameters
    ----------
    agent_class : type
        Clase del agente.
    load_path : Union[str, Path]
        Ruta de carga.
    env : Optional[Any]
        Entorno (si no se carga desde archivo).
    load_replay_buffer : bool
        Si cargar replay buffer.
    custom_objects : Optional[Dict]
        Objetos personalizados para carga.
        
    Returns
    -------
    RLAgent
        Agente cargado.
    """
    load_path = Path(load_path)
    
    # Cargar metadatos
    metadata_path = load_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    else:
        metadata = {}
        logger.warning(f"No metadata found at {metadata_path}")
    
    # Cargar entorno si está guardado y no se proporciona
    if env is None and 'env_path' in metadata:
        env_path = Path(metadata['env_path'])
        if env_path.exists():
            try:
                import pickle
                with open(env_path, 'rb') as f:
                    env = pickle.load(f)
                logger.info(f"Environment loaded from {env_path}")
            except Exception as e:
                logger.warning(f"Could not load environment: {e}")
    
    if env is None:
        raise ValueError("Environment must be provided or loaded from file")
    
    # Crear agente vacío
    from .base import RLConfig
    config = RLConfig()
    if 'config' in metadata:
        for key, value in metadata['config'].items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    agent = agent_class(env=env, config=config)
    
    # Cargar modelo
    model_path = load_path.with_suffix('.zip')
    if model_path.exists():
        agent.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Cargar replay buffer si se solicita
    if load_replay_buffer and 'replay_buffer_path' in metadata:
        buffer_path = Path(metadata['replay_buffer_path'])
        if buffer_path.exists():
            try:
                import pickle
                with open(buffer_path, 'rb') as f:
                    replay_buffer = pickle.load(f)
                if hasattr(agent, 'model') and hasattr(agent.model, 'replay_buffer'):
                    agent.model.replay_buffer = replay_buffer
                    logger.info(f"Replay buffer loaded from {buffer_path}")
            except Exception as e:
                logger.warning(f"Could not load replay buffer: {e}")
    
    return agent


def create_training_curves(
    log_dir: Union[str, Path],
    metrics: List[str] = ['rewards', 'lengths', 'losses'],
    window: int = 100,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """Crea gráficas de entrenamiento desde logs.
    
    Parameters
    ----------
    log_dir : Union[str, Path]
        Directorio con logs.
    metrics : List[str]
        Métricas a graficar.
    window : int
        Ventana para media móvil.
    save_path : Optional[Union[str, Path]]
        Ruta para guardar gráficas.
    """
    log_dir = Path(log_dir)
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Cannot create plots.")
        return
    
    # Buscar archivos de log
    log_files = list(log_dir.glob("*.csv")) + list(log_dir.glob("*.json"))
    
    if not log_files:
        logger.warning(f"No log files found in {log_dir}")
        return
    
    # Crear subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    # Procesar cada archivo de log
    for log_file in log_files:
        if log_file.suffix == '.csv':
            df = pd.read_csv(log_file)
        else:
            with open(log_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        
        # Graficar cada métrica
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Buscar columnas relevantes
            metric_cols = [col for col in df.columns if metric.lower() in col.lower()]
            
            for col in metric_cols:
                if col in df:
                    values = df[col].dropna()
                    if len(values) > 0:
                        # Valores originales
                        ax.plot(values.index, values.values, alpha=0.3, label=f"{log_file.stem}_{col}")
                        
                        # Media móvil
                        if len(values) >= window:
                            moving_avg = values.rolling(window=window).mean()
                            ax.plot(moving_avg.index, moving_avg.values, label=f"{log_file.stem}_{col}_avg")
            
            ax.set_xlabel("Step")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Training {metric.capitalize()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    plt.show()
    plt.close()


def compare_agents(
    agents: Dict[str, Any],
    env,
    n_eval_episodes: int = 10,
    metrics: List[str] = ['mean_reward', 'std_reward', 'mean_length'],
    save_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Compara múltiples agentes.
    
    Parameters
    ----------
    agents : Dict[str, Any]
        Diccionario nombre->agente.
    env : RLEnvironment
        Entorno de evaluación.
    n_eval_episodes : int
        Episodios de evaluación por agente.
    metrics : List[str]
        Métricas a comparar.
    save_path : Optional[Union[str, Path]]
        Ruta para guardar resultados.
        
    Returns
    -------
    pd.DataFrame
        Tabla comparativa.
    """
    results = []
    
    for name, agent in agents.items():
        logger.info(f"Evaluating {name}...")
        
        # Evaluar agente
        eval_results = agent.evaluate(
            n_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Añadir nombre
        eval_results['agent'] = name
        results.append(eval_results)
    
    # Crear DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reordenar columnas
    cols = ['agent'] + [col for col in comparison_df.columns if col != 'agent']
    comparison_df = comparison_df[cols]
    
    # Guardar si se especifica
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.csv':
            comparison_df.to_csv(save_path, index=False)
        else:
            comparison_df.to_json(save_path, orient='records', indent=2)
        
        logger.info(f"Comparison saved to {save_path}")
    
    # Mostrar resultados
    logger.info("\nAgent Comparison:")
    logger.info(comparison_df.to_string())
    
    # Crear gráficas si matplotlib está disponible
    try:
        import matplotlib.pyplot as plt
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[i]
                comparison_df.plot(x='agent', y=metric, kind='bar', ax=ax)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel("")
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend().remove()
        
        plt.suptitle("Agent Comparison")
        plt.tight_layout()
        
        if save_path:
            plot_path = save_path.with_suffix('.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {plot_path}")
        
        plt.show()
        plt.close()
    except ImportError:
        pass
    
    return comparison_df


def create_benchmark_suite(
    env_ids: List[str],
    agent_configs: Dict[str, Dict],
    n_train_steps: int = 100000,
    n_eval_episodes: int = 10,
    save_dir: Union[str, Path] = "./benchmarks"
) -> pd.DataFrame:
    """Crea suite de benchmarking.
    
    Parameters
    ----------
    env_ids : List[str]
        IDs de entornos a evaluar.
    agent_configs : Dict[str, Dict]
        Configuraciones de agentes.
    n_train_steps : int
        Pasos de entrenamiento.
    n_eval_episodes : int
        Episodios de evaluación.
    save_dir : Union[str, Path]
        Directorio para guardar resultados.
        
    Returns
    -------
    pd.DataFrame
        Resultados del benchmark.
    """
    from .environments import create_env
    from .agents import PPOAgent, DQNAgent, SACAgent, A2CAgent, TD3Agent
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    agent_classes = {
        'PPO': PPOAgent,
        'DQN': DQNAgent,
        'SAC': SACAgent,
        'A2C': A2CAgent,
        'TD3': TD3Agent
    }
    
    results = []
    
    for env_id in env_ids:
        logger.info(f"\nBenchmarking environment: {env_id}")
        
        # Crear entorno
        env = create_env(env_id, env_type='gym')
        
        for agent_name, agent_config in agent_configs.items():
            if agent_name not in agent_classes:
                logger.warning(f"Unknown agent: {agent_name}")
                continue
            
            # Verificar compatibilidad
            agent_class = agent_classes[agent_name]
            if agent_name in ['DQN'] and not env.is_discrete:
                logger.info(f"Skipping {agent_name} for continuous environment")
                continue
            if agent_name in ['SAC', 'TD3'] and env.is_discrete:
                logger.info(f"Skipping {agent_name} for discrete environment")
                continue
            
            logger.info(f"Training {agent_name}...")
            
            try:
                # Crear y entrenar agente
                agent = agent_class(env=env, **agent_config)
                train_stats = agent.train(total_timesteps=n_train_steps)
                
                # Evaluar
                eval_stats = agent.evaluate(
                    n_episodes=n_eval_episodes,
                    deterministic=True
                )
                
                # Guardar modelo
                model_path = save_dir / f"{env_id}_{agent_name}_model"
                save_model(agent, model_path)
                
                # Registrar resultados
                result = {
                    'environment': env_id,
                    'agent': agent_name,
                    'train_steps': n_train_steps,
                    'eval_episodes': n_eval_episodes,
                    **eval_stats
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error with {agent_name} on {env_id}: {e}")
                result = {
                    'environment': env_id,
                    'agent': agent_name,
                    'error': str(e)
                }
                results.append(result)
        
        # Cerrar entorno
        env.close()
    
    # Crear DataFrame de resultados
    benchmark_df = pd.DataFrame(results)
    
    # Guardar resultados
    results_path = save_dir / "benchmark_results.csv"
    benchmark_df.to_csv(results_path, index=False)
    logger.info(f"\nBenchmark results saved to {results_path}")
    
    # Mostrar resumen
    logger.info("\nBenchmark Summary:")
    if 'mean_reward' in benchmark_df.columns:
        summary = benchmark_df.groupby('agent')['mean_reward'].agg(['mean', 'std', 'min', 'max'])
        logger.info(summary.to_string())
    
    return benchmark_df