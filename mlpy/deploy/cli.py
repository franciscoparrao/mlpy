"""
CLI para el servidor de modelos MLPY.

Proporciona comandos para iniciar y gestionar el servidor de modelos.
"""

import click
import uvicorn
from pathlib import Path
from typing import Optional
import logging

from .api import create_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """MLPY Model Server CLI."""
    pass


@cli.command()
@click.option(
    '--registry-path',
    default="./mlpy_models",
    help='Ruta al registry de modelos'
)
@click.option(
    '--host',
    default="0.0.0.0",
    help='Host para el servidor'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Puerto para el servidor'
)
@click.option(
    '--default-model',
    default=None,
    help='Modelo por defecto'
)
@click.option(
    '--api-key',
    default=None,
    help='API key para autenticación'
)
@click.option(
    '--enable-auth/--no-auth',
    default=False,
    help='Habilitar autenticación'
)
@click.option(
    '--enable-cors/--no-cors',
    default=True,
    help='Habilitar CORS'
)
@click.option(
    '--workers',
    default=1,
    type=int,
    help='Número de workers'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Recargar automáticamente en cambios'
)
def serve(
    registry_path: str,
    host: str,
    port: int,
    default_model: Optional[str],
    api_key: Optional[str],
    enable_auth: bool,
    enable_cors: bool,
    workers: int,
    reload: bool
):
    """Inicia el servidor de modelos MLPY.
    
    Ejemplos:
        # Servidor básico
        mlpy-serve serve
        
        # Con autenticación
        mlpy-serve serve --enable-auth --api-key mysecretkey
        
        # Con modelo por defecto
        mlpy-serve serve --default-model iris_classifier
        
        # Desarrollo con recarga automática
        mlpy-serve serve --reload
    """
    # Verificar que el registry existe
    registry_path = Path(registry_path)
    if not registry_path.exists():
        logger.warning(f"Registry path {registry_path} does not exist. Creating...")
        registry_path.mkdir(parents=True, exist_ok=True)
    
    # Validar autenticación
    if enable_auth and not api_key:
        raise click.ClickException("API key required when authentication is enabled")
    
    # Crear aplicación
    logger.info(f"Creating MLPY Model Server...")
    logger.info(f"Registry path: {registry_path}")
    logger.info(f"Default model: {default_model or 'None'}")
    logger.info(f"Authentication: {'Enabled' if enable_auth else 'Disabled'}")
    logger.info(f"CORS: {'Enabled' if enable_cors else 'Disabled'}")
    
    app = create_app(
        registry_path=str(registry_path),
        default_model=default_model,
        enable_auth=enable_auth,
        api_key=api_key,
        enable_cors=enable_cors
    )
    
    # Configurar uvicorn
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )


@cli.command()
@click.option(
    '--registry-path',
    default="./mlpy_models",
    help='Ruta al registry de modelos'
)
def list_models(registry_path: str):
    """Lista todos los modelos en el registry.
    
    Ejemplo:
        mlpy-serve list-models
    """
    from ..registry import FileSystemRegistry
    
    registry_path = Path(registry_path)
    if not registry_path.exists():
        click.echo(f"Registry path {registry_path} does not exist")
        return
    
    registry = FileSystemRegistry(str(registry_path))
    models = registry.list_models()
    
    if not models:
        click.echo("No models found in registry")
        return
    
    click.echo(f"Found {len(models)} model(s):")
    for model_name in models:
        versions = registry.list_versions(model_name)
        click.echo(f"  - {model_name}: {len(versions)} version(s)")
        for version in versions[:3]:  # Mostrar solo las primeras 3 versiones
            model = registry.get_model(model_name, version)
            stage = model.metadata.stage.value
            click.echo(f"    * v{version} ({stage})")
        if len(versions) > 3:
            click.echo(f"    ... and {len(versions) - 3} more")


@cli.command()
@click.argument('model_name')
@click.option(
    '--registry-path',
    default="./mlpy_models",
    help='Ruta al registry de modelos'
)
@click.option(
    '--version',
    default=None,
    help='Versión del modelo'
)
def model_info(model_name: str, registry_path: str, version: Optional[str]):
    """Muestra información detallada de un modelo.
    
    Ejemplo:
        mlpy-serve model-info iris_classifier
        mlpy-serve model-info iris_classifier --version 1.0.0
    """
    from ..registry import FileSystemRegistry
    
    registry_path = Path(registry_path)
    if not registry_path.exists():
        click.echo(f"Registry path {registry_path} does not exist")
        return
    
    registry = FileSystemRegistry(str(registry_path))
    model = registry.get_model(model_name, version)
    
    if model is None:
        click.echo(f"Model {model_name} (version {version or 'latest'}) not found")
        return
    
    click.echo(f"Model: {model.metadata.name}")
    click.echo(f"Version: {model.metadata.version}")
    click.echo(f"Stage: {model.metadata.stage.value}")
    click.echo(f"Task Type: {model.metadata.task_type}")
    click.echo(f"Created: {model.metadata.created_at}")
    click.echo(f"Author: {model.metadata.author}")
    
    if model.metadata.description:
        click.echo(f"Description: {model.metadata.description}")
    
    if model.metadata.metrics:
        click.echo("Metrics:")
        for metric, value in model.metadata.metrics.items():
            click.echo(f"  - {metric}: {value:.4f}")
    
    if model.metadata.tags:
        click.echo("Tags:")
        for tag, value in model.metadata.tags.items():
            click.echo(f"  - {tag}: {value}")


@cli.command()
@click.argument('model_name')
@click.argument('stage', type=click.Choice(['development', 'staging', 'production', 'archived']))
@click.option(
    '--registry-path',
    default="./mlpy_models",
    help='Ruta al registry de modelos'
)
@click.option(
    '--version',
    default=None,
    help='Versión del modelo'
)
def set_stage(model_name: str, stage: str, registry_path: str, version: Optional[str]):
    """Cambia el stage de un modelo.
    
    Ejemplo:
        mlpy-serve set-stage iris_classifier production
        mlpy-serve set-stage iris_classifier staging --version 2.0.0
    """
    from ..registry import FileSystemRegistry, ModelStage
    
    registry_path = Path(registry_path)
    if not registry_path.exists():
        click.echo(f"Registry path {registry_path} does not exist")
        return
    
    registry = FileSystemRegistry(str(registry_path))
    
    # Obtener el modelo para verificar que existe
    model = registry.get_model(model_name, version)
    if model is None:
        click.echo(f"Model {model_name} (version {version or 'latest'}) not found")
        return
    
    # Actualizar stage
    stage_enum = ModelStage(stage)
    success = registry.update_model_stage(
        model_name,
        model.metadata.version,
        stage_enum
    )
    
    if success:
        click.echo(f"Updated {model_name} v{model.metadata.version} to {stage}")
    else:
        click.echo(f"Failed to update model stage")


@cli.command()
@click.option(
    '--host',
    default="localhost",
    help='Host del servidor'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Puerto del servidor'
)
@click.option(
    '--api-key',
    default=None,
    help='API key si está habilitada la autenticación'
)
def test_connection(host: str, port: int, api_key: Optional[str]):
    """Prueba la conexión con el servidor.
    
    Ejemplo:
        mlpy-serve test-connection
        mlpy-serve test-connection --host myserver.com --port 8080
    """
    from .client import MLPYClient
    
    base_url = f"http://{host}:{port}"
    
    try:
        client = MLPYClient(base_url=base_url, api_key=api_key)
        health = client.health_check()
        
        click.echo(f"✓ Connected to MLPY Model Server at {base_url}")
        click.echo(f"  Status: {health['status']}")
        click.echo(f"  Version: {health['version']}")
        click.echo(f"  Models loaded: {health['models_loaded']}")
        click.echo(f"  Uptime: {health['uptime']:.1f} seconds")
        
        # Listar modelos
        models = client.list_models()
        if models:
            click.echo(f"  Available models: {', '.join(models)}")
        else:
            click.echo("  No models available")
            
    except Exception as e:
        click.echo(f"✗ Failed to connect to server: {str(e)}")
        raise click.ClickException("Connection failed")


if __name__ == "__main__":
    cli()