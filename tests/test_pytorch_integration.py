"""
Tests para la integración de PyTorch en MLPY.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil

# Importar componentes de MLPY
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.learners.pytorch import (
    LearnerPyTorch, LearnerPyTorchClassif, LearnerPyTorchRegr,
    MLPYDataset, MLPYDataLoader, create_data_loaders
)
from mlpy.learners.pytorch.models import (
    MLPNet, CNNClassifier, TransformerModel, AutoEncoder, LSTMClassifier
)
from mlpy.learners.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    GradientClipping, MetricsLogger
)
from mlpy.learners.pytorch.utils import (
    get_device, count_parameters, freeze_layers, unfreeze_layers,
    save_checkpoint, load_checkpoint, model_summary
)
from mlpy.learners.pytorch.pretrained import (
    load_pretrained_model, get_available_models, finetune_model
)


# Fixtures
@pytest.fixture
def classification_data():
    """Datos de clasificación."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df


@pytest.fixture
def regression_data():
    """Datos de regresión."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y.astype(np.float32)
    
    return df


@pytest.fixture
def temp_dir():
    """Directorio temporal para tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Tests para datasets
def test_mlpy_dataset_classification(classification_data):
    """Test del dataset para clasificación."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear dataset
    dataset = MLPYDataset(task, training=True)
    
    # Verificar tamaño
    assert len(dataset) == len(classification_data)
    
    # Obtener un item
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10,)  # n_features
    assert y.dtype == torch.int64  # Para clasificación


def test_mlpy_dataset_regression(regression_data):
    """Test del dataset para regresión."""
    task = TaskRegr(
        id="test_regr",
        data=regression_data,
        target_col="target"
    )
    
    # Crear dataset
    dataset = MLPYDataset(task, training=True)
    
    # Verificar tamaño
    assert len(dataset) == len(regression_data)
    
    # Obtener un item
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (10,)  # n_features
    assert y.dtype == torch.float32  # Para regresión


def test_create_data_loaders(classification_data):
    """Test de creación de DataLoaders."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear DataLoaders
    train_loader, val_loader = create_data_loaders(
        task,
        batch_size=16,
        validation_split=0.2,
        shuffle=True
    )
    
    # Verificar train loader
    assert train_loader is not None
    assert len(train_loader) > 0
    
    # Verificar val loader
    assert val_loader is not None
    assert len(val_loader) > 0
    
    # Verificar batch
    batch = next(iter(train_loader))
    assert len(batch) == 2  # (X, y)
    X_batch, y_batch = batch
    assert X_batch.shape[0] <= 16  # batch_size
    assert X_batch.shape[1] == 10  # n_features


# Tests para learners base
def test_learner_pytorch_classif_train(classification_data):
    """Test de entrenamiento para clasificación."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear learner
    learner = LearnerPyTorchClassif(
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que el modelo fue creado
    assert learner.model is not None
    assert isinstance(learner.model, nn.Module)
    
    # Hacer predicciones
    predictions = learner.predict(task)
    assert predictions is not None
    assert len(predictions.response) == len(classification_data)


def test_learner_pytorch_regr_train(regression_data):
    """Test de entrenamiento para regresión."""
    task = TaskRegr(
        id="test_regr",
        data=regression_data,
        target_col="target"
    )
    
    # Crear learner
    learner = LearnerPyTorchRegr(
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que el modelo fue creado
    assert learner.model is not None
    assert isinstance(learner.model, nn.Module)
    
    # Hacer predicciones
    predictions = learner.predict(task)
    assert predictions is not None
    assert len(predictions.response) == len(regression_data)


# Tests para modelos
def test_mlp_net():
    """Test del modelo MLP."""
    model = MLPNet(
        input_dim=10,
        hidden_dims=[64, 32],
        output_dim=3,
        activation='relu',
        dropout=0.2,
        batch_norm=True
    )
    
    # Test forward pass
    x = torch.randn(16, 10)
    output = model(x)
    
    assert output.shape == (16, 3)
    assert not torch.isnan(output).any()


def test_cnn_classifier():
    """Test del modelo CNN."""
    model = CNNClassifier(
        input_channels=3,
        num_classes=10,
        conv_channels=[32, 64],
        kernel_sizes=[3, 3],
        pool_sizes=[2, 2],
        fc_dims=[128],
        dropout=0.5
    )
    
    # Test forward pass
    x = torch.randn(8, 3, 32, 32)  # Batch de imágenes pequeñas
    output = model(x)
    
    assert output.shape == (8, 10)
    assert not torch.isnan(output).any()


def test_transformer_model():
    """Test del modelo Transformer."""
    model = TransformerModel(
        input_dim=20,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=5
    )
    
    # Test forward pass
    x = torch.randn(4, 50, 20)  # [batch, seq_len, input_dim]
    output = model(x)
    
    assert output.shape == (4, 5)  # [batch, num_classes]
    assert not torch.isnan(output).any()


def test_autoencoder():
    """Test del modelo AutoEncoder."""
    model = AutoEncoder(
        input_dim=20,
        encoding_dims=[16, 8],
        latent_dim=4,
        activation='relu'
    )
    
    # Test forward pass
    x = torch.randn(8, 20)
    reconstruction, latent = model(x)
    
    assert reconstruction.shape == (8, 20)
    assert latent.shape == (8, 4)
    assert not torch.isnan(reconstruction).any()
    assert not torch.isnan(latent).any()


def test_lstm_classifier():
    """Test del modelo LSTM."""
    model = LSTMClassifier(
        input_dim=10,
        hidden_dim=32,
        num_layers=2,
        num_classes=3,
        dropout=0.2,
        bidirectional=True
    )
    
    # Test forward pass
    x = torch.randn(4, 25, 10)  # [batch, seq_len, input_dim]
    output = model(x)
    
    assert output.shape == (4, 3)  # [batch, num_classes]
    assert not torch.isnan(output).any()


# Tests para callbacks
def test_early_stopping_callback(classification_data):
    """Test del callback EarlyStopping."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear learner con early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.001,
        mode='min',
        restore_best=True,
        verbose=False
    )
    
    learner = LearnerPyTorchClassif(
        epochs=10,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que se detuvo antes de las 10 épocas
    assert learner.current_epoch < 10


def test_model_checkpoint_callback(classification_data, temp_dir):
    """Test del callback ModelCheckpoint."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    checkpoint_path = Path(temp_dir) / "model_epoch_{epoch}.pt"
    
    # Crear learner con checkpoint
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='train_loss',
        save_best_only=False,
        save_freq=1,
        verbose=False
    )
    
    learner = LearnerPyTorchClassif(
        epochs=2,
        batch_size=16,
        callbacks=[checkpoint],
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que se guardaron los checkpoints
    assert (Path(temp_dir) / "model_epoch_0.pt").exists()
    assert (Path(temp_dir) / "model_epoch_1.pt").exists()


def test_learning_rate_scheduler_callback(classification_data):
    """Test del callback LearningRateScheduler."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear learner con scheduler
    scheduler = LearningRateScheduler(
        schedule='exponential',
        gamma=0.9,
        verbose=False
    )
    
    learner = LearnerPyTorchClassif(
        epochs=3,
        batch_size=16,
        learning_rate=0.1,
        callbacks=[scheduler],
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que el learning rate cambió
    final_lr = learner.optimizer.param_groups[0]['lr']
    assert final_lr < 0.1  # Debe haber disminuido


def test_gradient_clipping_callback(classification_data):
    """Test del callback GradientClipping."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    # Crear learner con gradient clipping
    grad_clip = GradientClipping(max_norm=1.0, verbose=False)
    
    learner = LearnerPyTorchClassif(
        epochs=2,
        batch_size=16,
        callbacks=[grad_clip],
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # El test pasa si no hay errores durante el entrenamiento
    assert True


def test_metrics_logger_callback(classification_data, temp_dir):
    """Test del callback MetricsLogger."""
    task = TaskClassif(
        id="test_classif",
        data=classification_data,
        target_col="target"
    )
    
    metrics_path = Path(temp_dir) / "metrics.json"
    
    # Crear learner con logger
    logger = MetricsLogger(filepath=str(metrics_path))
    
    learner = LearnerPyTorchClassif(
        epochs=2,
        batch_size=16,
        callbacks=[logger],
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Verificar que se guardaron las métricas
    assert metrics_path.exists()
    
    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    assert len(metrics) == 2  # 2 épocas
    assert 'epoch' in metrics[0]
    assert 'train_loss' in metrics[0]


# Tests para utilidades
def test_get_device():
    """Test de detección de dispositivo."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ['cpu', 'cuda', 'mps']
    
    # Test especificando CPU
    device = get_device('cpu')
    assert device.type == 'cpu'


def test_count_parameters():
    """Test de conteo de parámetros."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Contar todos los parámetros
    total_params = count_parameters(model, trainable_only=False)
    assert total_params == 10*20 + 20 + 20*5 + 5  # weights + biases
    
    # Contar solo entrenables
    trainable_params = count_parameters(model, trainable_only=True)
    assert trainable_params == total_params  # Todos son entrenables por defecto


def test_freeze_unfreeze_layers():
    """Test de congelación/descongelación de capas."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Congelar todas las capas
    freeze_layers(model)
    trainable = count_parameters(model, trainable_only=True)
    assert trainable == 0
    
    # Descongelar todas las capas
    unfreeze_layers(model)
    trainable = count_parameters(model, trainable_only=True)
    assert trainable > 0


def test_save_load_checkpoint(temp_dir):
    """Test de guardado/carga de checkpoints."""
    # Crear modelo y optimizador
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint_path = Path(temp_dir) / "checkpoint.pt"
    
    # Guardar checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        path=checkpoint_path,
        metrics={'loss': 0.5}
    )
    
    assert checkpoint_path.exists()
    
    # Cargar checkpoint
    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer
    )
    
    assert checkpoint['epoch'] == 5
    assert checkpoint['metrics']['loss'] == 0.5


def test_model_summary():
    """Test del resumen del modelo."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    summary = model_summary(model)
    
    assert isinstance(summary, str)
    assert "Total parameters" in summary
    assert "Trainable parameters" in summary
    assert "Layers:" in summary


# Tests para modelos pre-entrenados
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere GPU")
def test_load_pretrained_model():
    """Test de carga de modelo pre-entrenado."""
    try:
        # Intentar cargar ResNet18
        model = load_pretrained_model(
            'resnet18',
            num_classes=10,
            pretrained=False,  # No descargar pesos para test rápido
            freeze_backbone=True
        )
        
        assert isinstance(model, nn.Module)
        
        # Verificar que el backbone está congelado
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        assert trainable < total
    except ImportError:
        pytest.skip("torchvision no disponible")


def test_get_available_models():
    """Test de obtención de modelos disponibles."""
    models = get_available_models()
    
    assert isinstance(models, dict)
    assert 'vision_classification' in models
    assert 'nlp_classification' in models
    assert len(models['vision_classification']) > 0


def test_finetune_model():
    """Test de configuración para fine-tuning."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Linear(20, 10),
        nn.Linear(10, 5)
    )
    
    # Congelar todo primero
    freeze_layers(model)
    
    # Fine-tune últimas 2 capas
    model, param_groups = finetune_model(
        model,
        layers_to_unfreeze=2
    )
    
    # Verificar que hay parámetros entrenables
    trainable = count_parameters(model, trainable_only=True)
    assert trainable > 0
    
    # Verificar grupos de parámetros
    assert len(param_groups) > 0


# Test de integración completa
def test_full_integration_pipeline(classification_data, temp_dir):
    """Test de pipeline completo con PyTorch."""
    # Crear tarea
    task = TaskClassif(
        id="test_full",
        data=classification_data,
        target_col="target"
    )
    
    # Crear modelo personalizado
    model = MLPNet(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=3,
        activation='relu',
        dropout=0.2
    )
    
    # Crear callbacks
    callbacks = [
        EarlyStopping(patience=3, verbose=False),
        ModelCheckpoint(
            filepath=str(Path(temp_dir) / "best_model.pt"),
            save_best_only=True,
            verbose=False
        ),
        LearningRateScheduler(schedule='step', step_size=2, gamma=0.5, verbose=False)
    ]
    
    # Crear learner
    learner = LearnerPyTorchClassif(
        model=model,
        epochs=5,
        batch_size=16,
        learning_rate=0.01,
        callbacks=callbacks,
        verbose=0
    )
    
    # Entrenar
    learner.train(task)
    
    # Hacer predicciones
    predictions = learner.predict(task)
    
    # Verificar predicciones
    assert predictions is not None
    assert len(predictions.response) == len(classification_data)
    assert predictions.prob is not None
    assert predictions.prob.shape == (len(classification_data), 3)
    
    # Guardar modelo
    model_path = Path(temp_dir) / "final_model.pt"
    learner.save_model(model_path)
    assert model_path.exists()
    
    # Cargar modelo en nuevo learner
    new_learner = LearnerPyTorchClassif(model=model)
    new_learner.load_model(model_path)
    
    # Hacer predicciones con modelo cargado
    new_predictions = new_learner.predict(task)
    
    # Las predicciones deberían ser idénticas
    np.testing.assert_array_equal(predictions.response, new_predictions.response)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])