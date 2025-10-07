"""
Learners basados en Transformers para MLPY.

Este módulo integra modelos transformer (BERT, GPT, RoBERTa) con 
el ecosistema MLPY incluyendo validación automática y explicabilidad.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List, Tuple
import warnings

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForCausalLM, BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel,
        RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel,
        Trainer, TrainingArguments, pipeline
    )
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from ..base import Learner
from ...tasks import Task, TaskClassif, TaskRegr
from ...predictions import PredictionClassif, PredictionRegr
from ...validation.validators import validate_task_data
from ...core.lazy import LazyEvaluationContext
from ...interpretability import create_explanation


if not _HAS_TRANSFORMERS:
    class TransformerLearnerBase:
        """Placeholder cuando transformers no está disponible."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Transformers not available. Install with: "
                "pip install transformers torch"
            )
else:
    
    class MLPYTextDataset(Dataset):
        """Dataset personalizado para textos MLPY."""
        
        def __init__(self, texts, labels, tokenizer, max_length=512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx] if self.labels is not None else 0
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
    
    
    class TransformerLearnerBase(Learner):
        """
        Clase base para learners basados en Transformers.
        """
        
        def __init__(
            self,
            model_name: str = 'bert-base-uncased',
            max_length: int = 512,
            batch_size: int = 16,
            learning_rate: float = 2e-5,
            num_epochs: int = 3,
            warmup_steps: int = 100,
            text_column: str = 'text',
            device: Optional[str] = None,
            **kwargs
        ):
            super().__init__(**kwargs)
            
            self.model_name = model_name
            self.max_length = max_length
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            self.warmup_steps = warmup_steps
            self.text_column = text_column
            
            # Device detection
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
            
            self.tokenizer = None
            self.model = None
            self.trainer = None
            self.training_history = []
        
        def _validate_text_data(self, task):
            """Validación específica para datos de texto."""
            validation = validate_task_data(task.data, target=task.target)
            
            if not validation['valid']:
                from ...core.exceptions import MLPYValidationError
                error_msg = "Text classification failed validation:\\n"
                for error in validation['errors']:
                    error_msg += f"  - {error}\\n"
                error_msg += "\\nFor text data:\\n"
                error_msg += f"  • Ensure '{self.text_column}' column exists\\n"
                error_msg += "  • Check for empty or null text entries\\n"
                error_msg += "  • Consider text preprocessing\\n"
                error_msg += "  • Verify text encoding (UTF-8 recommended)"
                raise MLPYValidationError(error_msg)
            
            # Validación específica de texto
            if self.text_column not in task.data.columns:
                raise MLPYValidationError(
                    f"Text column '{self.text_column}' not found in data. "
                    f"Available columns: {list(task.data.columns)}"
                )
            
            # Verificar textos vacíos
            text_data = task.data[self.text_column]
            empty_texts = text_data.isnull().sum() + (text_data == '').sum()
            if empty_texts > 0:
                warnings.warn(f"Found {empty_texts} empty text entries. These will be replaced with '[EMPTY]'")
        
        def _prepare_texts(self, task):
            """Preparar textos para el modelo."""
            texts = task.data[self.text_column].fillna('[EMPTY]').astype(str)
            
            if hasattr(task, 'target') and task.target:
                labels = task.data[task.target]
                
                # Convertir labels categóricas a números si es necesario
                if isinstance(task, TaskClassif):
                    unique_labels = labels.unique()
                    self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
                    self.reverse_label_map = {idx: label for label, idx in self.label_map.items()}
                    numeric_labels = [self.label_map[label] for label in labels]
                    return texts.tolist(), numeric_labels
                else:
                    return texts.tolist(), labels.tolist()
            else:
                return texts.tolist(), None
        
        def _setup_model_and_tokenizer(self, num_labels=None):
            """Configurar modelo y tokenizer."""
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if num_labels:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels
                )
            else:
                self.model = AutoModel.from_pretrained(self.model_name)
            
            self.model.to(self.device)
        
        def explain(self, X, method='attention', **kwargs):
            """
            Explicar predicciones del modelo transformer.
            
            Parameters:
            -----------
            X : array-like or str
                Datos de entrada (textos)
            method : str
                Método de explicación ('attention', 'gradient', 'lime')
            """
            if method == 'attention':
                return self._explain_attention(X, **kwargs)
            elif method == 'gradient':
                return self._explain_gradient(X, **kwargs)
            else:
                return create_explanation(self, X, method=method, **kwargs)
        
        def _explain_attention(self, texts, **kwargs):
            """Explicación basada en attention weights."""
            if isinstance(texts, str):
                texts = [texts]
            
            self.model.eval()
            explanations = []
            
            with torch.no_grad():
                for text in texts:
                    # Tokenizar
                    inputs = self.tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=self.max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward con attention
                    outputs = self.model(**inputs, output_attentions=True)
                    attentions = outputs.attentions
                    
                    # Promediar attention heads y layers
                    avg_attention = torch.mean(torch.stack(attentions), dim=[0, 1, 2])
                    avg_attention = avg_attention.cpu().numpy()
                    
                    # Mapear a tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    
                    explanation = {
                        'text': text,
                        'tokens': tokens,
                        'attention_weights': avg_attention.tolist(),
                        'important_tokens': [
                            (token, weight) for token, weight in 
                            zip(tokens, avg_attention) if weight > np.mean(avg_attention)
                        ]
                    }
                    explanations.append(explanation)
            
            return {
                'method': 'attention',
                'explanations': explanations
            }


class LearnerBERTClassifier(TransformerLearnerBase):
    """
    BERT Classifier con integración completa MLPY.
    
    Características:
    - Fine-tuning de BERT pre-entrenado
    - Validación automática de texto
    - Explicabilidad con attention
    - Serialización robusta
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
    
    def train(self, task: TaskClassif):
        """Entrenar BERT para clasificación."""
        with LazyEvaluationContext():
            # Validación
            self._validate_text_data(task)
            
            # Preparar datos
            texts, labels = self._prepare_texts(task)
            num_labels = len(set(labels))
            
            # Setup modelo
            self._setup_model_and_tokenizer(num_labels)
            
            # Crear dataset
            dataset = MLPYTextDataset(texts, labels, self.tokenizer, self.max_length)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./bert_results',
                num_train_epochs=self.num_epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                warmup_steps=self.warmup_steps,
                weight_decay=0.01,
                logging_dir='./bert_logs',
                logging_steps=10,
                learning_rate=self.learning_rate,
                save_strategy='no',  # No guardar checkpoints automáticamente
                evaluation_strategy='no'
            )
            
            # Trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            # Entrenar
            print(f"Training BERT classifier on {len(texts)} samples...")
            train_result = self.trainer.train()
            self.training_history.append(train_result)
            
            return self
    
    def predict(self, task: Task):
        """Predicciones con BERT."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preparar textos
        texts, _ = self._prepare_texts(task)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                # Tokenizar
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predicción
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Probabilidades
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probabilities.append(probs.cpu().numpy()[0])
                
                # Predicción final
                pred_idx = torch.argmax(logits, dim=-1).item()
                pred_label = self.reverse_label_map[pred_idx]
                predictions.append(pred_label)
        
        return PredictionClassif(
            task=task,
            learner_id=self.id or "bert_classifier",
            row_ids=list(range(len(texts))),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=predictions,
            prob=np.array(probabilities)
        )


class LearnerBERTRegressor(TransformerLearnerBase):
    """BERT para tareas de regresión."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def train(self, task: TaskRegr):
        """Entrenar BERT para regresión."""
        with LazyEvaluationContext():
            self._validate_text_data(task)
            
            # Para regresión, necesitamos modificar el modelo
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Modelo base + regression head
            self.base_model = AutoModel.from_pretrained(self.model_name)
            self.regression_head = nn.Linear(self.base_model.config.hidden_size, 1)
            
            # Combined model
            class BERTRegressor(nn.Module):
                def __init__(self, base_model, regression_head):
                    super().__init__()
                    self.base_model = base_model
                    self.regression_head = regression_head
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.base_model(input_ids, attention_mask)
                    pooled_output = outputs.pooler_output
                    return self.regression_head(pooled_output)
            
            self.model = BERTRegressor(self.base_model, self.regression_head)
            self.model.to(self.device)
            
            # Preparar datos y entrenar (implementación simplificada)
            texts, labels = self._prepare_texts(task)
            
            print(f"Training BERT regressor on {len(texts)} samples...")
            
            # Training loop básico
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            self.model.train()
            for epoch in range(self.num_epochs):
                total_loss = 0
                for i in range(0, len(texts), self.batch_size):
                    batch_texts = texts[i:i+self.batch_size]
                    batch_labels = labels[i:i+self.batch_size]
                    
                    # Tokenizar batch
                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors='pt',
                        truncation=True,
                        padding=True,
                        max_length=self.max_length
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    targets = torch.tensor(batch_labels, dtype=torch.float).to(self.device)
                    
                    # Forward
                    optimizer.zero_grad()
                    outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                    loss = criterion(outputs.squeeze(), targets)
                    
                    # Backward
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(texts):.4f}")
            
            return self
    
    def predict(self, task: Task):
        """Predicciones de regresión con BERT."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        texts, _ = self._prepare_texts(task)
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                output = self.model(inputs['input_ids'], inputs['attention_mask'])
                predictions.append(output.item())
        
        return PredictionRegr(
            task=task,
            learner_id=self.id or "bert_regressor",
            row_ids=list(range(len(texts))),
            truth=task.truth() if hasattr(task, 'truth') else None,
            response=predictions
        )


class LearnerGPTGenerator(TransformerLearnerBase):
    """
    GPT para generación de texto.
    
    Nota: Este es un ejemplo simplificado. Para uso completo,
    se recomienda usar bibliotecas especializadas.
    """
    
    def __init__(
        self,
        model_name: str = 'gpt2',
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def train(self, task: Task):
        """
        Entrenar GPT (implementación simplificada).
        
        Para fine-tuning completo, usar bibliotecas especializadas.
        """
        print("GPT training - using pre-trained model for generation")
        print("For fine-tuning, consider using specialized frameworks")
        
        # Cargar modelo pre-entrenado
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Agregar pad token si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generar texto a partir de un prompt."""
        if self.model is None:
            raise ValueError("Model not loaded. Call train() first.")
        
        # Configurar parámetros de generación
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        # Tokenizar prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generar
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remover prompt del resultado
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def predict(self, task: Task):
        """
        Para GPT, 'predict' es generar texto.
        
        Asume que la tarea contiene prompts en la columna de texto.
        """
        texts, _ = self._prepare_texts(task)
        
        generated_texts = []
        for prompt in texts:
            generated = self.generate_text(prompt)
            generated_texts.append(generated)
        
        # Retornar como predicción de regresión (texto como string)
        return PredictionRegr(
            task=task,
            learner_id=self.id or "gpt_generator",
            row_ids=list(range(len(texts))),
            truth=None,  # No hay "truth" para generación
            response=generated_texts
        )


class LearnerRoBERTaClassifier(LearnerBERTClassifier):
    """RoBERTa Classifier - versión optimizada de BERT."""
    
    def __init__(self, model_name: str = 'roberta-base', **kwargs):
        super().__init__(model_name=model_name, **kwargs)


class LearnerDistilBERTClassifier(LearnerBERTClassifier):
    """DistilBERT Classifier - versión compacta de BERT."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', **kwargs):
        super().__init__(model_name=model_name, **kwargs)


# Si transformers no está disponible, crear placeholders
if not _HAS_TRANSFORMERS:
    LearnerBERTClassifier = TransformerLearnerBase
    LearnerBERTRegressor = TransformerLearnerBase
    LearnerGPTGenerator = TransformerLearnerBase
    LearnerRoBERTaClassifier = TransformerLearnerBase
    LearnerDistilBERTClassifier = TransformerLearnerBase


__all__ = [
    'LearnerBERTClassifier',
    'LearnerBERTRegressor',
    'LearnerGPTGenerator',
    'LearnerRoBERTaClassifier',
    'LearnerDistilBERTClassifier'
]