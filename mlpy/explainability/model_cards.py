"""
Model Cards Generator
=====================

Automatically generate model documentation cards for transparency.
Based on "Model Cards for Model Reporting" by Mitchell et al.
"""

import json
import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import warnings
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelDetails:
    """Basic model information."""
    name: str
    version: str
    type: str  # e.g., "Random Forest Classifier"
    description: str
    owner: str
    contact: str
    date: str
    license: str = "Proprietary"
    citation: Optional[str] = None
    references: Optional[List[str]] = None


@dataclass
class IntendedUse:
    """Intended use cases and users."""
    primary_uses: List[str]
    primary_users: List[str]
    out_of_scope_uses: List[str]
    

@dataclass
class Factors:
    """Relevant factors for model performance."""
    relevant_factors: List[str]  # e.g., ["age", "geography", "language"]
    evaluation_factors: List[str]  # Factors used in evaluation


@dataclass
class Metrics:
    """Model performance metrics."""
    performance_measures: List[str]  # e.g., ["accuracy", "F1", "AUC"]
    decision_thresholds: Optional[Dict[str, float]] = None
    variation_approaches: Optional[List[str]] = None  # e.g., ["cross-validation"]


@dataclass
class EvaluationData:
    """Evaluation dataset information."""
    dataset: str
    motivation: str
    preprocessing: List[str]


@dataclass
class TrainingData:
    """Training dataset information."""
    dataset: str
    motivation: str
    preprocessing: List[str]


@dataclass
class QuantitativeAnalysis:
    """Quantitative performance results."""
    overall_performance: Dict[str, float]
    intersectional_performance: Optional[Dict[str, Dict[str, float]]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class EthicalConsiderations:
    """Ethical considerations and limitations."""
    ethical_considerations: List[str]
    caveats_recommendations: List[str]
    fairness_assessment: Optional[Dict[str, Any]] = None


class ModelCard:
    """Complete model card documentation."""
    
    def __init__(
        self,
        model_details: ModelDetails,
        intended_use: IntendedUse,
        factors: Factors,
        metrics: Metrics,
        evaluation_data: EvaluationData,
        training_data: TrainingData,
        quantitative_analysis: QuantitativeAnalysis,
        ethical_considerations: EthicalConsiderations
    ):
        """Initialize model card."""
        self.model_details = model_details
        self.intended_use = intended_use
        self.factors = factors
        self.metrics = metrics
        self.evaluation_data = evaluation_data
        self.training_data = training_data
        self.quantitative_analysis = quantitative_analysis
        self.ethical_considerations = ethical_considerations
        self.generated_date = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_details': asdict(self.model_details),
            'intended_use': asdict(self.intended_use),
            'factors': asdict(self.factors),
            'metrics': asdict(self.metrics),
            'evaluation_data': asdict(self.evaluation_data),
            'training_data': asdict(self.training_data),
            'quantitative_analysis': asdict(self.quantitative_analysis),
            'ethical_considerations': asdict(self.ethical_considerations),
            'generated_date': self.generated_date
        }
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export to JSON."""
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_html(self, path: Optional[str] = None) -> str:
        """Export to HTML format."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Card: {name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .metric {{
            display: inline-block;
            background: #ecf0f1;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 5px;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
        .success {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 10px 0;
        }}
        .info {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 10px;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        ul li {{
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }}
        ul li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #3498db;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>📋 Model Card: {name}</h1>
        <p><strong>Version:</strong> {version} | <strong>Generated:</strong> {date}</p>
        
        <h2>📝 Model Details</h2>
        <p><strong>Type:</strong> {type}</p>
        <p><strong>Description:</strong> {description}</p>
        <p><strong>Owner:</strong> {owner} | <strong>Contact:</strong> {contact}</p>
        <p><strong>License:</strong> {license}</p>
        
        <h2>🎯 Intended Use</h2>
        <h3>Primary Uses</h3>
        <ul>{primary_uses}</ul>
        <h3>Primary Users</h3>
        <ul>{primary_users}</ul>
        <div class="warning">
            <strong>⚠️ Out of Scope Uses:</strong>
            <ul>{out_of_scope}</ul>
        </div>
        
        <h2>📊 Performance Metrics</h2>
        <h3>Measures</h3>
        <div>{metrics_list}</div>
        <h3>Overall Performance</h3>
        <table>{performance_table}</table>
        
        <h2>🗂️ Training Data</h2>
        <div class="info">
            <strong>Dataset:</strong> {training_dataset}<br>
            <strong>Motivation:</strong> {training_motivation}<br>
            <strong>Preprocessing:</strong> {training_preprocessing}
        </div>
        
        <h2>🧪 Evaluation Data</h2>
        <div class="info">
            <strong>Dataset:</strong> {eval_dataset}<br>
            <strong>Motivation:</strong> {eval_motivation}<br>
            <strong>Preprocessing:</strong> {eval_preprocessing}
        </div>
        
        <h2>⚖️ Ethical Considerations</h2>
        <h3>Considerations</h3>
        <ul>{ethical}</ul>
        <h3>Recommendations</h3>
        <ul>{recommendations}</ul>
        
        <div class="footer">
            <p>Generated with MLPY Explainability Module | {generated_date}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Format lists
        def format_list(items):
            return '\n'.join([f'<li>{item}</li>' for item in items])
        
        def format_metrics(metrics):
            return ' '.join([f'<span class="metric">{m}</span>' for m in metrics])
        
        def format_table(data):
            rows = []
            for key, value in data.items():
                if isinstance(value, float):
                    rows.append(f'<tr><td>{key}</td><td>{value:.4f}</td></tr>')
                else:
                    rows.append(f'<tr><td>{key}</td><td>{value}</td></tr>')
            return '<table>' + '\n'.join(rows) + '</table>'
        
        # Fill template
        html = html_template.format(
            name=self.model_details.name,
            version=self.model_details.version,
            date=self.model_details.date,
            type=self.model_details.type,
            description=self.model_details.description,
            owner=self.model_details.owner,
            contact=self.model_details.contact,
            license=self.model_details.license,
            primary_uses=format_list(self.intended_use.primary_uses),
            primary_users=format_list(self.intended_use.primary_users),
            out_of_scope=format_list(self.intended_use.out_of_scope_uses),
            metrics_list=format_metrics(self.metrics.performance_measures),
            performance_table=format_table(self.quantitative_analysis.overall_performance),
            training_dataset=self.training_data.dataset,
            training_motivation=self.training_data.motivation,
            training_preprocessing=', '.join(self.training_data.preprocessing),
            eval_dataset=self.evaluation_data.dataset,
            eval_motivation=self.evaluation_data.motivation,
            eval_preprocessing=', '.join(self.evaluation_data.preprocessing),
            ethical=format_list(self.ethical_considerations.ethical_considerations),
            recommendations=format_list(self.ethical_considerations.caveats_recommendations),
            generated_date=self.generated_date
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html
    
    def to_markdown(self, path: Optional[str] = None) -> str:
        """Export to Markdown format."""
        md_template = """# Model Card: {name}

**Version:** {version}  
**Date:** {date}  
**Generated:** {generated_date}

## Model Details

- **Type:** {type}
- **Description:** {description}
- **Owner:** {owner}
- **Contact:** {contact}
- **License:** {license}

## Intended Use

### Primary Uses
{primary_uses}

### Primary Users
{primary_users}

### Out of Scope Uses
{out_of_scope}

## Factors

### Relevant Factors
{relevant_factors}

### Evaluation Factors
{evaluation_factors}

## Metrics

### Performance Measures
{metrics_list}

### Overall Performance
{performance_table}

## Training Data

- **Dataset:** {training_dataset}
- **Motivation:** {training_motivation}
- **Preprocessing:** {training_preprocessing}

## Evaluation Data

- **Dataset:** {eval_dataset}
- **Motivation:** {eval_motivation}
- **Preprocessing:** {eval_preprocessing}

## Ethical Considerations

### Considerations
{ethical}

### Caveats and Recommendations
{recommendations}

---
*Generated with MLPY Explainability Module*
        """
        
        # Format lists
        def format_list(items):
            return '\n'.join([f'- {item}' for item in items])
        
        def format_table(data):
            rows = ['| Metric | Value |', '|--------|-------|']
            for key, value in data.items():
                if isinstance(value, float):
                    rows.append(f'| {key} | {value:.4f} |')
                else:
                    rows.append(f'| {key} | {value} |')
            return '\n'.join(rows)
        
        # Fill template
        md = md_template.format(
            name=self.model_details.name,
            version=self.model_details.version,
            date=self.model_details.date,
            generated_date=self.generated_date,
            type=self.model_details.type,
            description=self.model_details.description,
            owner=self.model_details.owner,
            contact=self.model_details.contact,
            license=self.model_details.license,
            primary_uses=format_list(self.intended_use.primary_uses),
            primary_users=format_list(self.intended_use.primary_users),
            out_of_scope=format_list(self.intended_use.out_of_scope_uses),
            relevant_factors=format_list(self.factors.relevant_factors),
            evaluation_factors=format_list(self.factors.evaluation_factors),
            metrics_list=', '.join(self.metrics.performance_measures),
            performance_table=format_table(self.quantitative_analysis.overall_performance),
            training_dataset=self.training_data.dataset,
            training_motivation=self.training_data.motivation,
            training_preprocessing=', '.join(self.training_data.preprocessing),
            eval_dataset=self.evaluation_data.dataset,
            eval_motivation=self.evaluation_data.motivation,
            eval_preprocessing=', '.join(self.evaluation_data.preprocessing),
            ethical=format_list(self.ethical_considerations.ethical_considerations),
            recommendations=format_list(self.ethical_considerations.caveats_recommendations)
        )
        
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(md)
        
        return md


class ModelCardGenerator:
    """Automatically generate model cards from model and data."""
    
    def __init__(self, model: Any, model_name: str = "Model"):
        """
        Initialize model card generator.
        
        Args:
            model: Trained model
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
    
    def generate(
        self,
        X_train: Optional[Any] = None,
        y_train: Optional[Any] = None,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        intended_uses: Optional[List[str]] = None,
        ethical_considerations: Optional[List[str]] = None,
        owner: str = "Unknown",
        contact: str = "Unknown"
    ) -> ModelCard:
        """
        Generate model card automatically.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            performance_metrics: Model performance metrics
            intended_uses: List of intended use cases
            ethical_considerations: List of ethical considerations
            owner: Model owner
            contact: Contact information
            
        Returns:
            ModelCard object
        """
        # Model details
        model_details = ModelDetails(
            name=self.model_name,
            version="1.0.0",
            type=self.model.__class__.__name__,
            description=f"Machine learning model: {self.model.__class__.__name__}",
            owner=owner,
            contact=contact,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            license="Proprietary"
        )
        
        # Intended use
        if intended_uses is None:
            intended_uses = ["General prediction task"]
        
        intended_use = IntendedUse(
            primary_uses=intended_uses,
            primary_users=["Data scientists", "ML engineers"],
            out_of_scope_uses=["Production use without validation", 
                              "Decision making without human oversight"]
        )
        
        # Factors
        factors = Factors(
            relevant_factors=self._extract_feature_names(X_train) if X_train is not None else [],
            evaluation_factors=["Overall performance", "Per-class performance"]
        )
        
        # Metrics
        if performance_metrics:
            metric_names = list(performance_metrics.keys())
        else:
            metric_names = ["accuracy", "precision", "recall"]
        
        metrics = Metrics(
            performance_measures=metric_names,
            variation_approaches=["cross-validation", "bootstrap"]
        )
        
        # Training data
        training_data = TrainingData(
            dataset="Training dataset",
            motivation="Model training",
            preprocessing=["Standardization", "Missing value imputation"]
        )
        
        # Evaluation data
        evaluation_data = EvaluationData(
            dataset="Test dataset",
            motivation="Model evaluation",
            preprocessing=["Same as training"]
        )
        
        # Quantitative analysis
        if performance_metrics is None:
            performance_metrics = {"accuracy": 0.0}
        
        quantitative_analysis = QuantitativeAnalysis(
            overall_performance=performance_metrics
        )
        
        # Ethical considerations
        if ethical_considerations is None:
            ethical_considerations = [
                "Model predictions should be reviewed by domain experts",
                "Regular monitoring for performance degradation required",
                "Potential biases in training data not fully assessed"
            ]
        
        ethical = EthicalConsiderations(
            ethical_considerations=ethical_considerations,
            caveats_recommendations=[
                "Use with caution in high-stakes decisions",
                "Regularly retrain with updated data",
                "Monitor for distribution shift"
            ]
        )
        
        return ModelCard(
            model_details=model_details,
            intended_use=intended_use,
            factors=factors,
            metrics=metrics,
            evaluation_data=evaluation_data,
            training_data=training_data,
            quantitative_analysis=quantitative_analysis,
            ethical_considerations=ethical
        )
    
    def _extract_feature_names(self, X: Any) -> List[str]:
        """Extract feature names from data."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X.columns.tolist()[:10]  # Top 10 features
        except:
            pass
        
        return ["Feature 1", "Feature 2", "Feature 3"]