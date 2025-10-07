# 🛒 Caso de Uso: Predicción de Churn en E-commerce

## ShopSmart Online - Reteniendo Clientes Valiosos

---

## 📋 RESUMEN EJECUTIVO

**Empresa:** ShopSmart Online (E-commerce de moda)  
**Problema:** Pérdida silenciosa de clientes valiosos (25% churn anual)  
**Solución:** Sistema de early warning con MLPY  
**Resultado:** +25% retención, +$1.2M ingresos anuales  
**ROI:** 300% en 6 meses  

---

## 🏢 CONTEXTO DE NEGOCIO

### La Empresa
ShopSmart es una plataforma de e-commerce de moda con:
- 50,000 clientes activos
- $10M facturación anual
- Ticket promedio: $75
- Problema: 25% de clientes abandonan sin previo aviso

### El Problema
```
📊 Situación Inicial:
- Churn rate: 25% anual
- Costo adquisición: $50 por cliente
- Valor lifetime: $300 por cliente
- Pérdida anual: $3.75M por churn

❌ Síntomas:
- Clientes valiosos cancelan inesperadamente  
- No hay sistema de alertas tempranas
- Marketing reactivo, no predictivo
- Pérdida de market share vs competencia
```

### Oportunidad
Con un sistema predictivo, podrían:
- Identificar clientes en riesgo 90 días antes
- Crear campañas de retención personalizadas
- Reducir churn del 25% al 18%
- Generar $1.2M adicionales anuales

---

## 🔍 ANÁLISIS DE DATOS

### Datos Disponibles

```python
# Fuentes de datos ShopSmart
data_sources = {
    'transaccional': {
        'tabla': 'transactions',
        'registros': 500000,
        'periodo': '2 años',
        'campos': [
            'customer_id', 'order_date', 'order_value', 
            'products', 'category', 'discount_used'
        ]
    },
    'comportamiento': {
        'tabla': 'user_activity', 
        'registros': 2000000,
        'periodo': '2 años',
        'campos': [
            'customer_id', 'session_date', 'pages_viewed',
            'time_on_site', 'cart_abandonment', 'search_terms'
        ]
    },
    'demografico': {
        'tabla': 'customers',
        'registros': 50000,
        'campos': [
            'customer_id', 'age', 'gender', 'location',
            'signup_date', 'preferred_category', 'loyalty_tier'
        ]
    },
    'soporte': {
        'tabla': 'support_tickets',
        'registros': 75000,
        'campos': [
            'customer_id', 'ticket_date', 'issue_type',
            'resolution_time', 'satisfaction_score'
        ]
    }
}
```

### Feature Engineering Planificado

```python
# Features a crear
features_engineering = {
    'recency': [
        'days_since_last_purchase',
        'days_since_last_visit', 
        'days_since_signup'
    ],
    'frequency': [
        'orders_last_30d',
        'orders_last_90d',
        'avg_orders_per_month'
    ],
    'monetary': [
        'total_spend_lifetime',
        'avg_order_value',
        'spend_last_90d'
    ],
    'engagement': [
        'email_open_rate',
        'website_sessions_last_30d',
        'avg_time_on_site'
    ],
    'behavior': [
        'preferred_category',
        'discount_sensitivity',
        'cart_abandonment_rate'
    ],
    'satisfaction': [
        'support_tickets_count',
        'avg_satisfaction_score',
        'days_since_last_complaint'
    ]
}
```

---

## 🛠 IMPLEMENTACIÓN CON MLPY

### Setup del Proyecto

```python
# requirements.txt
"""
mlpy-framework[full]>=2.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
optuna>=3.0.0
plotly>=5.0.0
shap>=0.40.0
"""

# Estructura del proyecto
"""
shopsmart_churn/
├── data/
│   ├── raw/               # Datos originales
│   ├── processed/         # Datos procesados
│   └── features/          # Features engineered
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── modeling.py
├── models/                # Modelos serializados
├── reports/               # Reportes y dashboards
└── config/               # Configuraciones
"""
```

### Paso 1: Carga y Validación de Datos

```python
# src/data_processing.py
import pandas as pd
import numpy as np
from mlpy.validation import validate_task_data
from datetime import datetime, timedelta

def load_and_prepare_data():
    """Carga y prepara datos de múltiples fuentes"""
    
    print("🔄 Cargando datos de múltiples fuentes...")
    
    # Simular carga desde bases de datos
    # En producción: connection strings a PostgreSQL/MySQL
    
    # 1. Datos transaccionales
    transactions = pd.read_sql("""
        SELECT customer_id, order_date, order_value, 
               category, discount_used
        FROM transactions 
        WHERE order_date >= '2022-01-01'
    """, connection)
    
    # 2. Actividad web
    activity = pd.read_sql("""
        SELECT customer_id, session_date, pages_viewed,
               time_on_site, cart_abandonment
        FROM user_activity
        WHERE session_date >= '2022-01-01'
    """, connection)
    
    # 3. Datos demográficos
    customers = pd.read_sql("""
        SELECT customer_id, age, gender, location,
               signup_date, loyalty_tier
        FROM customers
    """, connection)
    
    # 4. Soporte al cliente
    support = pd.read_sql("""
        SELECT customer_id, ticket_date, issue_type,
               satisfaction_score
        FROM support_tickets
        WHERE ticket_date >= '2022-01-01'
    """, connection)
    
    print(f"✅ Datos cargados:")
    print(f"  - Transacciones: {len(transactions):,}")
    print(f"  - Actividad web: {len(activity):,}")
    print(f"  - Clientes: {len(customers):,}")
    print(f"  - Tickets soporte: {len(support):,}")
    
    return transactions, activity, customers, support

def create_churn_labels(transactions, reference_date=None):
    """Crear labels de churn basado en actividad"""
    
    if reference_date is None:
        reference_date = datetime.now()
    
    # Definir churn: no compras en últimos 90 días
    churn_threshold = reference_date - timedelta(days=90)
    
    # Última compra por cliente
    last_purchase = transactions.groupby('customer_id')['order_date'].max()
    
    # Crear labels
    churn_labels = (last_purchase < churn_threshold).astype(int)
    
    print(f"📊 Labels de churn creados:")
    print(f"  - Clientes activos: {(churn_labels == 0).sum():,}")
    print(f"  - Clientes churn: {(churn_labels == 1).sum():,}")
    print(f"  - Tasa de churn: {churn_labels.mean():.2%}")
    
    return churn_labels

# Ejecutar
transactions, activity, customers, support = load_and_prepare_data()
churn_labels = create_churn_labels(transactions)
```

### Paso 2: Feature Engineering Avanzado

```python
# src/feature_engineering.py
from mlpy.lazy import ComputationGraph, ComputationNode

def create_features_pipeline():
    """Pipeline de feature engineering con lazy evaluation"""
    
    print("⚡ Creando pipeline de features con lazy evaluation...")
    
    graph = ComputationGraph()
    
    # Nodo 1: Features RFM (Recency, Frequency, Monetary)
    def compute_rfm_features(transactions):
        print("  [LAZY] Computando features RFM...")
        
        reference_date = transactions['order_date'].max()
        
        rfm = transactions.groupby('customer_id').agg({
            'order_date': [
                lambda x: (reference_date - x.max()).days,  # Recency
                'count'  # Frequency
            ],
            'order_value': [
                'sum',   # Monetary total
                'mean'   # Monetary promedio
            ]
        }).round(2)
        
        # Aplanar columnas
        rfm.columns = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg']
        
        return rfm
    
    rfm_node = ComputationNode(
        id="rfm_features",
        operation="compute_rfm",
        func=lambda: compute_rfm_features(transactions)
    )
    graph.add_node(rfm_node)
    
    # Nodo 2: Features de engagement
    def compute_engagement_features(activity):
        print("  [LAZY] Computando features de engagement...")
        
        engagement = activity.groupby('customer_id').agg({
            'session_date': 'count',  # Número de sesiones
            'pages_viewed': 'mean',   # Páginas promedio por sesión
            'time_on_site': 'mean',   # Tiempo promedio en sitio
            'cart_abandonment': 'mean'  # Tasa de abandono de carrito
        }).round(3)
        
        engagement.columns = [
            'num_sessions', 'avg_pages_viewed', 
            'avg_time_on_site', 'cart_abandonment_rate'
        ]
        
        return engagement
    
    engagement_node = ComputationNode(
        id="engagement_features",
        operation="compute_engagement", 
        func=lambda: compute_engagement_features(activity)
    )
    graph.add_node(engagement_node)
    
    # Nodo 3: Features de soporte
    def compute_support_features(support):
        print("  [LAZY] Computando features de soporte...")
        
        support_stats = support.groupby('customer_id').agg({
            'ticket_date': 'count',        # Número de tickets
            'satisfaction_score': 'mean'   # Satisfacción promedio
        }).round(2)
        
        support_stats.columns = ['num_tickets', 'avg_satisfaction']
        
        # Rellenar NaN para clientes sin tickets
        support_stats['num_tickets'] = support_stats['num_tickets'].fillna(0)
        support_stats['avg_satisfaction'] = support_stats['avg_satisfaction'].fillna(5.0)
        
        return support_stats
    
    support_node = ComputationNode(
        id="support_features",
        operation="compute_support",
        func=lambda: compute_support_features(support)
    )
    graph.add_node(support_node)
    
    # Nodo 4: Consolidar todas las features
    def merge_all_features():
        print("  [LAZY] Consolidando todas las features...")
        
        rfm = graph.nodes["rfm_features"].result
        engagement = graph.nodes["engagement_features"].result
        support_feat = graph.nodes["support_features"].result
        
        # Merge demographics
        demo_features = customers.set_index('customer_id')[['age', 'gender', 'loyalty_tier']]
        
        # Combinar todas las features
        all_features = rfm.join([engagement, support_feat, demo_features], how='inner')
        
        # Encoding categóricas
        all_features = pd.get_dummies(all_features, columns=['gender', 'loyalty_tier'])
        
        # Rellenar NaN si existen
        all_features = all_features.fillna(all_features.median())
        
        return all_features
    
    merge_node = ComputationNode(
        id="merge_features",
        operation="merge_all",
        func=merge_all_features,
        dependencies=["rfm_features", "engagement_features", "support_features"]
    )
    graph.add_node(merge_node)
    
    return graph

# Crear y ejecutar pipeline
print("🚀 Ejecutando pipeline de feature engineering...")
features_graph = create_features_pipeline()
features_graph.optimize()
results = features_graph.execute()

features_df = results["merge_features"]
print(f"\n✅ Features creadas: {features_df.shape}")
print(f"Columnas: {list(features_df.columns)}")
```

### Paso 3: Validación Inteligente y Creación de Task

```python
# Preparar dataset final para ML
def prepare_ml_dataset(features_df, churn_labels):
    """Prepara dataset final con validación MLPY"""
    
    print("🔍 Preparando dataset para ML...")
    
    # Combinar features con labels
    ml_dataset = features_df.join(churn_labels.rename('churn'), how='inner')
    
    print(f"Dataset combinado: {ml_dataset.shape}")
    print(f"Distribución de churn: {ml_dataset['churn'].value_counts()}")
    
    return ml_dataset

# Preparar dataset
ml_dataset = prepare_ml_dataset(features_df, churn_labels)

# Validación automática con MLPY
from mlpy.validation import validate_task_data, ValidatedTask

print("\n🔍 Validando datos con MLPY...")
validation = validate_task_data(ml_dataset, target='churn')

if validation['valid']:
    print("✅ Datos válidos para ML")
else:
    print("❌ Problemas encontrados:")
    for error in validation['errors']:
        print(f"  - {error}")

if validation['warnings']:
    print("\n⚠️ Advertencias:")
    for warning in validation['warnings']:
        print(f"  - {warning}")

# Crear tarea MLPY validada
print("\n📝 Creando tarea de clasificación...")
task = ValidatedTask(
    data=ml_dataset,
    target='churn',
    task_type='classif',
    id='shopsmart_churn_prediction'
)

print(f"✅ Tarea creada:")
print(f"  - ID: {task.task.id}")
print(f"  - Tipo: {task.task.task_type}")
print(f"  - Features: {task.task.n_features}")
print(f"  - Muestras: {task.task.n_obs}")
print(f"  - Clases: {task.task.n_classes}")
```

### Paso 4: AutoML y Optimización de Modelos

```python
# src/modeling.py
from mlpy.automl import SimpleAutoML
from mlpy.visualization.dashboard import create_dashboard, TrainingMetrics
from sklearn.model_selection import train_test_split
import time

def run_automl_experiment(task):
    """Ejecuta experimento de AutoML con dashboard"""
    
    print("🤖 Iniciando experimento de AutoML...")
    
    # Crear dashboard para monitoreo
    dashboard = create_dashboard(
        title="ShopSmart Churn Prediction - AutoML",
        auto_open=False
    )
    
    # Split train/validation/test
    train_idx, temp_idx = train_test_split(
        range(task.task.n_obs), 
        test_size=0.4, 
        random_state=42,
        stratify=task.task.y
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=task.task.y.iloc[temp_idx]
    )
    
    task_train = task.task.subset(train_idx)
    task_val = task.task.subset(val_idx) 
    task_test = task.task.subset(test_idx)
    
    print(f"División de datos:")
    print(f"  - Train: {len(train_idx)} ({len(train_idx)/task.task.n_obs:.1%})")
    print(f"  - Validation: {len(val_idx)} ({len(val_idx)/task.task.n_obs:.1%})")
    print(f"  - Test: {len(test_idx)} ({len(test_idx)/task.task.n_obs:.1%})")
    
    # Modelos a probar
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models_to_test = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for model_name, model in models_to_test.items():
        print(f"\n🔄 Entrenando {model_name}...")
        
        start_time = time.time()
        
        # Crear learner MLPY
        from mlpy.learners import LearnerClassifSklearn
        learner = LearnerClassifSklearn(estimator=model)
        
        # Entrenar
        learner.train(task_train)
        
        # Evaluar en validación
        predictions = learner.predict(task_val)
        
        # Métricas
        from mlpy.measures import MeasureAccuracy, MeasurePrecision, MeasureRecall, MeasureF1
        
        accuracy = MeasureAccuracy().score(predictions)
        precision = MeasurePrecision().score(predictions)
        recall = MeasureRecall().score(predictions)
        f1 = MeasureF1().score(predictions)
        
        train_time = time.time() - start_time
        
        # Registrar en dashboard
        model_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': train_time
        }
        
        dashboard.log_model(model_name, model_metrics)
        results[model_name] = {
            'learner': learner,
            'metrics': model_metrics
        }
        
        print(f"  ✅ {model_name}:")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     F1-Score: {f1:.4f}")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        
        # Actualizar mejor modelo
        if f1 > best_score:  # Usamos F1 para datos desbalanceados
            best_score = f1
            best_model = model_name
    
    print(f"\n🏆 Mejor modelo: {best_model} (F1: {best_score:.4f})")
    
    return results, best_model, dashboard, (task_train, task_val, task_test)

# Ejecutar AutoML
results, best_model, dashboard, data_splits = run_automl_experiment(task)
task_train, task_val, task_test = data_splits
```

### Paso 5: Explicabilidad y Feature Importance

```python
def analyze_model_explanations(best_learner, task_test):
    """Analiza explicabilidad del mejor modelo"""
    
    print("🔍 Analizando explicabilidad del modelo...")
    
    # Feature importance nativa del modelo
    if hasattr(best_learner.estimator, 'feature_importances_'):
        importance = best_learner.estimator.feature_importances_
        feature_names = task_test.X.columns
        
        feature_importance = dict(zip(feature_names, importance))
        
        # Top 10 features más importantes
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n📊 Top 10 Features Más Importantes:")
        for i, (feat, imp) in enumerate(top_features, 1):
            bar = '█' * int(imp * 50)
            print(f"  {i:2}. {feat:25} {bar} {imp:.4f}")
        
        # Registrar en dashboard
        dashboard.log_feature_importance(feature_importance)
    
    # Análisis con SHAP (si está disponible)
    try:
        import shap
        
        print("\n🔬 Análisis SHAP...")
        
        # Crear explainer
        explainer = shap.TreeExplainer(best_learner.estimator)
        
        # Calcular SHAP values para muestra del test set
        X_sample = task_test.X.iloc[:100]  # Muestra para velocidad
        shap_values = explainer.shap_values(X_sample)
        
        # Para clasificación binaria, tomar clase positiva
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Feature importance promedio según SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_features = dict(zip(X_sample.columns, shap_importance))
        
        print("\nTop 5 Features según SHAP:")
        for feat, imp in sorted(shap_features.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat:25} {imp:.4f}")
            
    except ImportError:
        print("⚠️ SHAP no disponible. Instala con: pip install shap")
    
    # Insights de negocio
    print("\n💡 INSIGHTS DE NEGOCIO:")
    
    business_insights = {
        'recency_days': "Clientes que no compran recientemente tienen alto riesgo",
        'frequency': "Frecuencia de compra baja indica posible churn",
        'monetary_avg': "Valor promedio de orden refleja engagement",
        'avg_satisfaction': "Baja satisfacción predice abandono",
        'num_tickets': "Muchos tickets de soporte son señal de problemas",
        'cart_abandonment_rate': "Alto abandono de carrito indica fricción"
    }
    
    for feat, imp in top_features[:5]:
        if feat in business_insights:
            print(f"  📈 {feat}: {business_insights[feat]}")

# Analizar mejor modelo
best_learner = results[best_model]['learner']
analyze_model_explanations(best_learner, task_test)
```

### Paso 6: Evaluación Final y Serialización

```python
def final_evaluation_and_save(best_learner, task_test, model_name):
    """Evaluación final y guardado del modelo"""
    
    print("\n📊 EVALUACIÓN FINAL EN TEST SET")
    print("="*50)
    
    # Predicciones finales
    final_predictions = best_learner.predict(task_test)
    
    # Métricas completas
    from mlpy.measures import MeasureAccuracy, MeasurePrecision, MeasureRecall, MeasureF1
    from sklearn.metrics import classification_report, confusion_matrix
    
    accuracy = MeasureAccuracy().score(final_predictions)
    precision = MeasurePrecision().score(final_predictions)
    recall = MeasureRecall().score(final_predictions)
    f1 = MeasureF1().score(final_predictions)
    
    print(f"Métricas finales:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Matriz de confusión
    y_true = task_test.y.values
    y_pred = final_predictions.response
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nMatriz de Confusión:")
    print(f"  Predicho:    No Churn  Churn")
    print(f"  No Churn:    {cm[0,0]:8d}  {cm[0,1]:5d}")
    print(f"  Churn:       {cm[1,0]:8d}  {cm[1,1]:5d}")
    
    # Análisis de negocio
    print(f"\n💼 IMPACTO DE NEGOCIO:")
    
    total_customers = len(y_true)
    true_churn = (y_true == 1).sum()
    predicted_churn = (y_pred == 1).sum()
    correctly_identified = cm[1,1]  # True positives
    
    # Suponiendo intervención exitosa en 70% de casos identificados
    intervention_success_rate = 0.70
    customers_saved = correctly_identified * intervention_success_rate
    
    # Valor económico
    customer_ltv = 300  # Valor de vida promedio del cliente
    revenue_saved = customers_saved * customer_ltv
    
    print(f"  Clientes en riesgo detectados: {predicted_churn}")
    print(f"  Clientes correctamente identificados: {correctly_identified}")
    print(f"  Clientes potencialmente salvados: {customers_saved:.0f}")
    print(f"  Ingresos potencialmente salvados: ${revenue_saved:,.0f}")
    
    # Guardar modelo con serialización robusta
    from mlpy.serialization import RobustSerializer
    from datetime import datetime
    
    print(f"\n💾 Guardando modelo...")
    
    serializer = RobustSerializer()
    
    metadata = {
        'model_name': model_name,
        'algorithm': type(best_learner.estimator).__name__,
        'performance': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'business_impact': {
            'customers_at_risk_detected': int(predicted_churn),
            'correctly_identified': int(correctly_identified),
            'potential_revenue_saved': float(revenue_saved)
        },
        'training_info': {
            'train_samples': task_train.n_obs,
            'test_samples': task_test.n_obs,
            'n_features': task_test.n_features,
            'training_date': datetime.now().isoformat()
        },
        'version': '1.0.0',
        'use_case': 'customer_churn_prediction'
    }
    
    model_path = f"models/shopsmart_churn_{model_name.lower()}_{datetime.now().strftime('%Y%m%d')}.pkl"
    
    save_result = serializer.save(
        obj=best_learner,
        path=model_path,
        metadata=metadata
    )
    
    print(f"✅ Modelo guardado:")
    print(f"  Archivo: {model_path}")
    print(f"  Checksum: {save_result['checksum'][:16]}...")
    print(f"  Metadata: {len(metadata)} campos")
    
    return {
        'model_path': model_path,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1
        },
        'business_impact': {
            'revenue_saved': revenue_saved,
            'customers_saved': customers_saved
        }
    }

# Evaluación final
final_results = final_evaluation_and_save(best_learner, task_test, best_model)
```

---

## 📊 RESULTADOS Y MÉTRICAS

### Métricas Técnicas Logradas

```python
# Resultados del modelo final
final_metrics = {
    'accuracy': 0.89,      # 89% de predicciones correctas
    'precision': 0.85,     # 85% de churn predicho es real
    'recall': 0.78,        # 78% de churn real es detectado  
    'f1': 0.81            # F1-score balanceado
}

# Matriz de confusión del test set
confusion_matrix = [
    [3850, 185],   # No churn: 3850 correctos, 185 falsos positivos
    [267, 948]     # Churn: 267 falsos negativos, 948 correctos
]
```

### Impacto de Negocio

```python
business_results = {
    'baseline': {
        'annual_churn_rate': 0.25,
        'customers_lost': 12500,
        'revenue_lost': 3750000  # $3.75M
    },
    'with_model': {
        'churn_correctly_identified': 948,
        'intervention_success_rate': 0.70,
        'customers_saved': 664,
        'revenue_saved': 199200,  # $199K por mes
        'annual_revenue_saved': 2390400  # $2.39M anual
    },
    'roi': {
        'implementation_cost': 50000,  # $50K
        'annual_benefit': 2390400,
        'roi_percentage': 4680  # 4,680% ROI
    }
}
```

### Dashboard de Monitoreo

```python
# El dashboard genera automáticamente:
dashboard_outputs = {
    'training_curves': 'Progreso de entrenamiento por modelo',
    'model_comparison': 'Comparación de accuracy/F1/tiempo',
    'feature_importance': 'Top 20 features más importantes',
    'confusion_matrix': 'Matriz de confusión interactiva',
    'roc_curve': 'Curva ROC para diferentes umbrales',
    'business_metrics': 'KPIs de impacto de negocio'
}

# URL del dashboard: http://localhost:8050/shopsmart-churn-dashboard
```

---

## 🎯 IMPLEMENTACIÓN EN PRODUCCIÓN

### Sistema de Scoring en Tiempo Real

```python
# production/real_time_scoring.py
from mlpy.serialization import RobustSerializer
import pandas as pd
from datetime import datetime, timedelta

class ChurnScoringService:
    """Servicio de scoring de churn en producción"""
    
    def __init__(self, model_path):
        self.serializer = RobustSerializer()
        self.model = self.serializer.load(model_path)
        print(f"✅ Modelo cargado: {model_path}")
        
    def score_customer(self, customer_id):
        """Score individual de un cliente"""
        
        # 1. Extraer features en tiempo real
        features = self._extract_features(customer_id)
        
        # 2. Hacer predicción
        prediction = self.model.predict_proba([features])[0]
        churn_probability = prediction[1]  # Probabilidad de churn
        
        # 3. Clasificar riesgo
        risk_level = self._classify_risk(churn_probability)
        
        return {
            'customer_id': customer_id,
            'churn_probability': churn_probability,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_features(self, customer_id):
        """Extrae features en tiempo real desde la BD"""
        
        # Query optimizado para features en tiempo real
        features_query = f"""
        WITH customer_stats AS (
            SELECT 
                {customer_id} as customer_id,
                DATEDIFF(NOW(), MAX(order_date)) as recency_days,
                COUNT(*) as frequency,
                AVG(order_value) as monetary_avg,
                -- ... más features
            FROM transactions 
            WHERE customer_id = {customer_id}
            AND order_date >= DATE_SUB(NOW(), INTERVAL 1 YEAR)
        )
        SELECT * FROM customer_stats
        """
        
        # En producción: connection a DB optimizada
        features_df = pd.read_sql(features_query, connection)
        return features_df.iloc[0].values
    
    def _classify_risk(self, probability):
        """Clasifica nivel de riesgo"""
        if probability >= 0.8:
            return 'ALTO'
        elif probability >= 0.5:
            return 'MEDIO'
        else:
            return 'BAJO'

# Uso en producción
scoring_service = ChurnScoringService('models/shopsmart_churn_model.pkl')

# Score individual
score = scoring_service.score_customer(customer_id=12345)
print(score)
# {'customer_id': 12345, 'churn_probability': 0.73, 'risk_level': 'MEDIO', ...}
```

### Pipeline de Batch Scoring

```python
# production/batch_scoring.py
import pandas as pd
from mlpy.lazy import ComputationGraph, ComputationNode

def daily_batch_scoring():
    """Pipeline diario de scoring de todos los clientes"""
    
    print("🔄 Iniciando batch scoring diario...")
    
    # Crear pipeline lazy para eficiencia
    graph = ComputationGraph()
    
    # Nodo 1: Extraer clientes activos
    def extract_active_customers():
        query = """
        SELECT DISTINCT customer_id 
        FROM customers 
        WHERE status = 'active'
        AND last_activity >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
        """
        return pd.read_sql(query, connection)
    
    customers_node = ComputationNode(
        id="active_customers",
        operation="extract",
        func=extract_active_customers
    )
    graph.add_node(customers_node)
    
    # Nodo 2: Extraer features para batch
    def extract_batch_features():
        customers = graph.nodes["active_customers"].result
        customer_ids = customers['customer_id'].tolist()
        
        # Query optimizado para batch
        features_query = f"""
        SELECT 
            customer_id,
            recency_days,
            frequency,
            monetary_avg,
            -- ... todas las features
        FROM customer_features_view
        WHERE customer_id IN ({','.join(map(str, customer_ids))})
        """
        
        return pd.read_sql(features_query, connection)
    
    features_node = ComputationNode(
        id="batch_features", 
        operation="extract_features",
        func=extract_batch_features,
        dependencies=["active_customers"]
    )
    graph.add_node(features_node)
    
    # Nodo 3: Scoring masivo
    def batch_scoring():
        features_df = graph.nodes["batch_features"].result
        
        # Cargar modelo
        model = scoring_service.model
        
        # Predicciones batch
        probabilities = model.predict_proba(features_df.drop('customer_id', axis=1))
        churn_probs = probabilities[:, 1]
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'customer_id': features_df['customer_id'],
            'churn_probability': churn_probs,
            'risk_level': [scoring_service._classify_risk(p) for p in churn_probs],
            'scoring_date': datetime.now().date()
        })
        
        return results
    
    scoring_node = ComputationNode(
        id="batch_scores",
        operation="score_batch",
        func=batch_scoring,
        dependencies=["batch_features"]
    )
    graph.add_node(scoring_node)
    
    # Ejecutar pipeline optimizado
    graph.optimize()
    results = graph.execute()
    
    scores_df = results["batch_scores"]
    
    # Guardar resultados
    scores_df.to_sql('daily_churn_scores', connection, if_exists='replace')
    
    # Alertas para alto riesgo
    high_risk = scores_df[scores_df['risk_level'] == 'ALTO']
    if len(high_risk) > 0:
        send_alerts(high_risk)
    
    print(f"✅ Batch scoring completado:")
    print(f"  - Clientes scoring: {len(scores_df)}")
    print(f"  - Alto riesgo: {len(high_risk)}")
    
    return scores_df

# Programar ejecución diaria (cron job)
# 0 6 * * * /usr/bin/python /path/to/batch_scoring.py
```

---

## 📈 LECCIONES APRENDIDAS

### Desafíos Enfrentados

1. **Datos Desbalanceados**
   - **Problema**: Solo 25% de clientes con churn
   - **Solución**: F1-score como métrica, técnicas de sampling
   - **MLPY ayudó**: Validación automática detectó el desbalance

2. **Feature Engineering Complejo**
   - **Problema**: Múltiples fuentes de datos, cálculos costosos
   - **Solución**: Lazy evaluation para optimización automática
   - **MLPY ayudó**: ComputationGraph optimizó el pipeline

3. **Interpretabilidad para Negocio**
   - **Problema**: Stakeholders necesitaban entender predicciones
   - **Solución**: SHAP + feature importance + insights de negocio
   - **MLPY ayudó**: Explicabilidad integrada en el workflow

### Decisiones Técnicas Clave

1. **¿Por qué F1-score sobre Accuracy?**
   - Datos desbalanceados requieren métrica que considere precision y recall
   - F1 penaliza tanto falsos positivos como falsos negativos

2. **¿Por qué Random Forest vs modelos más complejos?**
   - Interpretabilidad nativa (feature importance)
   - Robusto ante outliers y datos faltantes
   - Buen rendimiento out-of-the-box

3. **¿Por qué lazy evaluation?**
   - Pipeline de features computacionalmente costoso
   - Necesidad de re-ejecutar con nuevos datos
   - Optimización automática ahorra tiempo

### Mejores Prácticas Aplicadas

```python
best_practices = {
    'data_validation': {
        'practice': 'Validar datos antes de cualquier procesamiento',
        'mlpy_feature': 'validate_task_data()',
        'benefit': 'Previene errores tardíos y frustrantes'
    },
    'feature_engineering': {
        'practice': 'Pipeline reproducible y optimizable',
        'mlpy_feature': 'ComputationGraph + lazy evaluation',
        'benefit': 'Experimentos más rápidos y consistentes'
    },
    'model_comparison': {
        'practice': 'Comparar múltiples algoritmos sistemáticamente',
        'mlpy_feature': 'Dashboard + AutoML',
        'benefit': 'Decisiones informadas basadas en datos'
    },
    'model_persistence': {
        'practice': 'Guardar modelos con metadata e integridad',
        'mlpy_feature': 'RobustSerializer + checksums',
        'benefit': 'Trazabilidad y confianza en producción'
    },
    'explainability': {
        'practice': 'Explicar predicciones tanto técnica como comercialmente',
        'mlpy_feature': 'SHAP integration + business insights',
        'benefit': 'Adopción y confianza de stakeholders'
    }
}
```

---

## 🚀 PRÓXIMOS PASOS

### Roadmap de Mejoras

**Corto Plazo (1-3 meses):**
- [ ] A/B testing de estrategias de retención
- [ ] Modelo de propensión a compra específica por categoría
- [ ] Integración con sistema de email marketing

**Mediano Plazo (3-6 meses):**
- [ ] Modelo de Customer Lifetime Value (CLV)
- [ ] Predicción de cantidad de compra esperada
- [ ] Segmentación dinámica de clientes

**Largo Plazo (6-12 meses):**
- [ ] Real-time personalization engine
- [ ] Multi-channel attribution modeling
- [ ] Expansion a otros mercados geográficos

### Oportunidades de Expansión

1. **Otros Casos de Uso con los Mismos Datos:**
   - Predicción de valor de próxima compra
   - Recomendaciones personalizadas
   - Detección de fraude

2. **Integración con Otros Sistemas:**
   - CRM (Salesforce, HubSpot)
   - Email marketing (Mailchimp, SendGrid)
   - Customer service (Zendesk, Intercom)

3. **Automatización Avanzada:**
   - Auto-retraining mensual
   - Drift detection automático
   - Alert system para cambios en distribución

---

## 📁 RECURSOS DESCARGABLES

### Código Completo
- **Jupyter Notebook**: [shopsmart_churn_complete.ipynb](./notebooks/shopsmart_churn_complete.ipynb)
- **Scripts Python**: [src/](./src/)
- **Configuraciones**: [config/](./config/)

### Datasets
- **Datos sintéticos**: [data/shopsmart_synthetic_data.csv](./data/)
- **Features engineered**: [data/features_final.parquet](./data/)

### Dashboards
- **Interactive Dashboard**: [reports/dashboard.html](./reports/)
- **Executive Summary**: [reports/executive_summary.pdf](./reports/)

### Modelos
- **Modelo final**: [models/shopsmart_churn_randomforest.pkl](./models/)
- **Metadata**: [models/model_metadata.json](./models/)

---

*"El éxito en e-commerce no se trata de adquirir más clientes,  
sino de retener los que ya tienes."*

**→ Siguiente caso de uso:** [Detección de Fraude en Finanzas](./finanzas_deteccion_fraude.md)