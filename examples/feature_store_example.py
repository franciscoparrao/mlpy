"""
Ejemplo de uso del Feature Store de MLPY.

Este ejemplo muestra cómo:
1. Crear y registrar features
2. Definir grupos de features
3. Crear vistas de features
4. Aplicar transformaciones
5. Materializar features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Agregar el directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlpy.feature_store import (
    Feature, FeatureGroup, FeatureView,
    LocalFeatureStore, FeatureRegistry,
    FeatureType, DataSource,
    AggregationTransform, WindowTransform, RatioTransform,
    MaterializationScheduler, MaterializationStatus
)


def create_sample_data():
    """Crea datos de ejemplo para el feature store."""
    
    # Datos de usuarios
    np.random.seed(42)
    n_users = 100
    
    users_df = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 70, n_users),
        'tenure_days': np.random.randint(1, 1000, n_users),
        'total_purchases': np.random.randint(0, 100, n_users),
        'total_spent': np.random.uniform(0, 5000, n_users),
        'last_login_days': np.random.randint(0, 30, n_users),
        'email_verified': np.random.choice([0, 1], n_users),
        'phone_verified': np.random.choice([0, 1], n_users)
    })
    
    # Datos de productos
    n_products = 50
    
    products_df = pd.DataFrame({
        'product_id': range(n_products),
        'price': np.random.uniform(10, 500, n_products),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_products),
        'rating': np.random.uniform(1, 5, n_products),
        'reviews_count': np.random.randint(0, 1000, n_products),
        'in_stock': np.random.choice([0, 1], n_products),
        'discount_percent': np.random.uniform(0, 50, n_products)
    })
    
    # Datos de transacciones
    n_transactions = 500
    
    transactions_df = pd.DataFrame({
        'transaction_id': range(n_transactions),
        'user_id': np.random.randint(0, n_users, n_transactions),
        'product_id': np.random.randint(0, n_products, n_transactions),
        'quantity': np.random.randint(1, 5, n_transactions),
        'amount': np.random.uniform(10, 1000, n_transactions),
        'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30)) 
                     for _ in range(n_transactions)]
    })
    
    return users_df, products_df, transactions_df


def setup_feature_store():
    """Configura el feature store con features y grupos."""
    
    print("=" * 60)
    print("CONFIGURANDO FEATURE STORE")
    print("=" * 60)
    
    # Crear feature store local
    store = LocalFeatureStore(storage_path="./example_feature_store")
    
    # Crear registry de features
    registry = FeatureRegistry()
    
    # 1. Definir features de usuarios
    print("\n1. Definiendo features de usuarios...")
    
    user_features = [
        Feature("age", FeatureType.NUMERIC, "Edad del usuario"),
        Feature("tenure_days", FeatureType.NUMERIC, "Días desde registro"),
        Feature("total_purchases", FeatureType.NUMERIC, "Total de compras"),
        Feature("total_spent", FeatureType.NUMERIC, "Total gastado"),
        Feature("last_login_days", FeatureType.NUMERIC, "Días desde último login"),
        Feature("email_verified", FeatureType.BINARY, "Email verificado"),
        Feature("phone_verified", FeatureType.BINARY, "Teléfono verificado")
    ]
    
    # Registrar features
    for feature in user_features:
        registry.register_feature(feature)
    
    # Crear grupo de features de usuarios
    user_group = FeatureGroup(
        name="user_features",
        features=user_features,
        entity="user",
        source=DataSource.CSV,
        description="Features demográficas y de comportamiento de usuarios"
    )
    
    store.register_feature_group(user_group)
    print(f"   ✓ Registrado grupo: user_features ({len(user_features)} features)")
    
    # 2. Definir features de productos
    print("\n2. Definiendo features de productos...")
    
    product_features = [
        Feature("price", FeatureType.NUMERIC, "Precio del producto"),
        Feature("category", FeatureType.CATEGORICAL, "Categoría del producto"),
        Feature("rating", FeatureType.NUMERIC, "Rating promedio"),
        Feature("reviews_count", FeatureType.NUMERIC, "Número de reseñas"),
        Feature("in_stock", FeatureType.BINARY, "Disponible en stock"),
        Feature("discount_percent", FeatureType.NUMERIC, "Porcentaje de descuento")
    ]
    
    for feature in product_features:
        registry.register_feature(feature)
    
    product_group = FeatureGroup(
        name="product_features",
        features=product_features,
        entity="product",
        source=DataSource.CSV,
        description="Features de productos del catálogo"
    )
    
    store.register_feature_group(product_group)
    print(f"   ✓ Registrado grupo: product_features ({len(product_features)} features)")
    
    # 3. Crear vistas de features
    print("\n3. Creando vistas de features...")
    
    # Vista para modelo de recomendación
    recommendation_view = FeatureView(
        name="recommendation_features",
        feature_groups=["user_features", "product_features"],
        features=["age", "total_purchases", "total_spent", "price", "rating", "category"],
        entity="user",
        ttl=timedelta(hours=24),
        description="Features para sistema de recomendación"
    )
    
    store.create_feature_view(recommendation_view)
    print("   ✓ Creada vista: recommendation_features")
    
    # Vista para modelo de churn
    churn_view = FeatureView(
        name="churn_prediction_features",
        feature_groups=["user_features"],
        features=["tenure_days", "total_purchases", "last_login_days", "email_verified"],
        entity="user",
        ttl=timedelta(hours=12),
        description="Features para predicción de churn"
    )
    
    store.create_feature_view(churn_view)
    print("   ✓ Creada vista: churn_prediction_features")
    
    return store, registry


def write_sample_data(store, users_df, products_df):
    """Escribe datos de ejemplo al feature store."""
    
    print("\n" + "=" * 60)
    print("ESCRIBIENDO DATOS AL FEATURE STORE")
    print("=" * 60)
    
    # Escribir datos de usuarios
    print("\n1. Escribiendo features de usuarios...")
    success = store.write_features("user_features", users_df)
    if success:
        print(f"   ✓ Escritas {len(users_df)} filas de user_features")
    
    # Escribir datos de productos
    print("\n2. Escribiendo features de productos...")
    success = store.write_features("product_features", products_df)
    if success:
        print(f"   ✓ Escritas {len(products_df)} filas de product_features")


def demonstrate_transformations():
    """Demuestra el uso de transformaciones."""
    
    print("\n" + "=" * 60)
    print("DEMOSTRANDO TRANSFORMACIONES")
    print("=" * 60)
    
    # Crear datos de ejemplo
    df = pd.DataFrame({
        'feature1': [10, 20, 30, 40, 50],
        'feature2': [5, 10, 15, 20, 25],
        'feature3': [1, 2, 3, 4, 5]
    })
    
    # 1. Agregación
    print("\n1. Transformación de Agregación (suma):")
    agg_transform = AggregationTransform(
        name="sum_features",
        features=['feature1', 'feature2'],
        method="sum"
    )
    result = agg_transform.transform(df)
    print(f"   Entrada: feature1={df['feature1'].tolist()}, feature2={df['feature2'].tolist()}")
    print(f"   Resultado: {result.tolist()}")
    
    # 2. Ratio
    print("\n2. Transformación de Ratio:")
    ratio_transform = RatioTransform(
        name="ratio_features",
        numerator="feature1",
        denominator="feature2"
    )
    result = ratio_transform.transform(df)
    print(f"   feature1/feature2 = {result.tolist()}")
    
    # 3. Ventana temporal
    print("\n3. Transformación de Ventana (media móvil):")
    window_transform = WindowTransform(
        name="moving_avg",
        feature="feature1",
        window_size=3,
        method="mean"
    )
    result = window_transform.transform(df)
    print(f"   Media móvil (ventana=3): {[f'{x:.1f}' for x in result.tolist()]}")


def demonstrate_materialization(store):
    """Demuestra la materialización de features."""
    
    print("\n" + "=" * 60)
    print("DEMOSTRANDO MATERIALIZACIÓN")
    print("=" * 60)
    
    # Crear scheduler
    scheduler = MaterializationScheduler(
        feature_store=store,
        job_history_path="./example_materialization"
    )
    
    # Materializar vista de recomendación
    print("\n1. Materializando vista 'recommendation_features'...")
    job = scheduler.materialize_view("recommendation_features")
    
    print(f"   Job ID: {job.job_id}")
    print(f"   Estado: {job.status.value}")
    print(f"   Filas procesadas: {job.rows_processed}")
    if job.features_computed:
        print(f"   Features computadas: {', '.join(job.features_computed[:5])}...")
    
    # Materializar vista de churn
    print("\n2. Materializando vista 'churn_prediction_features'...")
    job = scheduler.materialize_view("churn_prediction_features")
    
    print(f"   Job ID: {job.job_id}")
    print(f"   Estado: {job.status.value}")
    print(f"   Filas procesadas: {job.rows_processed}")
    
    # Listar jobs
    print("\n3. Historial de jobs:")
    jobs = scheduler.list_jobs(limit=5)
    for job in jobs:
        print(f"   - {job.job_id}: {job.status.value} ({job.rows_processed} filas)")
    
    return scheduler


def query_features(store):
    """Consulta features del store."""
    
    print("\n" + "=" * 60)
    print("CONSULTANDO FEATURES")
    print("=" * 60)
    
    # Obtener features para usuarios específicos
    print("\n1. Obteniendo features para usuarios [0, 1, 2]...")
    
    features_to_get = ["age", "total_purchases", "total_spent"]
    data = store.get_features(
        entity_ids=[0, 1, 2],
        features=features_to_get
    )
    
    if not data.empty:
        print("\n   Resultado:")
        print(data.head())
    else:
        print("   No se encontraron datos")
    
    # Obtener grupo de features
    print("\n2. Información del grupo 'user_features':")
    group = store.get_feature_group("user_features")
    if group:
        print(f"   - Nombre: {group.name}")
        print(f"   - Entidad: {group.entity}")
        print(f"   - Número de features: {len(group.features)}")
        print(f"   - Schema: {list(group.get_schema().keys())}")


def main():
    """Función principal."""
    
    print("\n" + "=" * 60)
    print("EJEMPLO DE FEATURE STORE DE MLPY")
    print("=" * 60)
    
    # Crear datos de ejemplo
    users_df, products_df, transactions_df = create_sample_data()
    print(f"\n✓ Creados datos de ejemplo:")
    print(f"  - {len(users_df)} usuarios")
    print(f"  - {len(products_df)} productos")
    print(f"  - {len(transactions_df)} transacciones")
    
    # Configurar feature store
    store, registry = setup_feature_store()
    
    # Escribir datos
    write_sample_data(store, users_df, products_df)
    
    # Demostrar transformaciones
    demonstrate_transformations()
    
    # Demostrar materialización
    scheduler = demonstrate_materialization(store)
    
    # Consultar features
    query_features(store)
    
    print("\n" + "=" * 60)
    print("✓ EJEMPLO COMPLETADO CON ÉXITO")
    print("=" * 60)
    print("\nEl Feature Store permite:")
    print("  • Gestión centralizada de features")
    print("  • Versionado y linaje de features")
    print("  • Transformaciones reutilizables")
    print("  • Materialización programada")
    print("  • Serving de features para training y predicción")


if __name__ == "__main__":
    main()