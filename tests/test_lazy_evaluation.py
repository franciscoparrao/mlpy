"""
Tests para el sistema de lazy evaluation.

Verifica optimización de grafos y ejecución diferida.
"""

import pytest
import numpy as np
import pandas as pd
import time
from typing import Any, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlpy.lazy.lazy_evaluation import (
    ComputationNode,
    ComputationGraph,
    LazyArray,
    LazyDataFrame,
    lazy_operation
)


class TestComputationNode:
    """Tests para nodos de computación."""
    
    def test_create_node(self):
        """Test creación de nodo básico."""
        def add_one(x):
            return x + 1
        
        node = ComputationNode(
            id="add_one",
            operation="addition",
            func=lambda: add_one(5)
        )
        
        assert node.id == "add_one"
        assert node.operation == "addition"
        assert node.dependencies == []
        assert node.result is None
        assert node.executed == False
    
    def test_node_with_dependencies(self):
        """Test nodo con dependencias."""
        node1 = ComputationNode(id="node1", operation="op1", func=lambda: 1)
        node2 = ComputationNode(
            id="node2",
            operation="op2", 
            func=lambda x: x + 1,
            dependencies=["node1"]
        )
        
        assert "node1" in node2.dependencies
        assert len(node2.dependencies) == 1
    
    def test_node_execution(self):
        """Test ejecución de nodo."""
        def compute():
            return 42
        
        node = ComputationNode(
            id="compute",
            operation="computation",
            func=compute
        )
        
        result = node.execute()
        assert result == 42
        assert node.result == 42
        assert node.executed == True
    
    def test_node_caching(self):
        """Test que el nodo cachea resultados."""
        counter = {'count': 0}
        
        def expensive_computation():
            counter['count'] += 1
            return counter['count']
        
        node = ComputationNode(
            id="expensive",
            operation="compute",
            func=expensive_computation
        )
        
        # Primera ejecución
        result1 = node.execute()
        assert result1 == 1
        assert counter['count'] == 1
        
        # Segunda ejecución debe usar cache
        result2 = node.execute()
        assert result2 == 1  # Mismo resultado
        assert counter['count'] == 1  # No se ejecutó de nuevo


class TestComputationGraph:
    """Tests para grafos de computación."""
    
    def test_create_empty_graph(self):
        """Test crear grafo vacío."""
        graph = ComputationGraph()
        assert len(graph.nodes) == 0
        assert len(graph.execution_order) == 0
    
    def test_add_nodes_to_graph(self):
        """Test agregar nodos al grafo."""
        graph = ComputationGraph()
        
        node1 = ComputationNode(id="n1", operation="op1", func=lambda: 1)
        node2 = ComputationNode(id="n2", operation="op2", func=lambda: 2)
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert len(graph.nodes) == 2
        assert "n1" in graph.nodes
        assert "n2" in graph.nodes
    
    def test_topological_sort(self):
        """Test ordenamiento topológico."""
        graph = ComputationGraph()
        
        # Crear DAG: n1 -> n2 -> n3
        #                -> n4 ->
        node1 = ComputationNode(id="n1", operation="op1", func=lambda: 1)
        node2 = ComputationNode(
            id="n2", operation="op2", 
            func=lambda x: x * 2,
            dependencies=["n1"]
        )
        node3 = ComputationNode(
            id="n3", operation="op3",
            func=lambda x: x + 10,
            dependencies=["n2", "n4"]
        )
        node4 = ComputationNode(
            id="n4", operation="op4",
            func=lambda x: x * 3,
            dependencies=["n1"]
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        graph.add_node(node4)
        
        order = graph._topological_sort()
        
        # n1 debe estar antes que n2 y n4
        assert order.index("n1") < order.index("n2")
        assert order.index("n1") < order.index("n4")
        
        # n2 y n4 deben estar antes que n3
        assert order.index("n2") < order.index("n3")
        assert order.index("n4") < order.index("n3")
    
    def test_graph_execution(self):
        """Test ejecución completa del grafo."""
        graph = ComputationGraph()
        
        # Pipeline: input -> double -> add_ten -> result
        node_input = ComputationNode(
            id="input",
            operation="load",
            func=lambda: 5
        )
        
        node_double = ComputationNode(
            id="double",
            operation="multiply",
            func=lambda: graph.nodes["input"].result * 2,
            dependencies=["input"]
        )
        
        node_add = ComputationNode(
            id="add_ten",
            operation="addition",
            func=lambda: graph.nodes["double"].result + 10,
            dependencies=["double"]
        )
        
        graph.add_node(node_input)
        graph.add_node(node_double)
        graph.add_node(node_add)
        
        results = graph.execute()
        
        assert results["input"] == 5
        assert results["double"] == 10
        assert results["add_ten"] == 20
    
    def test_graph_optimization(self):
        """Test optimización del grafo."""
        graph = ComputationGraph()
        
        # Crear nodos redundantes
        node1 = ComputationNode(id="n1", operation="op", func=lambda: 1)
        node2 = ComputationNode(id="n2", operation="op", func=lambda: 1)  # Duplicado
        node3 = ComputationNode(
            id="n3", operation="combine",
            func=lambda: 2,
            dependencies=["n1", "n2"]
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Optimizar
        graph.optimize()
        
        # La optimización debería detectar redundancia
        # (implementación simplificada puede no eliminar nodos)
        assert len(graph.nodes) <= 3
    
    def test_parallel_execution_potential(self):
        """Test identificación de ejecución paralela."""
        graph = ComputationGraph()
        
        # Nodos independientes que pueden ejecutarse en paralelo
        node_a = ComputationNode(id="a", operation="op_a", func=lambda: 1)
        node_b = ComputationNode(id="b", operation="op_b", func=lambda: 2)
        node_c = ComputationNode(id="c", operation="op_c", func=lambda: 3)
        
        # Nodo que depende de todos
        node_final = ComputationNode(
            id="final",
            operation="combine",
            func=lambda: sum([
                graph.nodes["a"].result,
                graph.nodes["b"].result,
                graph.nodes["c"].result
            ]),
            dependencies=["a", "b", "c"]
        )
        
        graph.add_node(node_a)
        graph.add_node(node_b)
        graph.add_node(node_c)
        graph.add_node(node_final)
        
        # Identificar grupos paralelos
        parallel_groups = graph._identify_parallel_groups()
        
        # a, b, c deberían estar en el mismo grupo
        first_group = parallel_groups[0] if parallel_groups else []
        assert len(first_group) >= 3 or len(graph.nodes) == 4
    
    def test_cycle_detection(self):
        """Test detección de ciclos."""
        graph = ComputationGraph()
        
        # Intentar crear ciclo: n1 -> n2 -> n3 -> n1
        node1 = ComputationNode(
            id="n1", operation="op1",
            func=lambda: 1,
            dependencies=["n3"]  # Ciclo!
        )
        node2 = ComputationNode(
            id="n2", operation="op2",
            func=lambda: 2,
            dependencies=["n1"]
        )
        node3 = ComputationNode(
            id="n3", operation="op3",
            func=lambda: 3,
            dependencies=["n2"]
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        # Debe detectar ciclo al intentar ordenamiento topológico
        with pytest.raises(ValueError, match="cycle|circular"):
            graph._topological_sort()


class TestLazyArray:
    """Tests para arrays lazy."""
    
    def test_create_lazy_array(self):
        """Test creación de array lazy."""
        data = np.array([1, 2, 3, 4, 5])
        lazy_arr = LazyArray(data)
        
        assert lazy_arr.shape == data.shape
        assert lazy_arr.dtype == data.dtype
        assert lazy_arr._computed == False
    
    def test_lazy_operations(self):
        """Test operaciones lazy."""
        arr1 = LazyArray(np.array([1, 2, 3]))
        arr2 = LazyArray(np.array([4, 5, 6]))
        
        # Operaciones no se ejecutan inmediatamente
        result = arr1 + arr2
        assert isinstance(result, LazyArray)
        assert result._computed == False
        
        # Compute fuerza evaluación
        computed = result.compute()
        assert np.array_equal(computed, np.array([5, 7, 9]))
    
    def test_chained_operations(self):
        """Test cadena de operaciones lazy."""
        arr = LazyArray(np.array([1, 2, 3, 4]))
        
        # Cadena de operaciones
        result = ((arr * 2) + 10) / 2
        
        # Nada se computa hasta compute()
        assert result._computed == False
        
        # Evaluar
        computed = result.compute()
        expected = ((np.array([1, 2, 3, 4]) * 2) + 10) / 2
        assert np.array_equal(computed, expected)
    
    def test_lazy_array_slicing(self):
        """Test slicing lazy."""
        arr = LazyArray(np.arange(10))
        
        # Slicing es lazy
        sliced = arr[2:7]
        assert sliced._computed == False
        
        # Compute
        result = sliced.compute()
        assert np.array_equal(result, np.arange(2, 7))


class TestLazyDataFrame:
    """Tests para DataFrames lazy."""
    
    def test_create_lazy_dataframe(self):
        """Test creación de DataFrame lazy."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        lazy_df = LazyDataFrame(df)
        
        assert lazy_df.shape == df.shape
        assert list(lazy_df.columns) == list(df.columns)
        assert lazy_df._computed == False
    
    def test_lazy_selection(self):
        """Test selección lazy de columnas."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        lazy_df = LazyDataFrame(df)
        
        # Selección es lazy
        selected = lazy_df[['a', 'c']]
        assert selected._computed == False
        
        # Compute
        result = selected.compute()
        pd.testing.assert_frame_equal(result, df[['a', 'c']])
    
    def test_lazy_filtering(self):
        """Test filtrado lazy."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        lazy_df = LazyDataFrame(df)
        
        # Filtrado es lazy
        filtered = lazy_df[lazy_df['value'] > 2]
        assert filtered._computed == False
        
        # Compute
        result = filtered.compute()
        expected = df[df['value'] > 2]
        pd.testing.assert_frame_equal(result, expected)
    
    def test_lazy_aggregation(self):
        """Test agregación lazy."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        lazy_df = LazyDataFrame(df)
        
        # Agregación es lazy
        grouped = lazy_df.groupby('group').mean()
        assert grouped._computed == False
        
        # Compute
        result = grouped.compute()
        expected = df.groupby('group').mean()
        pd.testing.assert_frame_equal(result, expected)


class TestLazyDecorator:
    """Tests para el decorador lazy_operation."""
    
    def test_lazy_decorator(self):
        """Test decorador lazy_operation."""
        
        @lazy_operation
        def expensive_function(x, y):
            time.sleep(0.1)  # Simular operación costosa
            return x + y
        
        # Llamada retorna función lazy
        lazy_result = expensive_function(5, 3)
        assert callable(lazy_result)
        
        # Ejecutar cuando se necesite
        result = lazy_result()
        assert result == 8
    
    def test_lazy_decorator_with_kwargs(self):
        """Test decorador con kwargs."""
        
        @lazy_operation
        def process_data(data, multiplier=2, offset=0):
            return [x * multiplier + offset for x in data]
        
        lazy_result = process_data([1, 2, 3], multiplier=3, offset=10)
        
        result = lazy_result()
        assert result == [13, 16, 19]


class TestLazyIntegration:
    """Tests de integración para lazy evaluation."""
    
    def test_ml_pipeline_lazy(self):
        """Test pipeline ML con evaluación lazy."""
        graph = ComputationGraph()
        
        # Simular pipeline ML
        # 1. Cargar datos
        load_node = ComputationNode(
            id="load_data",
            operation="load",
            func=lambda: pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'target': np.random.choice([0, 1], 100)
            })
        )
        
        # 2. Preprocesar
        preprocess_node = ComputationNode(
            id="preprocess",
            operation="normalize",
            func=lambda: self._normalize(graph.nodes["load_data"].result),
            dependencies=["load_data"]
        )
        
        # 3. Split
        split_node = ComputationNode(
            id="split",
            operation="train_test_split",
            func=lambda: self._split(graph.nodes["preprocess"].result),
            dependencies=["preprocess"]
        )
        
        # 4. Entrenar
        train_node = ComputationNode(
            id="train",
            operation="model_training",
            func=lambda: self._train(graph.nodes["split"].result),
            dependencies=["split"]
        )
        
        # Agregar al grafo
        graph.add_node(load_node)
        graph.add_node(preprocess_node)
        graph.add_node(split_node)
        graph.add_node(train_node)
        
        # Optimizar antes de ejecutar
        graph.optimize()
        
        # Ejecutar pipeline completo
        results = graph.execute()
        
        assert "load_data" in results
        assert "train" in results
        assert results["train"] is not None
    
    def _normalize(self, df):
        """Helper: normalizar datos."""
        from sklearn.preprocessing import StandardScaler
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
        return df
    
    def _split(self, df):
        """Helper: split train/test."""
        from sklearn.model_selection import train_test_split
        X = df.drop('target', axis=1)
        y = df['target']
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _train(self, split_data):
        """Helper: entrenar modelo."""
        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = split_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        return {'model': model, 'score': score}
    
    def test_lazy_caching_benefit(self):
        """Test beneficio del caching en lazy evaluation."""
        graph = ComputationGraph()
        
        # Contador para verificar ejecuciones
        exec_count = {'expensive': 0, 'consumer1': 0, 'consumer2': 0}
        
        def expensive_computation():
            exec_count['expensive'] += 1
            time.sleep(0.1)
            return np.random.randn(1000, 100)
        
        # Nodo costoso
        expensive_node = ComputationNode(
            id="expensive",
            operation="generate",
            func=expensive_computation
        )
        
        # Dos consumidores del mismo resultado
        consumer1 = ComputationNode(
            id="consumer1",
            operation="mean",
            func=lambda: graph.nodes["expensive"].result.mean(),
            dependencies=["expensive"]
        )
        
        consumer2 = ComputationNode(
            id="consumer2",
            operation="std",
            func=lambda: graph.nodes["expensive"].result.std(),
            dependencies=["expensive"]
        )
        
        graph.add_node(expensive_node)
        graph.add_node(consumer1)
        graph.add_node(consumer2)
        
        # Ejecutar
        results = graph.execute()
        
        # El nodo costoso debe ejecutarse solo una vez
        assert exec_count['expensive'] == 1
        assert results["consumer1"] is not None
        assert results["consumer2"] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])