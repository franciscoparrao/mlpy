"""Graph-based pipelines for MLPY.

This module provides the GraphLearner class for composing
pipeline operations into complex workflows.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque

from ..learners import Learner
from ..tasks import Task
from ..predictions import Prediction
from .base import PipeOp, PipeOpLearner


class Edge:
    """Edge in a pipeline graph.
    
    Represents a connection between two PipeOps.
    
    Parameters
    ----------
    src_id : str
        Source PipeOp ID.
    src_channel : str
        Source output channel.
    dst_id : str
        Destination PipeOp ID.
    dst_channel : str
        Destination input channel.
    """
    
    def __init__(
        self,
        src_id: str,
        src_channel: str,
        dst_id: str,
        dst_channel: str
    ):
        self.src_id = src_id
        self.src_channel = src_channel
        self.dst_id = dst_id
        self.dst_channel = dst_channel
        
    def __repr__(self):
        return f"Edge({self.src_id}.{self.src_channel} -> {self.dst_id}.{self.dst_channel})"


class Graph:
    """Directed acyclic graph of pipeline operations.
    
    Parameters
    ----------
    pipeops : dict
        Mapping from PipeOp IDs to PipeOp instances.
    """
    
    def __init__(self, pipeops: Optional[Dict[str, PipeOp]] = None):
        self.pipeops = pipeops or {}
        self.edges: List[Edge] = []
        self._sorted_ids: Optional[List[str]] = None
        
    def add_pipeop(self, pipeop: PipeOp) -> None:
        """Add a PipeOp to the graph."""
        if pipeop.id in self.pipeops:
            raise ValueError(f"PipeOp with ID '{pipeop.id}' already exists")
        self.pipeops[pipeop.id] = pipeop
        self._sorted_ids = None
        
    def add_edge(
        self,
        src_id: str,
        src_channel: str,
        dst_id: str,
        dst_channel: str
    ) -> None:
        """Add an edge between PipeOps.
        
        Parameters
        ----------
        src_id : str
            Source PipeOp ID.
        src_channel : str  
            Source output channel.
        dst_id : str
            Destination PipeOp ID.
        dst_channel : str
            Destination input channel.
        """
        # Validate
        if src_id not in self.pipeops:
            raise ValueError(f"Source PipeOp '{src_id}' not found")
        if dst_id not in self.pipeops:
            raise ValueError(f"Destination PipeOp '{dst_id}' not found")
            
        src_op = self.pipeops[src_id]
        dst_op = self.pipeops[dst_id]
        
        if src_channel not in src_op.output:
            raise ValueError(
                f"PipeOp '{src_id}' has no output channel '{src_channel}'"
            )
        if dst_channel not in dst_op.input:
            raise ValueError(
                f"PipeOp '{dst_id}' has no input channel '{dst_channel}'"
            )
            
        # Add edge
        edge = Edge(src_id, src_channel, dst_id, dst_channel)
        self.edges.append(edge)
        self._sorted_ids = None
        
    def topological_sort(self) -> List[str]:
        """Get PipeOp IDs in topological order.
        
        Returns
        -------
        list
            PipeOp IDs in execution order.
            
        Raises
        ------
        ValueError
            If graph contains cycles.
        """
        if self._sorted_ids is not None:
            return self._sorted_ids
            
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize all nodes
        for op_id in self.pipeops:
            in_degree[op_id] = 0
            
        # Add edges
        for edge in self.edges:
            graph[edge.src_id].append(edge.dst_id)
            in_degree[edge.dst_id] += 1
            
        # Find nodes with no incoming edges
        queue = deque([op_id for op_id in self.pipeops if in_degree[op_id] == 0])
        sorted_ids = []
        
        while queue:
            op_id = queue.popleft()
            sorted_ids.append(op_id)
            
            # Remove edges from this node
            for neighbor in graph[op_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(sorted_ids) != len(self.pipeops):
            raise ValueError("Graph contains cycles")
            
        self._sorted_ids = sorted_ids
        return sorted_ids
        
    def get_input_edges(self, op_id: str) -> Dict[str, Edge]:
        """Get input edges for a PipeOp.
        
        Returns
        -------
        dict
            Mapping from input channel to edge.
        """
        edges = {}
        for edge in self.edges:
            if edge.dst_id == op_id:
                edges[edge.dst_channel] = edge
        return edges
        
    def get_source_ops(self) -> List[str]:
        """Get PipeOps with no input edges."""
        has_input = {edge.dst_id for edge in self.edges}
        return [op_id for op_id in self.pipeops if op_id not in has_input]
        
    def get_sink_ops(self) -> List[str]:
        """Get PipeOps with no output edges."""
        has_output = {edge.src_id for edge in self.edges}
        return [op_id for op_id in self.pipeops if op_id not in has_output]
        
    def validate(self) -> None:
        """Validate the graph structure.
        
        Raises
        ------
        ValueError
            If graph is invalid.
        """
        # Check for cycles
        self.topological_sort()
        
        # Check all inputs are connected (except sources)
        sources = self.get_source_ops()
        for op_id, op in self.pipeops.items():
            if op_id in sources:
                continue
                
            input_edges = self.get_input_edges(op_id)
            for channel in op.input:
                if channel not in input_edges:
                    raise ValueError(
                        f"PipeOp '{op_id}' input '{channel}' is not connected"
                    )
                    
        # Check type compatibility
        for edge in self.edges:
            src_op = self.pipeops[edge.src_id]
            dst_op = self.pipeops[edge.dst_id]
            
            src_type = src_op.output[edge.src_channel]
            dst_type = dst_op.input[edge.dst_channel]
            
            # For now, just check object compatibility
            # Could add more sophisticated type checking
            
    def clone(self, deep: bool = True) -> "Graph":
        """Create a copy of the graph."""
        new_graph = Graph()
        
        # Clone PipeOps
        for op_id, op in self.pipeops.items():
            new_graph.pipeops[op_id] = op.clone(deep=deep)
            
        # Copy edges
        new_graph.edges = self.edges.copy()
        
        return new_graph


class GraphLearner(Learner):
    """Learner that executes a pipeline graph.
    
    This learner composes multiple PipeOps into a complex
    workflow that can be trained and used for prediction.
    
    Parameters
    ----------
    graph : Graph
        The pipeline graph to execute.
    id : str, optional
        Unique identifier.
    """
    
    def __init__(
        self,
        graph: Graph,
        id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            id=id or "graph",
            predict_type="response",
            **kwargs
        )
        self.graph = graph
        self._validate_graph()
        
    def _validate_graph(self) -> None:
        """Validate the graph is suitable for a learner."""
        self.graph.validate()
        
        # Must have exactly one source (input)
        sources = self.graph.get_source_ops()
        if len(sources) != 1:
            raise ValueError(
                f"GraphLearner requires exactly one source, got {len(sources)}"
            )
            
        # Must have exactly one sink (output)
        sinks = self.graph.get_sink_ops()
        if len(sinks) != 1:
            raise ValueError(
                f"GraphLearner requires exactly one sink, got {len(sinks)}"
            )
            
        # Source must accept Task
        source_op = self.graph.pipeops[sources[0]]
        if len(source_op.input) != 1:
            raise ValueError("Source PipeOp must have exactly one input")
            
        # Sink must produce Prediction
        sink_op = self.graph.pipeops[sinks[0]]
        if len(sink_op.output) != 1:
            raise ValueError("Sink PipeOp must have exactly one output")
            
    @property
    def task_type(self) -> str:
        """Infer task type from final learner in pipeline."""
        # Find learner ops
        for op in self.graph.pipeops.values():
            if isinstance(op, PipeOpLearner):
                return op.learner.task_type
        
        # Default to both
        return "classif"
        
    def train(self, task: Task, row_ids: Optional[List[int]] = None) -> "GraphLearner":
        """Train the pipeline.
        
        Parameters
        ----------
        task : Task
            The task to train on.
        row_ids : list of int, optional
            Subset of rows to use.
            
        Returns
        -------
        self
            The trained learner.
        """
        # Execute graph in topological order
        sorted_ids = self.graph.topological_sort()
        outputs = {}
        
        # Get source
        source_id = self.graph.get_source_ops()[0]
        
        for op_id in sorted_ids:
            op = self.graph.pipeops[op_id]
            
            # Prepare inputs
            if op_id == source_id:
                # Source gets the task
                inputs = {"input": task}
            else:
                # Collect inputs from edges
                inputs = {}
                input_edges = self.graph.get_input_edges(op_id)
                
                for channel, edge in input_edges.items():
                    src_outputs = outputs[edge.src_id]
                    inputs[channel] = src_outputs[edge.src_channel]
                    
            # Execute operation
            op_outputs = op.train(inputs)
            outputs[op_id] = op_outputs
            
        # Mark as trained
        self._model = self.graph
        self._train_task = task
        
        return self
        
    def predict(self, task: Task, row_ids: Optional[List[int]] = None) -> Prediction:
        """Make predictions with the trained pipeline.
        
        Parameters
        ----------
        task : Task
            The task to predict on.
        row_ids : list of int, optional
            Subset of rows to predict.
            
        Returns
        -------
        Prediction
            The predictions.
        """
        if not self.is_trained:
            raise RuntimeError("GraphLearner must be trained before predict")
            
        # Execute graph
        sorted_ids = self.graph.topological_sort()
        outputs = {}
        
        source_id = self.graph.get_source_ops()[0]
        sink_id = self.graph.get_sink_ops()[0]
        
        for op_id in sorted_ids:
            op = self.graph.pipeops[op_id]
            
            # Prepare inputs
            if op_id == source_id:
                inputs = {"input": task}
            else:
                inputs = {}
                input_edges = self.graph.get_input_edges(op_id)
                
                for channel, edge in input_edges.items():
                    src_outputs = outputs[edge.src_id]
                    inputs[channel] = src_outputs[edge.src_channel]
                    
            # Execute
            op_outputs = op.predict(inputs)
            outputs[op_id] = op_outputs
            
        # Return final output
        final_outputs = outputs[sink_id]
        return list(final_outputs.values())[0]
        
    def clone(self, deep: bool = True) -> "GraphLearner":
        """Create a copy of the learner."""
        new_graph = self.graph.clone(deep=deep)
        return GraphLearner(
            graph=new_graph,
            id=self.id
        )
        
    def plot(self, filename: Optional[str] = None) -> None:
        """Plot the pipeline graph.
        
        Parameters
        ----------
        filename : str, optional
            If provided, save plot to file.
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(f"Plotting requires matplotlib and networkx: {e}")
            
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for op_id in self.graph.pipeops:
            G.add_node(op_id)
            
        # Add edges
        for edge in self.graph.edges:
            G.add_edge(
                edge.src_id,
                edge.dst_id,
                label=f"{edge.src_channel}->{edge.dst_channel}"
            )
            
        # Layout
        pos = nx.spring_layout(G)
        
        # Draw
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20)
                
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title("Pipeline Graph")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


def linear_pipeline(*pipeops) -> Graph:
    """Create a linear pipeline from a sequence of PipeOps.

    Each PipeOp's output is connected to the next PipeOp's input.

    Parameters
    ----------
    *pipeops : PipeOp or list of PipeOp
        Sequence of pipeline operations. Can be passed as individual
        arguments or as a single list.

    Returns
    -------
    Graph
        The linear pipeline graph.

    Examples
    --------
    >>> # Can be called either way:
    >>> pipeline = linear_pipeline(op1, op2, op3)
    >>> pipeline = linear_pipeline([op1, op2, op3])
    """
    # Handle both linear_pipeline(op1, op2) and linear_pipeline([op1, op2])
    if len(pipeops) == 1 and isinstance(pipeops[0], (list, tuple)):
        pipeops = pipeops[0]

    if not pipeops:
        raise ValueError("At least one PipeOp required")

    graph = Graph()

    # Add all ops
    for op in pipeops:
        graph.add_pipeop(op)

    # Connect them linearly
    for i in range(len(pipeops) - 1):
        src = pipeops[i]
        dst = pipeops[i + 1]
        
        # Assume single output/input channels
        if len(src.output) != 1:
            raise ValueError(
                f"PipeOp '{src.id}' has multiple outputs, "
                "cannot create linear pipeline"
            )
        if len(dst.input) != 1:
            raise ValueError(
                f"PipeOp '{dst.id}' has multiple inputs, "
                "cannot create linear pipeline"
            )
            
        src_channel = list(src.output.keys())[0]
        dst_channel = list(dst.input.keys())[0]
        
        graph.add_edge(src.id, src_channel, dst.id, dst_channel)
        
    return graph


__all__ = [
    "Graph",
    "GraphLearner",
    "Edge",
    "linear_pipeline"
]