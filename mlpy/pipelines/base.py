"""Base classes for MLPY pipelines.

This module provides the fundamental abstractions for building
machine learning pipelines with composable operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
import copy
from dataclasses import dataclass, field

from ..base import MLPYObject
from ..tasks import Task
from ..learners import Learner
from ..predictions import Prediction
from ..utils.registry import Registry


# Registry for pipeline operations
mlpy_pipeops = Registry("pipeops")


@dataclass
class PipeOpInput:
    """Input specification for a PipeOp.
    
    Parameters
    ----------
    name : str
        Name of the input channel.
    train : type
        Expected type during training (e.g., Task, np.ndarray).
    predict : type
        Expected type during prediction.
    """
    name: str
    train: type
    predict: type


@dataclass
class PipeOpOutput:
    """Output specification for a PipeOp.
    
    Parameters
    ----------
    name : str
        Name of the output channel.
    train : type
        Type produced during training.
    predict : type
        Type produced during prediction.
    """
    name: str
    train: type
    predict: type


class PipeOpState:
    """State of a PipeOp after training.
    
    This class stores any learned parameters or state that
    needs to be preserved between training and prediction.
    """
    def __init__(self):
        self.is_trained = False
        self.params = {}
        
    def __getitem__(self, key: str) -> Any:
        """Get a state parameter."""
        return self.params.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a state parameter."""
        self.params[key] = value


class PipeOp(MLPYObject, ABC):
    """Abstract base class for pipeline operations.
    
    A PipeOp is an atomic operation in a pipeline that can transform
    data, train models, or perform other operations. PipeOps can be
    composed to create complex ML workflows.
    
    Parameters
    ----------
    id : str
        Unique identifier for the operation.
    param_set : dict, optional
        Parameters for the operation.
    """
    
    def __init__(
        self,
        id: str,
        param_set: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(id=id, **kwargs)
        self.param_set = param_set or {}
        self.state = PipeOpState()
        
    @property
    @abstractmethod
    def input(self) -> Dict[str, PipeOpInput]:
        """Input channel specifications.
        
        Returns
        -------
        dict
            Mapping from channel names to input specifications.
        """
        pass
        
    @property
    @abstractmethod
    def output(self) -> Dict[str, PipeOpOutput]:
        """Output channel specifications.
        
        Returns
        -------
        dict
            Mapping from channel names to output specifications.
        """
        pass
        
    @property
    def n_inputs(self) -> int:
        """Number of input channels."""
        return len(self.input)
        
    @property
    def n_outputs(self) -> int:
        """Number of output channels."""
        return len(self.output)
        
    @property
    def is_trained(self) -> bool:
        """Whether the operation has been trained."""
        return self.state.is_trained
        
    @abstractmethod
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train the operation and transform inputs.
        
        Parameters
        ----------
        inputs : dict
            Input data mapped by channel name.
            
        Returns
        -------
        dict
            Output data mapped by channel name.
        """
        pass
        
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the operation to new data.
        
        Parameters
        ----------
        inputs : dict
            Input data mapped by channel name.
            
        Returns
        -------
        dict
            Output data mapped by channel name.
        """
        pass
        
    def reset(self) -> "PipeOp":
        """Reset the operation to untrained state.
        
        Returns
        -------
        self
            The reset operation.
        """
        self.state = PipeOpState()
        return self
        
    def clone(self, deep: bool = True) -> "PipeOp":
        """Create a copy of the operation.
        
        Parameters
        ----------
        deep : bool, default=True
            Whether to make a deep copy.
            
        Returns
        -------
        PipeOp
            A copy of the operation.
        """
        if deep:
            cloned = copy.deepcopy(self)
        else:
            cloned = copy.copy(self)
        cloned.reset()
        return cloned
        
    def validate_inputs(self, inputs: Dict[str, Any], phase: str = "train") -> None:
        """Validate that inputs match specifications.
        
        Parameters
        ----------
        inputs : dict
            Input data to validate.
        phase : str
            Either "train" or "predict".
            
        Raises
        ------
        ValueError
            If inputs don't match specifications.
        """
        # Check all required inputs are provided
        for name, spec in self.input.items():
            if name not in inputs:
                raise ValueError(f"Missing required input '{name}'")
                
            # Check type
            expected_type = spec.train if phase == "train" else spec.predict
            if not isinstance(inputs[name], expected_type):
                raise TypeError(
                    f"Input '{name}' expected {expected_type.__name__}, "
                    f"got {type(inputs[name]).__name__}"
                )
                
        # Check no extra inputs
        extra = set(inputs.keys()) - set(self.input.keys())
        if extra:
            raise ValueError(f"Unexpected inputs: {extra}")
            
    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"<{self.__class__.__name__}:{self.id}> ({status})"


class PipeOpLearner(PipeOp):
    """Pipeline operation that wraps a learner.
    
    This operation trains a learner during the training phase
    and makes predictions during the prediction phase.
    
    Parameters
    ----------
    learner : Learner
        The learner to wrap.
    id : str, optional
        Unique identifier. If None, uses learner's ID.
    """
    
    def __init__(
        self,
        learner: Learner,
        id: Optional[str] = None,
        **kwargs
    ):
        id = id or f"learner.{learner.id}"
        super().__init__(id=id, **kwargs)
        self.learner = learner
        self._trained_learner = None
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Expects a task as input."""
        return {
            "input": PipeOpInput(
                name="input",
                train=Task,
                predict=Task
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Produces predictions as output."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=Prediction,
                predict=Prediction
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Train the learner and return predictions.
        
        Parameters
        ----------
        inputs : dict
            Must contain "input" key with a Task.
            
        Returns
        -------
        dict
            Contains "output" key with predictions on training data.
        """
        self.validate_inputs(inputs, "train")
        
        task = inputs["input"]
        
        # Clone learner to avoid side effects
        self._trained_learner = self.learner.clone()
        self._trained_learner.train(task)
        
        # Get predictions on training data
        predictions = self._trained_learner.predict(task)
        
        self.state.is_trained = True
        self.state["learner"] = self._trained_learner
        
        return {"output": predictions}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions with the trained learner.
        
        Parameters
        ----------
        inputs : dict
            Must contain "input" key with a Task.
            
        Returns
        -------
        dict
            Contains "output" key with predictions.
        """
        if not self.is_trained:
            raise RuntimeError("PipeOpLearner must be trained before predict")
            
        self.validate_inputs(inputs, "predict")
        
        task = inputs["input"]
        predictions = self._trained_learner.predict(task)
        
        return {"output": predictions}
        
    def reset(self) -> "PipeOpLearner":
        """Reset to untrained state."""
        super().reset()
        self._trained_learner = None
        return self


class PipeOpNOP(PipeOp):
    """No-operation pipeline operation.
    
    This operation simply passes its input through unchanged.
    Useful for testing and as a placeholder.
    
    Parameters
    ----------
    id : str, default="nop"
        Unique identifier.
    """
    
    def __init__(self, id: str = "nop", **kwargs):
        super().__init__(id=id, **kwargs)
        
    @property
    def input(self) -> Dict[str, PipeOpInput]:
        """Accepts any type."""
        return {
            "input": PipeOpInput(
                name="input",
                train=object,
                predict=object
            )
        }
        
    @property
    def output(self) -> Dict[str, PipeOpOutput]:
        """Returns the same type as input."""
        return {
            "output": PipeOpOutput(
                name="output",
                train=object,
                predict=object
            )
        }
        
    def train(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through unchanged."""
        self.validate_inputs(inputs, "train")
        self.state.is_trained = True
        return {"output": inputs["input"]}
        
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Pass through unchanged."""
        return {"output": inputs["input"]}


# Register base operations
mlpy_pipeops.register("nop", PipeOpNOP)


__all__ = [
    "PipeOp",
    "PipeOpInput", 
    "PipeOpOutput",
    "PipeOpState",
    "PipeOpLearner",
    "PipeOpNOP",
    "mlpy_pipeops"
]