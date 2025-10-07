"""Tests for base classes."""

import pytest
from mlpy.core.base import MLPYObject, ParamSet, ParamDef


class ConcreteMLPYObject(MLPYObject):
    """Concrete implementation for testing."""
    
    @property
    def _properties(self):
        return {"prop1", "prop2", "prop3"}


class TestMLPYObject:
    """Test MLPYObject functionality."""
    
    def test_create_object(self):
        """Test creating an MLPYObject."""
        obj = ConcreteMLPYObject()
        
        # Check ID is generated
        assert obj.id.startswith("concretemlpyobject_")
        assert len(obj.id) > len("concretemlpyobject_")
        
        # Check label defaults to ID
        assert obj.label == obj.id
    
    def test_create_with_id_and_label(self):
        """Test creating object with custom ID and label."""
        obj = ConcreteMLPYObject(id="custom_id", label="Custom Label")
        
        assert obj.id == "custom_id"
        assert obj.label == "Custom Label"
    
    def test_hash_calculation(self):
        """Test hash calculation and caching."""
        obj = ConcreteMLPYObject(id="test")
        
        # Get hash twice - should be cached
        hash1 = obj.hash
        hash2 = obj.hash
        
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 produces 64 hex chars
    
    def test_hash_changes_with_params(self):
        """Test that hash changes when parameters change."""
        obj = ConcreteMLPYObject()
        hash1 = obj.hash
        
        obj.set_params(param1="value1")
        hash2 = obj.hash
        
        assert hash1 != hash2
    
    def test_clone(self):
        """Test cloning objects."""
        obj1 = ConcreteMLPYObject(id="original", label="Original")
        obj1.set_params(param1="value1", param2=42)
        
        # Deep clone
        obj2 = obj1.clone(deep=True)
        
        # Different objects but same ID (clones preserve ID)
        assert obj1 is not obj2
        assert obj1.id == obj2.id  # Clones keep the same ID
        
        # Same parameters
        assert obj1.get_params() == obj2.get_params()
        
        # Modifying clone doesn't affect original
        obj2.set_params(param1="modified")
        assert obj1.get_param("param1") == "value1"
        assert obj2.get_param("param1") == "modified"
    
    def test_shallow_clone(self):
        """Test shallow cloning."""
        obj1 = ConcreteMLPYObject()
        obj1.set_params(data=[1, 2, 3])
        
        obj2 = obj1.clone(deep=False)
        
        # Different objects
        assert obj1 is not obj2
        assert obj1.id != obj2.id
    
    def test_parameters(self):
        """Test parameter management."""
        obj = ConcreteMLPYObject()
        
        # Set parameters
        obj.set_params(param1="value1", param2=42, param3=True)
        
        # Get individual parameter
        assert obj.get_param("param1") == "value1"
        assert obj.get_param("param2") == 42
        assert obj.get_param("missing", "default") == "default"
        
        # Get all parameters
        params = obj.get_params()
        assert params == {"param1": "value1", "param2": 42, "param3": True}
    
    def test_properties(self):
        """Test property checking."""
        obj = ConcreteMLPYObject()
        
        assert obj.has_property("prop1")
        assert obj.has_property("prop2")
        assert not obj.has_property("missing")
    
    def test_equality(self):
        """Test object equality based on hash."""
        obj1 = ConcreteMLPYObject(id="test")
        obj2 = ConcreteMLPYObject(id="test")
        obj3 = ConcreteMLPYObject(id="different")
        
        # Same parameters = same hash = equal
        assert obj1 == obj2
        
        # Different ID = different hash = not equal
        assert obj1 != obj3
        
        # Not equal to non-MLPYObject
        assert obj1 != "string"
        assert obj1 != 42
    
    def test_string_representations(self):
        """Test __str__ and __repr__."""
        obj = ConcreteMLPYObject(id="test_id", label="Test Label")
        
        assert str(obj) == "ConcreteMLPYObject(Test Label)"
        assert repr(obj) == "<ConcreteMLPYObject:test_id>"
    
    def test_hashable(self):
        """Test that objects are hashable."""
        obj1 = ConcreteMLPYObject(id="test1")
        obj2 = ConcreteMLPYObject(id="test2")
        
        # Can be used in sets
        obj_set = {obj1, obj2, obj1}  # obj1 appears twice
        assert len(obj_set) == 2
        
        # Can be used as dict keys
        obj_dict = {obj1: "value1", obj2: "value2"}
        assert obj_dict[obj1] == "value1"


class TestParamDef:
    """Test ParamDef functionality."""
    
    def test_basic_validation(self):
        """Test basic type validation."""
        param = ParamDef(type=int)
        
        # Valid value
        assert param.validate(42) == 42
        
        # Invalid type
        with pytest.raises(ValueError, match="Expected type int"):
            param.validate("string")
    
    def test_numeric_bounds(self):
        """Test numeric bounds validation."""
        param = ParamDef(type=float, lower=0.0, upper=1.0)
        
        # Valid values
        assert param.validate(0.5) == 0.5
        assert param.validate(0.0) == 0.0
        assert param.validate(1.0) == 1.0
        
        # Below lower bound
        with pytest.raises(ValueError, match="below lower bound"):
            param.validate(-0.1)
        
        # Above upper bound
        with pytest.raises(ValueError, match="above upper bound"):
            param.validate(1.1)
    
    def test_categorical_values(self):
        """Test categorical value validation."""
        param = ParamDef(type=str, values=["a", "b", "c"])
        
        # Valid values
        assert param.validate("a") == "a"
        assert param.validate("b") == "b"
        
        # Invalid value
        with pytest.raises(ValueError, match="not in allowed values"):
            param.validate("d")
    
    def test_multiple_types(self):
        """Test validation with multiple allowed types."""
        param = ParamDef(type=(int, float))
        
        assert param.validate(42) == 42
        assert param.validate(3.14) == 3.14
        
        with pytest.raises(ValueError):
            param.validate("string")


class TestParamSet:
    """Test ParamSet functionality."""
    
    def test_validate_params(self):
        """Test parameter set validation."""
        param_set = ParamSet({
            "n_estimators": ParamDef(type=int, default=100, lower=1),
            "max_depth": ParamDef(type=(int, type(None)), default=None),
            "criterion": ParamDef(type=str, default="gini", values=["gini", "entropy"]),
        })
        
        # Valid parameters
        validated = param_set.validate({
            "n_estimators": 50,
            "criterion": "entropy"
        })
        
        assert validated["n_estimators"] == 50
        assert validated["criterion"] == "entropy"
        # max_depth not included because its default is None
        assert "max_depth" not in validated
    
    def test_unknown_parameter(self):
        """Test error on unknown parameter."""
        param_set = ParamSet({
            "known": ParamDef(type=str)
        })
        
        with pytest.raises(ValueError, match="Unknown parameter: unknown"):
            param_set.validate({"unknown": "value"})
    
    def test_get_defaults(self):
        """Test getting default values."""
        param_set = ParamSet({
            "param1": ParamDef(type=int, default=42),
            "param2": ParamDef(type=str, default="default"),
            "param3": ParamDef(type=float),  # No default
        })
        
        defaults = param_set.get_defaults()
        assert defaults == {"param1": 42, "param2": "default"}