"""Tests for the Registry system."""

import pytest
from mlpy.utils.registry import Registry


class TestRegistry:
    """Test Registry functionality."""
    
    def test_create_registry(self):
        """Test creating a new registry."""
        registry = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0
        assert repr(registry) == "<Registry 'test' (0 items, 0 aliases)>"
    
    def test_add_and_get_item(self):
        """Test adding and retrieving items."""
        registry = Registry("test")
        
        # Add item directly
        registry["item1"] = "value1"
        assert registry["item1"] == "value1"
        assert "item1" in registry
        assert len(registry) == 1
    
    def test_register_decorator(self):
        """Test using register as a decorator."""
        registry = Registry("test")
        
        @registry.register("my_class")
        class MyClass:
            pass
        
        assert registry["my_class"] == MyClass
        assert "my_class" in registry
    
    def test_register_with_aliases(self):
        """Test registering with aliases."""
        registry = Registry("test")
        
        @registry.register("full_name", aliases=["fn", "fname"])
        class MyClass:
            pass
        
        # All names should work
        assert registry["full_name"] == MyClass
        assert registry["fn"] == MyClass
        assert registry["fname"] == MyClass
        
        # Check contains
        assert "full_name" in registry
        assert "fn" in registry
        assert "fname" in registry
    
    def test_add_alias(self):
        """Test adding alias to existing item."""
        registry = Registry("test")
        registry["item1"] = "value1"
        
        registry.add_alias("alias1", "item1")
        assert registry["alias1"] == "value1"
    
    def test_get_keys(self):
        """Test getting registry keys."""
        registry = Registry("test")
        registry["item1"] = "value1"
        registry["item2"] = "value2"
        registry.add_alias("alias1", "item1")
        
        # Without aliases
        keys = registry.get_keys(include_aliases=False)
        assert keys == ["item1", "item2"]
        
        # With aliases
        keys = registry.get_keys(include_aliases=True)
        assert keys == ["alias1", "item1", "item2"]
    
    def test_delete_item(self):
        """Test deleting items."""
        registry = Registry("test")
        registry["item1"] = "value1"
        registry.add_alias("alias1", "item1")
        
        del registry["item1"]
        
        assert "item1" not in registry
        assert "alias1" not in registry
        assert len(registry) == 0
    
    def test_clear(self):
        """Test clearing registry."""
        registry = Registry("test")
        registry["item1"] = "value1"
        registry["item2"] = "value2"
        registry.add_alias("alias1", "item1")
        
        registry.clear()
        
        assert len(registry) == 0
        assert "item1" not in registry
        assert "alias1" not in registry
    
    def test_iteration(self):
        """Test iterating over registry."""
        registry = Registry("test")
        registry["item1"] = "value1"
        registry["item2"] = "value2"
        
        items = list(registry)
        assert sorted(items) == ["item1", "item2"]
        
        # Test items() method
        items_dict = dict(registry.items())
        assert items_dict == {"item1": "value1", "item2": "value2"}
    
    def test_error_cases(self):
        """Test error handling."""
        registry = Registry("test")
        
        # KeyError for missing item
        with pytest.raises(KeyError, match="'missing' not found"):
            _ = registry["missing"]
        
        # ValueError for duplicate key with register
        registry.register("item1", "value1")
        with pytest.raises(ValueError, match="already exists"):
            registry.register("item1", "value2")
        
        # KeyError when adding alias for non-existent key
        with pytest.raises(KeyError, match="'missing' not found"):
            registry.add_alias("alias", "missing")
    
    def test_overwrite_warning(self):
        """Test warning when overwriting existing item."""
        registry = Registry("test")
        registry["item1"] = "value1"
        
        # Should warn when overwriting
        with pytest.warns(UserWarning, match="Overwriting existing item"):
            registry["item1"] = "value2"
        
        assert registry["item1"] == "value2"
    
    def test_force_register(self):
        """Test force registration."""
        registry = Registry("test")
        registry.register("item1", "value1")
        
        # Force should not raise error
        registry.register("item1", "value2", force=True)
        assert registry["item1"] == "value2"