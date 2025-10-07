"""Unit tests for resampling strategies."""

import pytest
import numpy as np
import pandas as pd

from mlpy.resamplings import (
    Resampling, ResamplingHoldout, ResamplingCV,
    ResamplingLOO, ResamplingRepeatedCV,
    ResamplingBootstrap, ResamplingSubsampling
)
from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.backends import DataBackendPandas
from mlpy.utils.registry import mlpy_resamplings


class TestResamplingBase:
    """Test base Resampling functionality."""
    
    @pytest.fixture
    def simple_task(self):
        """Create a simple classification task."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'x2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'y': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })
        backend = DataBackendPandas(df)
        return TaskClassif(
            data=backend,
            target='y',
            id='test_task'
        )
        
    def test_resampling_properties(self):
        """Test resampling initialization and properties."""
        resampling = ResamplingHoldout(ratio=0.8, seed=42)
        
        assert resampling.id == 'holdout'
        assert resampling.iters == 1
        assert resampling.ratio == 0.8
        assert resampling.seed == 42
        assert not resampling.duplicated_ids
        assert not resampling.is_instantiated
        
    def test_resampling_registry(self):
        """Test that resamplings are registered."""
        # Check some resamplings are registered
        assert 'holdout' in mlpy_resamplings
        assert 'cv' in mlpy_resamplings
        assert 'bootstrap' in mlpy_resamplings
        assert 'subsampling' in mlpy_resamplings
        
        # Can retrieve resamplings
        holdout = mlpy_resamplings['holdout']
        assert isinstance(holdout, ResamplingHoldout)
        
    def test_instantiation(self, simple_task):
        """Test resampling instantiation."""
        resampling = ResamplingHoldout(ratio=0.7)
        
        # Not instantiated yet
        assert not resampling.is_instantiated
        
        # Instantiate
        resampling.instantiate(simple_task)
        assert resampling.is_instantiated
        
        # Should have train/test sets
        train = resampling.train_set(0)
        test = resampling.test_set(0)
        
        assert len(train) == 7  # 70% of 10
        assert len(test) == 3   # 30% of 10
        
        # No overlap
        assert len(np.intersect1d(train, test)) == 0
        
        # All indices covered
        all_indices = np.concatenate([train, test])
        assert set(all_indices) == set(range(10))
        
    def test_iteration(self, simple_task):
        """Test iterating over resampling splits."""
        resampling = ResamplingCV(folds=3)
        resampling.instantiate(simple_task)
        
        splits = list(resampling)
        assert len(splits) == 3
        
        for i, (train, test) in enumerate(splits):
            # Check same as direct access
            assert np.array_equal(train, resampling.train_set(i))
            assert np.array_equal(test, resampling.test_set(i))
            
    def test_not_instantiated_error(self):
        """Test error when accessing splits before instantiation."""
        resampling = ResamplingHoldout()
        
        with pytest.raises(RuntimeError, match="must be instantiated"):
            resampling.train_set(0)
            
        with pytest.raises(RuntimeError, match="must be instantiated"):
            resampling.test_set(0)
            
        with pytest.raises(RuntimeError, match="must be instantiated"):
            list(resampling)
            
    def test_index_out_of_range(self, simple_task):
        """Test error for invalid iteration index."""
        resampling = ResamplingHoldout()
        resampling.instantiate(simple_task)
        
        with pytest.raises(IndexError, match="out of range"):
            resampling.train_set(1)  # Only has 1 iteration
            
        with pytest.raises(IndexError, match="out of range"):
            resampling.test_set(-1)
            
    def test_clone(self, simple_task):
        """Test cloning resampling."""
        resampling = ResamplingHoldout(ratio=0.75, seed=123)
        resampling.instantiate(simple_task)
        
        # Clone
        cloned = resampling.clone()
        
        # Properties preserved
        assert cloned.ratio == 0.75
        assert cloned.seed == 123
        
        # But not instantiated
        assert not cloned.is_instantiated


class TestResamplingHoldout:
    """Test holdout resampling."""
    
    def test_holdout_basic(self):
        """Test basic holdout functionality."""
        # Create task with 100 samples
        df = pd.DataFrame({
            'x': range(100),
            'y': np.random.choice(['A', 'B'], 100)
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingHoldout(ratio=0.8, seed=42)
        resampling.instantiate(task)
        
        train = resampling.train_set(0)
        test = resampling.test_set(0)
        
        assert len(train) == 80
        assert len(test) == 20
        assert len(np.unique(train)) == 80  # No duplicates
        assert len(np.unique(test)) == 20
        
    def test_holdout_stratified(self):
        """Test stratified holdout."""
        # Create imbalanced task
        y = ['A'] * 80 + ['B'] * 20  # 80% A, 20% B
        df = pd.DataFrame({
            'x': range(100),
            'y': y
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingHoldout(ratio=0.7, stratify=True, seed=42)
        resampling.instantiate(task)
        
        train = resampling.train_set(0)
        test = resampling.test_set(0)
        
        # Check stratification
        train_labels = task.data(rows=train, cols=['y'])['y'].values
        test_labels = task.data(rows=test, cols=['y'])['y'].values
        
        # Proportions should be maintained
        train_prop_A = (train_labels == 'A').sum() / len(train_labels)
        test_prop_A = (test_labels == 'A').sum() / len(test_labels)
        
        assert abs(train_prop_A - 0.8) < 0.05
        assert abs(test_prop_A - 0.8) < 0.05
        
    def test_holdout_deterministic(self):
        """Test that holdout is deterministic with seed."""
        df = pd.DataFrame({'x': range(50), 'y': range(50)})
        task = TaskRegr(data=df, target='y', id='test')
        
        # Same seed should give same splits
        r1 = ResamplingHoldout(ratio=0.6, seed=123)
        r2 = ResamplingHoldout(ratio=0.6, seed=123)
        
        r1.instantiate(task)
        r2.instantiate(task)
        
        assert np.array_equal(r1.train_set(0), r2.train_set(0))
        assert np.array_equal(r1.test_set(0), r2.test_set(0))
        
        # Different seed should give different splits
        r3 = ResamplingHoldout(ratio=0.6, seed=456)
        r3.instantiate(task)
        
        assert not np.array_equal(r1.train_set(0), r3.train_set(0))
        
    def test_holdout_invalid_ratio(self):
        """Test error for invalid ratio."""
        with pytest.raises(ValueError, match="ratio must be between"):
            ResamplingHoldout(ratio=0)
            
        with pytest.raises(ValueError, match="ratio must be between"):
            ResamplingHoldout(ratio=1)
            
        with pytest.raises(ValueError, match="ratio must be between"):
            ResamplingHoldout(ratio=1.5)


class TestResamplingCV:
    """Test cross-validation resampling."""
    
    def test_cv_basic(self):
        """Test basic CV functionality."""
        df = pd.DataFrame({
            'x': range(10),
            'y': range(10)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingCV(folds=5, seed=42)
        resampling.instantiate(task)
        
        assert resampling.iters == 5
        
        # Check each fold
        all_test_indices = []
        for i in range(5):
            train = resampling.train_set(i)
            test = resampling.test_set(i)
            
            assert len(train) == 8  # 4/5 of 10
            assert len(test) == 2   # 1/5 of 10
            
            # No overlap
            assert len(np.intersect1d(train, test)) == 0
            
            # Union is complete
            assert set(np.concatenate([train, test])) == set(range(10))
            
            all_test_indices.extend(test)
            
        # Each sample appears exactly once as test
        assert sorted(all_test_indices) == list(range(10))
        
    def test_cv_unequal_folds(self):
        """Test CV with unequal fold sizes."""
        # 13 samples don't divide evenly by 5
        df = pd.DataFrame({
            'x': range(13),
            'y': range(13)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingCV(folds=5)
        resampling.instantiate(task)
        
        fold_sizes = []
        for i in range(5):
            test = resampling.test_set(i)
            fold_sizes.append(len(test))
            
        # First 3 folds get 3 samples, last 2 get 2
        assert fold_sizes == [3, 3, 3, 2, 2]
        
    def test_cv_stratified(self):
        """Test stratified CV."""
        # Create data with 3 classes
        y = ['A'] * 30 + ['B'] * 30 + ['C'] * 30
        df = pd.DataFrame({
            'x': range(90),
            'y': y
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingCV(folds=3, stratify=True, seed=42)
        resampling.instantiate(task)
        
        for i in range(3):
            test = resampling.test_set(i)
            test_labels = task.data(rows=test, cols=['y'])['y'].values
            
            # Each fold should have equal representation
            unique, counts = np.unique(test_labels, return_counts=True)
            assert len(unique) == 3  # All classes present
            assert all(counts == 10)  # 10 of each class
            
    def test_cv_loo(self):
        """Test leave-one-out CV."""
        df = pd.DataFrame({
            'x': range(5),
            'y': ['A', 'B', 'A', 'B', 'A']
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingLOO()
        resampling.instantiate(task)
        
        assert resampling.iters == 5
        
        for i in range(5):
            train = resampling.train_set(i)
            test = resampling.test_set(i)
            
            assert len(train) == 4
            assert len(test) == 1
            assert test[0] == i
            
    def test_cv_too_many_folds(self):
        """Test error when folds > samples."""
        df = pd.DataFrame({'x': [1, 2], 'y': [1, 2]})
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingCV(folds=3)
        
        with pytest.raises(ValueError, match="more folds.*than samples"):
            resampling.instantiate(task)
            
    def test_cv_invalid_folds(self):
        """Test error for invalid number of folds."""
        with pytest.raises(ValueError, match="at least 2"):
            ResamplingCV(folds=1)


class TestResamplingRepeatedCV:
    """Test repeated cross-validation."""
    
    def test_repeated_cv_basic(self):
        """Test basic repeated CV."""
        df = pd.DataFrame({
            'x': range(10),
            'y': range(10)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingRepeatedCV(folds=2, repeats=3, seed=42)
        resampling.instantiate(task)
        
        assert resampling.iters == 6  # 2 folds * 3 repeats
        
        # Collect test sets for each repeat
        repeat_test_sets = []
        for r in range(3):
            repeat_tests = []
            for f in range(2):
                i = r * 2 + f
                test = resampling.test_set(i)
                repeat_tests.append(set(test))
            repeat_test_sets.append(repeat_tests)
            
        # Different repeats should have different splits
        assert repeat_test_sets[0] != repeat_test_sets[1]
        assert repeat_test_sets[1] != repeat_test_sets[2]
        
    def test_repeated_cv_stratified(self):
        """Test stratified repeated CV."""
        y = ['A'] * 50 + ['B'] * 50
        df = pd.DataFrame({
            'x': range(100),
            'y': y
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingRepeatedCV(
            folds=5, repeats=2, stratify=True, seed=42
        )
        resampling.instantiate(task)
        
        # Check stratification is maintained
        for i in range(10):  # 5 folds * 2 repeats
            test = resampling.test_set(i)
            test_labels = task.data(rows=test, cols=['y'])['y'].values
            
            prop_A = (test_labels == 'A').sum() / len(test_labels)
            assert abs(prop_A - 0.5) < 0.1


class TestResamplingBootstrap:
    """Test bootstrap resampling."""
    
    def test_bootstrap_basic(self):
        """Test basic bootstrap with OOB."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingBootstrap(
            iters=10, ratio=1.0, oob=True, seed=42
        )
        resampling.instantiate(task)
        
        assert resampling.iters == 10
        assert resampling.duplicated_ids  # Bootstrap allows duplicates
        
        for i in range(10):
            train = resampling.train_set(i)
            test = resampling.test_set(i)
            
            assert len(train) == 100  # Same size as original
            assert len(np.unique(train)) < 100  # Has duplicates
            
            # OOB samples are those not in train
            expected_oob = set(range(100)) - set(train)
            assert set(test) == expected_oob or len(test) > 0
            
    def test_bootstrap_fixed_test(self):
        """Test bootstrap with fixed test set."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingBootstrap(
            iters=5, ratio=1.0, oob=False, test_ratio=0.2, seed=42
        )
        resampling.instantiate(task)
        
        # Test set should be same across iterations
        test_sets = []
        for i in range(5):
            test = resampling.test_set(i)
            test_sets.append(set(test))
            
        # All test sets are identical
        assert all(ts == test_sets[0] for ts in test_sets)
        assert len(test_sets[0]) == 20  # 20% of 100
        
    def test_bootstrap_stratified(self):
        """Test stratified bootstrap."""
        y = ['A'] * 70 + ['B'] * 30
        df = pd.DataFrame({
            'x': range(100),
            'y': y
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingBootstrap(
            iters=10, ratio=1.0, stratify=True, seed=42
        )
        resampling.instantiate(task)
        
        for i in range(10):
            train = resampling.train_set(i)
            train_labels = task.data(rows=train, cols=['y'])['y'].values
            
            # Check proportions are maintained
            prop_A = (train_labels == 'A').sum() / len(train_labels)
            assert abs(prop_A - 0.7) < 0.1
            
    def test_bootstrap_smaller_sample(self):
        """Test bootstrap with ratio < 1."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingBootstrap(
            iters=5, ratio=0.5, oob=True, seed=42
        )
        resampling.instantiate(task)
        
        for i in range(5):
            train = resampling.train_set(i)
            assert len(train) == 50  # 50% of 100


class TestResamplingSubsampling:
    """Test subsampling (Monte Carlo CV)."""
    
    def test_subsampling_basic(self):
        """Test basic subsampling."""
        df = pd.DataFrame({
            'x': range(100),
            'y': range(100)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        resampling = ResamplingSubsampling(
            iters=10, ratio=0.8, seed=42
        )
        resampling.instantiate(task)
        
        assert resampling.iters == 10
        
        # Different iterations should have different splits
        train_sets = []
        for i in range(10):
            train = resampling.train_set(i)
            test = resampling.test_set(i)
            
            assert len(train) == 80
            assert len(test) == 20
            assert len(np.unique(train)) == 80  # No duplicates
            
            # No overlap between train and test
            assert len(np.intersect1d(train, test)) == 0
            
            train_sets.append(set(train))
            
        # Different iterations should have different train sets
        assert len(set(tuple(ts) for ts in train_sets)) > 1
        
    def test_subsampling_stratified(self):
        """Test stratified subsampling."""
        y = ['A'] * 60 + ['B'] * 40
        df = pd.DataFrame({
            'x': range(100),
            'y': y
        })
        task = TaskClassif(data=df, target='y', id='test')
        
        resampling = ResamplingSubsampling(
            iters=10, ratio=0.7, stratify=True, seed=42
        )
        resampling.instantiate(task)
        
        for i in range(10):
            train = resampling.train_set(i)
            train_labels = task.data(rows=train, cols=['y'])['y'].values
            
            # Proportions maintained
            prop_A = (train_labels == 'A').sum() / len(train_labels)
            assert abs(prop_A - 0.6) < 0.1


class TestResamplingIntegration:
    """Integration tests for resamplings."""
    
    def test_all_resamplings_registered(self):
        """Test that all resamplings are in registry."""
        expected = [
            'holdout', 'cv', 'loo', 'repeated_cv',
            'bootstrap', 'subsampling'
        ]
        
        for name in expected:
            assert name in mlpy_resamplings
            
    def test_resampling_with_missing_data(self):
        """Test resamplings work with tasks that have row roles."""
        df = pd.DataFrame({
            'x': range(20),
            'y': range(20)
        })
        task = TaskRegr(data=df, target='y', id='test')
        
        # Use only subset of rows
        task.set_row_roles({'use': list(range(10))})
        
        # All resamplings should work with subset
        for ResamplClass in [ResamplingHoldout, ResamplingCV, 
                             ResamplingBootstrap, ResamplingSubsampling]:
            resampling = ResamplClass()
            resampling.instantiate(task)
            
            train = resampling.train_set(0)
            test = resampling.test_set(0)
            
            # All indices should be from the 'use' subset
            assert all(idx < 10 for idx in train)
            assert all(idx < 10 for idx in test)