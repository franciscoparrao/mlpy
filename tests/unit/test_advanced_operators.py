"""Tests for advanced pipeline operators."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from mlpy.tasks import TaskClassif, TaskRegr
from mlpy.pipelines import (
    PipeOpPCA, PipeOpTargetEncode, PipeOpOutlierDetect,
    PipeOpBin, PipeOpTextVectorize, PipeOpPolynomial
)


@pytest.fixture
def regression_task():
    """Create a regression task for testing."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=10, n_informative=5, noise=0.1)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    return TaskRegr(data=df, target='target', id='test_regr')


@pytest.fixture
def classification_task():
    """Create a classification task for testing."""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_classes=2)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    return TaskClassif(data=df, target='target', id='test_classif')


@pytest.fixture
def mixed_task():
    """Create a task with mixed data types."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples) * 2 + 1,
        'categorical': np.random.choice(['A', 'B', 'C'], n_samples),
        'text': ['sample text ' + str(i) for i in range(n_samples)],
        'target': np.random.randn(n_samples)
    })
    
    return TaskRegr(data=df, target='target', id='test_mixed')


class TestPipeOpPCA:
    """Test PCA operator."""
    
    def test_pca_basic(self, regression_task):
        """Test basic PCA functionality."""
        op = PipeOpPCA(n_components=5)
        
        # Train
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Check output
        assert transformed_task.ncol == 6  # 5 PCs + target
        assert all(col.startswith('PC') for col in transformed_task.feature_names)
        assert len(transformed_task.feature_names) == 5
        
        # Check explained variance is stored
        assert 'explained_variance_ratio' in op.state
        assert len(op.state['explained_variance_ratio']) == 5
        
    def test_pca_variance_threshold(self, regression_task):
        """Test PCA with variance threshold."""
        op = PipeOpPCA(n_components=0.95)  # Keep 95% variance
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have fewer components than original features
        assert len(transformed_task.feature_names) < 10
        
    def test_pca_predict(self, regression_task):
        """Test PCA prediction."""
        op = PipeOpPCA(n_components=3)
        
        # Train
        op.train({'input': regression_task})
        
        # Predict on same data
        result = op.predict({'input': regression_task})
        transformed_task = result['output']
        
        assert len(transformed_task.feature_names) == 3
        assert transformed_task.nrow == regression_task.nrow
        
    def test_pca_no_numeric_features(self, mixed_task):
        """Test PCA with non-numeric features."""
        # Remove numeric features
        data = mixed_task.data()[['categorical', 'text', 'target']]
        task = TaskRegr(data=data, target='target')
        
        op = PipeOpPCA()
        result = op.train({'input': task})
        
        # Should return unchanged task
        assert result['output'].ncol == task.ncol


class TestPipeOpTargetEncode:
    """Test target encoding operator."""
    
    def test_target_encode_basic(self, mixed_task):
        """Test basic target encoding."""
        op = PipeOpTargetEncode(columns=['categorical'])
        
        result = op.train({'input': mixed_task})
        transformed_task = result['output']
        
        # Categorical column should be numeric now
        assert pd.api.types.is_numeric_dtype(transformed_task.data()['categorical'])
        
        # Check encodings are stored
        assert 'categorical' in op._encodings
        assert len(op._encodings['categorical']) == 3  # A, B, C
        
    def test_target_encode_smoothing(self, mixed_task):
        """Test target encoding with smoothing."""
        op1 = PipeOpTargetEncode(columns=['categorical'], smoothing=0)
        op2 = PipeOpTargetEncode(columns=['categorical'], smoothing=10)
        
        result1 = op1.train({'input': mixed_task})
        result2 = op2.train({'input': mixed_task})
        
        # With more smoothing, values should be closer to global mean
        values1 = result1['output'].data()['categorical'].unique()
        values2 = result2['output'].data()['categorical'].unique()
        
        # Check that smoothing reduces variance
        assert np.var(values2) < np.var(values1)
        
    def test_target_encode_classification(self, classification_task):
        """Test target encoding for classification."""
        # Add categorical feature
        data = classification_task.data()
        data['category'] = np.random.choice(['X', 'Y', 'Z'], len(data))
        task = TaskClassif(data=data, target='target')
        
        op = PipeOpTargetEncode(columns=['category'])
        result = op.train({'input': task})
        
        # Should encode to probabilities (0-1 range)
        encoded_values = result['output'].data()['category']
        assert encoded_values.min() >= 0
        assert encoded_values.max() <= 1
        
    def test_target_encode_unknown_categories(self, mixed_task):
        """Test handling of unknown categories."""
        op = PipeOpTargetEncode(columns=['categorical'])
        op.train({'input': mixed_task})
        
        # Create new data with unknown category
        new_data = mixed_task.data().copy()
        new_data.loc[0, 'categorical'] = 'Unknown'
        new_task = TaskRegr(data=new_data, target='target')
        
        result = op.predict({'input': new_task})
        
        # Unknown category should get global mean
        encoded_value = result['output'].data().loc[0, 'categorical']
        assert encoded_value == op._global_mean


class TestPipeOpOutlierDetect:
    """Test outlier detection operator."""
    
    def test_outlier_detect_isolation(self, regression_task):
        """Test isolation forest outlier detection."""
        op = PipeOpOutlierDetect(method='isolation', contamination=0.1, action='flag')
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have outlier flag column
        assert 'is_outlier' in transformed_task.data().columns
        
        # Check approximately 10% marked as outliers
        outlier_ratio = transformed_task.data()['is_outlier'].mean()
        assert 0.05 < outlier_ratio < 0.15
        
    def test_outlier_detect_remove(self, regression_task):
        """Test outlier removal."""
        op = PipeOpOutlierDetect(method='isolation', contamination=0.1, action='remove')
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have fewer rows
        assert transformed_task.nrow < regression_task.nrow
        assert transformed_task.nrow == pytest.approx(90, abs=5)
        
    def test_outlier_detect_impute(self, regression_task):
        """Test outlier imputation."""
        op = PipeOpOutlierDetect(method='elliptic', contamination=0.1, action='impute')
        
        # Add some extreme values
        data = regression_task.data().copy()
        data.iloc[0, 0] = 1000  # Extreme outlier
        task = TaskRegr(data=data, target='target')
        
        result = op.train({'input': task})
        transformed_task = result['output']
        
        # Extreme value should be replaced
        assert transformed_task.data().iloc[0, 0] != 1000
        assert transformed_task.nrow == task.nrow
        
    def test_outlier_detect_methods(self, regression_task):
        """Test different outlier detection methods."""
        methods = ['isolation', 'elliptic', 'lof']
        
        for method in methods:
            op = PipeOpOutlierDetect(method=method, contamination=0.1)
            result = op.train({'input': regression_task})
            
            assert op.state['n_outliers'] > 0
            assert op.state['outlier_fraction'] < 0.2


class TestPipeOpBin:
    """Test binning operator."""
    
    def test_bin_ordinal(self, regression_task):
        """Test ordinal binning."""
        op = PipeOpBin(n_bins=5, strategy='quantile', encode='ordinal')
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Check bins are created
        for col in regression_task.feature_names:
            values = transformed_task.data()[col].unique()
            assert len(values) <= 5
            assert all(isinstance(v, (int, float)) for v in values)
            
    def test_bin_onehot(self, regression_task):
        """Test one-hot binning."""
        op = PipeOpBin(
            n_bins=3,
            columns=['feature_0', 'feature_1'],
            strategy='uniform',
            encode='onehot'
        )
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Original columns should be replaced with bin columns
        assert 'feature_0' not in transformed_task.data().columns
        assert 'feature_1' not in transformed_task.data().columns
        
        # Should have bin columns
        bin_cols = [col for col in transformed_task.data().columns if '_bin_' in col]
        assert len(bin_cols) == 6  # 2 features * 3 bins
        
    def test_bin_strategies(self, regression_task):
        """Test different binning strategies."""
        strategies = ['uniform', 'quantile', 'kmeans']
        
        for strategy in strategies:
            op = PipeOpBin(n_bins=4, strategy=strategy)
            result = op.train({'input': regression_task})
            
            # Check all features are binned
            assert op.state['binned_columns'] == regression_task.feature_names


class TestPipeOpTextVectorize:
    """Test text vectorization operator."""
    
    def test_text_vectorize_tfidf(self, mixed_task):
        """Test TF-IDF vectorization."""
        op = PipeOpTextVectorize(
            columns=['text'],
            method='tfidf',
            max_features=20,
            ngram_range=(1, 2)
        )
        
        result = op.train({'input': mixed_task})
        transformed_task = result['output']
        
        # Original text column should be replaced
        assert 'text' not in transformed_task.data().columns
        
        # Should have TF-IDF features
        tfidf_cols = [col for col in transformed_task.data().columns if col.startswith('text_')]
        assert len(tfidf_cols) <= 20
        
        # Check values are TF-IDF scores
        for col in tfidf_cols:
            values = transformed_task.data()[col]
            assert values.min() >= 0
            assert values.max() <= 1
            
    def test_text_vectorize_count(self, mixed_task):
        """Test count vectorization."""
        op = PipeOpTextVectorize(
            columns=['text'],
            method='count',
            max_features=10
        )
        
        result = op.train({'input': mixed_task})
        transformed_task = result['output']
        
        # Check count features
        count_cols = [col for col in transformed_task.data().columns if col.startswith('text_')]
        assert len(count_cols) <= 10
        
        # Values should be counts (integers)
        for col in count_cols:
            values = transformed_task.data()[col]
            assert all(v == int(v) for v in values)
            
    def test_text_vectorize_predict(self, mixed_task):
        """Test text vectorization on new data."""
        op = PipeOpTextVectorize(columns=['text'], max_features=15)
        op.train({'input': mixed_task})
        
        # New data with different text
        new_data = mixed_task.data().copy()
        new_data['text'] = ['new text ' + str(i) for i in range(len(new_data))]
        new_task = TaskRegr(data=new_data, target='target')
        
        result = op.predict({'input': new_task})
        
        # Should have same features as training
        train_features = op.state['feature_names']['text']
        pred_features = [col for col in result['output'].data().columns if col.startswith('text_')]
        assert len(pred_features) == len(train_features)


class TestPipeOpPolynomial:
    """Test polynomial features operator."""
    
    def test_polynomial_degree2(self, regression_task):
        """Test degree 2 polynomial features."""
        # Use fewer features for manageable output
        op = PipeOpPolynomial(
            degree=2,
            columns=['feature_0', 'feature_1'],
            include_bias=False
        )
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have: 1 + x1 + x2 + x1^2 + x1*x2 + x2^2 = 6 features (no bias)
        poly_features = [col for col in transformed_task.feature_names if 'feature_' in col]
        assert len(poly_features) == 5  # excluding '1' (bias)
        
        # Check feature names
        assert 'feature_0' in poly_features
        assert 'feature_1' in poly_features
        assert 'feature_0^2' in poly_features
        assert 'feature_1^2' in poly_features
        assert any('feature_0 feature_1' in col for col in poly_features)
        
    def test_polynomial_interaction_only(self, regression_task):
        """Test interaction-only features."""
        op = PipeOpPolynomial(
            degree=2,
            columns=['feature_0', 'feature_1', 'feature_2'],
            interaction_only=True,
            include_bias=False
        )
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have original features + interactions
        poly_features = transformed_task.feature_names
        
        # Check no squared terms
        assert not any('^2' in col for col in poly_features)
        
        # Check interactions exist
        assert any('feature_0 feature_1' in col for col in poly_features)
        assert any('feature_0 feature_2' in col for col in poly_features)
        assert any('feature_1 feature_2' in col for col in poly_features)
        
    def test_polynomial_high_degree(self, regression_task):
        """Test higher degree polynomials."""
        op = PipeOpPolynomial(
            degree=3,
            columns=['feature_0'],  # Single feature to avoid explosion
            include_bias=True
        )
        
        result = op.train({'input': regression_task})
        transformed_task = result['output']
        
        # Should have: 1 + x + x^2 + x^3 = 4 features
        poly_features = [col for col in transformed_task.data().columns if 'feature' in col or col == '1']
        assert len(poly_features) == 4
        
        # Check powers
        assert '1' in poly_features  # bias
        assert 'feature_0' in poly_features
        assert 'feature_0^2' in poly_features
        assert 'feature_0^3' in poly_features


class TestOperatorIntegration:
    """Test integration of multiple operators."""
    
    def test_pca_after_scaling(self, regression_task):
        """Test PCA works after scaling."""
        from mlpy.pipelines import PipeOpScale, linear_pipeline
        
        pipeline = linear_pipeline(
            PipeOpScale(id="scale", method="standard"),
            PipeOpPCA(id="pca", n_components=5)
        )
        
        result = pipeline.train(regression_task)
        final_task = result['output']
        
        assert len(final_task.feature_names) == 5
        assert all(col.startswith('PC') for col in final_task.feature_names)
        
    def test_complex_pipeline(self, mixed_task):
        """Test complex pipeline with multiple operators."""
        from mlpy.pipelines import PipeOpEncode, PipeOpScale, linear_pipeline
        
        # Build pipeline: encode -> target encode -> scale -> bin
        pipeline = linear_pipeline(
            PipeOpEncode(id="encode", method="onehot"),
            PipeOpTargetEncode(id="target_enc", columns=None),  # Auto-detect
            PipeOpScale(id="scale"),
            PipeOpBin(id="bin", n_bins=3, encode="ordinal")
        )
        
        result = pipeline.train(mixed_task)
        final_task = result['output']
        
        # Check pipeline executed successfully
        assert final_task.nrow == mixed_task.nrow
        
        # All features should be numeric and binned
        for col in final_task.feature_names:
            values = final_task.data()[col]
            assert pd.api.types.is_numeric_dtype(values)
            assert len(values.unique()) <= 3  # Binned to 3 bins
            
    def test_outlier_then_pca(self, regression_task):
        """Test outlier removal before PCA."""
        from mlpy.pipelines import linear_pipeline
        
        pipeline = linear_pipeline(
            PipeOpOutlierDetect(id="outlier", action="remove", contamination=0.1),
            PipeOpPCA(id="pca", n_components=0.95)
        )
        
        result = pipeline.train(regression_task)
        final_task = result['output']
        
        # Should have fewer rows and fewer features
        assert final_task.nrow < regression_task.nrow
        assert len(final_task.feature_names) < len(regression_task.feature_names)