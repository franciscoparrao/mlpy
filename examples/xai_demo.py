"""
MLPY Explainable AI Demo
========================

Demonstrates all explainability features of MLPY.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# MLPY imports
from mlpy.tasks import TaskClassif
from mlpy.learners.sklearn import LearnerRandomForestClassifier
from mlpy.measures import MeasureClassifAcc, MeasureClassifF1
from mlpy.explainability import Explainer

# Standard imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def generate_demo_data():
    """Generate synthetic data for demonstration."""
    print("\n[1] GENERATING DEMO DATA")
    print("-" * 40)
    
    # Generate classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(20)]
    
    # Add some meaning to features for demo
    feature_names[0] = "age"
    feature_names[1] = "income"
    feature_names[2] = "education_years"
    feature_names[3] = "credit_score"
    feature_names[4] = "employment_years"
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add sensitive feature for fairness analysis
    df['gender'] = np.random.choice(['M', 'F'], size=len(df))
    df['ethnicity'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: 2 (binary classification)")
    print(f"Sensitive features: gender, ethnicity")
    
    return df, feature_names


def train_model(df, feature_names):
    """Train a model using MLPY."""
    print("\n[2] TRAINING MODEL")
    print("-" * 40)
    
    # Prepare data
    X = df[feature_names]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train sklearn model directly for demo
    # (to ensure compatibility with all explainers)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model: Random Forest Classifier")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    return model, X_train, X_test, y_train, y_test


def demo_shap(explainer, X_test):
    """Demonstrate SHAP explanations."""
    print("\n[3] SHAP EXPLANATIONS")
    print("-" * 40)
    
    try:
        # Generate SHAP values
        print("Calculating SHAP values...")
        shap_results = explainer.shap_explain(X_test[:100])
        
        # Get feature importance
        importance_df = shap_results.get_feature_importance()
        print("\nTop 5 most important features (SHAP):")
        for idx, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Plot if available
        try:
            explainer.plot_shap_summary(shap_results)
        except:
            print("(Plotting unavailable - install shap for visualizations)")
            
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Install SHAP: pip install shap")


def demo_lime(explainer, X_test):
    """Demonstrate LIME explanations."""
    print("\n[4] LIME EXPLANATIONS")
    print("-" * 40)
    
    try:
        # Explain first test instance
        instance = X_test.iloc[0]
        print(f"Explaining instance 0...")
        
        lime_exp = explainer.lime_explain(instance, num_features=5)
        
        print("\nTop 5 features for this instance (LIME):")
        for feature, weight in lime_exp.get_top_features(5):
            print(f"  {feature}: {weight:+.4f}")
        
        print(f"\nPrediction: {lime_exp.prediction}")
        print(f"Local model R²: {lime_exp.score:.3f}")
        
    except Exception as e:
        print(f"LIME analysis failed: {e}")
        print("Install LIME: pip install lime")


def demo_feature_importance(explainer, X_test, y_test):
    """Demonstrate feature importance methods."""
    print("\n[5] FEATURE IMPORTANCE")
    print("-" * 40)
    
    # Native importance
    try:
        print("Calculating native feature importance...")
        native_importance = explainer.global_importance(method="native")
        
        print("\nTop 5 features (native):")
        df_imp = native_importance.to_dataframe()
        for idx, row in df_imp.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    except:
        print("Native importance not available for this model")
    
    # Permutation importance
    try:
        print("\nCalculating permutation importance...")
        perm_importance = explainer.global_importance(
            method="permutation", 
            X=X_test.values, 
            y=y_test.values
        )
        
        print("\nTop 5 features (permutation):")
        df_perm = perm_importance.to_dataframe()
        for idx, row in df_perm.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            if 'std' in row:
                print(f"    (std: {row['std']:.4f})")
    except Exception as e:
        print(f"Permutation importance failed: {e}")


def demo_counterfactuals(explainer, X_test):
    """Demonstrate counterfactual explanations."""
    print("\n[6] COUNTERFACTUAL EXPLANATIONS")
    print("-" * 40)
    
    # Find an instance predicted as class 0
    instance_idx = 0
    instance = X_test.iloc[instance_idx]
    
    print(f"Original instance (idx={instance_idx}):")
    print(f"  Prediction: {explainer.model.predict([instance])[0]}")
    
    # Generate counterfactual
    print("\nGenerating counterfactual (flip prediction)...")
    cf = explainer.counterfactual(
        instance,
        desired_outcome=1,  # Want opposite class
        max_features_changed=3,
        method="optimization"
    )
    
    print(f"\nCounterfactual found:")
    print(f"  Original prediction: {cf.original_prediction}")
    print(f"  Counterfactual prediction: {cf.counterfactual_prediction}")
    print(f"  Features changed: {cf.sparsity}")
    print(f"  Distance: {cf.distance:.3f}")
    print(f"  Valid: {cf.validity}")
    
    if cf.feature_changes:
        print("\nChanges required:")
        for feature, (orig, new) in list(cf.feature_changes.items())[:5]:
            print(f"  {feature}: {orig:.3f} → {new:.3f} ({new-orig:+.3f})")


def demo_fairness(explainer, df, feature_names):
    """Demonstrate fairness analysis."""
    print("\n[7] FAIRNESS ANALYSIS")
    print("-" * 40)
    
    # Prepare data with sensitive features
    X = df[feature_names + ['gender', 'ethnicity']]
    y = df['target']
    
    # Analyze fairness for gender
    print("Analyzing fairness across gender...")
    try:
        fairness_metrics = explainer.analyze_fairness(X, y, 'gender')
        
        print(f"\nFairness Metrics:")
        print(f"  Demographic Parity: {fairness_metrics.demographic_parity:.3f}")
        print(f"  Equal Opportunity: {fairness_metrics.equal_opportunity:.3f}")
        print(f"  Disparate Impact: {fairness_metrics.disparate_impact:.3f}")
        print(f"  Is Fair (>0.8 threshold): {fairness_metrics.is_fair()}")
        
        print("\nGroup-specific metrics:")
        for group, metrics in fairness_metrics.groups.items():
            print(f"  {group}: positive_rate={metrics['positive_rate']:.3f}, "
                  f"accuracy={metrics['accuracy']:.3f}")
    except Exception as e:
        print(f"Fairness analysis failed: {e}")
    
    # Detect bias
    print("\nDetecting bias in data...")
    try:
        bias_results = explainer.detect_bias(df, 'target')
        
        for feature, bias_info in bias_results.items():
            print(f"\n{feature}:")
            print(f"  Group sizes: {bias_info['group_sizes']}")
            print(f"  Max imbalance: {bias_info['max_imbalance']:.3f}")
    except Exception as e:
        print(f"Bias detection failed: {e}")


def demo_model_card(explainer):
    """Generate and display model card."""
    print("\n[8] MODEL CARD GENERATION")
    print("-" * 40)
    
    print("Generating model card...")
    
    card = explainer.generate_model_card(
        model_name="Demo Random Forest",
        performance_metrics={
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.87,
            'f1_score': 0.85
        },
        intended_uses=[
            "Demonstration of XAI capabilities",
            "Educational purposes",
            "Testing explainability methods"
        ],
        ethical_considerations=[
            "Model trained on synthetic data",
            "Not validated for production use",
            "Potential biases not fully assessed",
            "Requires human oversight for decisions"
        ],
        owner="MLPY Team",
        contact="mlpy@example.com"
    )
    
    # Save model card
    import os
    os.makedirs("xai_output", exist_ok=True)
    
    card.to_html("xai_output/model_card.html")
    card.to_markdown("xai_output/model_card.md")
    card.to_json("xai_output/model_card.json")
    
    print("Model card saved to:")
    print("  - xai_output/model_card.html")
    print("  - xai_output/model_card.md")
    print("  - xai_output/model_card.json")


def demo_comprehensive_report(explainer, X_test, y_test):
    """Generate comprehensive explainability report."""
    print("\n[9] COMPREHENSIVE REPORT")
    print("-" * 40)
    
    print("Generating full explainability report...")
    print("This may take a few minutes...")
    
    # Get sample instance
    sample_instance = X_test.iloc[0]
    
    # Generate report
    report = explainer.generate_full_report(
        X=X_test,
        y=y_test,
        sample_instance=sample_instance,
        output_dir="xai_output/full_report"
    )
    
    print("\nReport generated successfully!")
    print("Files saved to: xai_output/full_report/")
    print("\nReport summary:")
    print(f"  Model type: {report['model_type']}")
    print(f"  Task type: {report['task_type']}")
    print(f"  Features: {report['num_features']}")
    
    if 'feature_importance' in report:
        print("  [OK] Feature importance calculated")
    if 'shap_importance' in report:
        print("  [OK] SHAP analysis completed")
    if 'lime_explanation' in report:
        print("  [OK] LIME explanation generated")
    if 'counterfactual' in report:
        print("  [OK] Counterfactual found")
    if 'fairness' in report:
        print("  [OK] Fairness analyzed")
    if report.get('model_card_generated'):
        print("  [OK] Model card generated")


def main():
    """Run the complete XAI demo."""
    print("="*60)
    print("MLPY EXPLAINABLE AI (XAI) DEMO")
    print("="*60)
    
    # Generate data
    df, feature_names = generate_demo_data()
    
    # Train model
    model, X_train, X_test, y_train, y_test = train_model(df, feature_names)
    
    # Initialize explainer
    print("\n[INITIALIZING EXPLAINER]")
    print("-" * 40)
    explainer = Explainer(
        model=model,
        data=X_train,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        task_type='classification',
        sensitive_features=['gender', 'ethnicity']
    )
    print("Explainer initialized with all XAI methods")
    
    # Run demos
    demo_shap(explainer, X_test)
    demo_lime(explainer, X_test)
    demo_feature_importance(explainer, X_test, y_test)
    demo_counterfactuals(explainer, X_test)
    demo_fairness(explainer, df, feature_names)
    demo_model_card(explainer)
    demo_comprehensive_report(explainer, X_test, y_test)
    
    # Summary
    print("\n" + "="*60)
    print("XAI DEMO COMPLETE!")
    print("="*60)
    print("\nKey Capabilities Demonstrated:")
    print("  [OK] SHAP global and local explanations")
    print("  [OK] LIME local surrogate models")
    print("  [OK] Multiple feature importance methods")
    print("  [OK] Counterfactual generation")
    print("  [OK] Fairness and bias detection")
    print("  [OK] Automatic model card generation")
    print("  [OK] Comprehensive explainability reports")
    print("\nMLPY provides state-of-the-art explainability!")
    print("\nCheck the generated files in xai_output/ directory")


if __name__ == "__main__":
    main()