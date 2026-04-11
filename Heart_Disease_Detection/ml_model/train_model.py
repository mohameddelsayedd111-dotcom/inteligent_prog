"""
ml_model/train_model.py
Train and evaluate a Decision Tree Classifier for heart disease prediction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_and_prepare(path: str) -> tuple:
    """Load cleaned data and split into X, y."""
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


def train(X_train, y_train, params: dict = None) -> DecisionTreeClassifier:
    """Train DecisionTree with optional GridSearch."""
    if params:
        clf = DecisionTreeClassifier(random_state=42, **params)
    else:
        # Hyperparameter tuning
        grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        base = DecisionTreeClassifier(random_state=42)
        gs = GridSearchCV(base, grid, cv=5, scoring='f1', n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"[✔] Best params: {gs.best_params_}")
        clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    return clf


def evaluate(clf, X_test, y_test) -> dict:
    """Evaluate and print metrics."""
    y_pred = clf.predict(X_test)
    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred), 4),
        'f1_score':  round(f1_score(y_test, y_pred), 4),
    }
    print("\n[📊] Decision Tree Evaluation:")
    print(f"  Accuracy  : {metrics['accuracy']:.2%}")
    print(f"  Precision : {metrics['precision']:.2%}")
    print(f"  Recall    : {metrics['recall']:.2%}")
    print(f"  F1-Score  : {metrics['f1_score']:.2%}")
    print("\n[📋] Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['No Disease', 'Disease']))
    return metrics


def save_model(clf, path: str = '../ml_model/heart_disease_model.pkl'):
    """Export model with joblib."""
    joblib.dump(clf, path)
    print(f"[✔] Model saved to: {path}")


def plot_feature_importance(clf, feature_names: list, save_path: str = None):
    """Bar chart of feature importances."""
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[idx], color='steelblue', alpha=0.85)
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.title('Feature Importances — Decision Tree', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✔] Feature importance plot saved.")


def plot_confusion(clf, X_test, y_test, save_path: str = None):
    """Confusion matrix heatmap."""
    import seaborn as sns
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix — Decision Tree')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    from data_processing import preprocess_pipeline
    preprocess_pipeline(
        raw_path=os.path.join(BASE, 'data', 'raw_data.csv'),
        cleaned_path=os.path.join(BASE, 'data', 'cleaned_data.csv')
    )

    X, y = load_and_prepare(os.path.join(BASE, 'data', 'cleaned_data.csv'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"[✔] Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    clf = train(X_train, y_train)
    metrics = evaluate(clf, X_test, y_test)

    plot_feature_importance(clf, list(X.columns),
                            save_path=os.path.join(BASE, 'reports', 'feature_importance.png'))
    plot_confusion(clf, X_test, y_test,
                   save_path=os.path.join(BASE, 'reports', 'confusion_matrix.png'))

    model_path = os.path.join(BASE, 'ml_model', 'heart_disease_model.pkl')
    save_model(clf, model_path)

    # Save metrics for comparison report
    pd.DataFrame([metrics]).to_csv(
        os.path.join(BASE, 'reports', 'dt_metrics.csv'), index=False)
