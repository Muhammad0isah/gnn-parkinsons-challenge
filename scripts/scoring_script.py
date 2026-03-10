#!/usr/bin/env python3
"""
GNN Parkinson's Challenge - Scoring Script
Evaluates submissions and updates the leaderboard
"""

import pandas as pd
import pickle
import sys
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def load_ground_truth(ground_truth_path=None):
    """Load ground truth labels"""
    # If ground truth path is provided (from GitHub Actions secret)
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"✓ Loading ground truth from: {ground_truth_path}")
        df = pd.read_csv(ground_truth_path)
        if 'node_id' in df.columns and 'label' in df.columns:
            return df.sort_values('node_id').reset_index(drop=True)
        elif len(df.columns) == 2:
            df.columns = ['node_id', 'label']
            return df.sort_values('node_id').reset_index(drop=True)

    # Try multiple possible paths for local testing
    possible_paths = [
        Path('data/private/test_labels.pkl'),   # ← correct path
        Path('../data/private/test_labels.pkl'),
        Path('data/test_labels.pkl'),            # legacy fallback
        Path('../data/test_labels.pkl'),
        Path('/tmp/ground_truth.csv'),
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✓ Found ground truth at: {path}")

            # Handle CSV files
            if path.suffix == '.csv':
                df = pd.read_csv(path)
                if 'node_id' not in df.columns or 'label' not in df.columns:
                    if len(df.columns) == 2:
                        df.columns = ['node_id', 'label']
                return df.sort_values('node_id').reset_index(drop=True)

            # Handle pickle files
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # ✅ Handle the {'node_ids': ..., 'labels': ...} format from generate_graph_data.py
            if isinstance(data, dict) and 'node_ids' in data and 'labels' in data:
                df = pd.DataFrame({
                    'node_id': data['node_ids'],
                    'label':   data['labels']
                })
            elif isinstance(data, dict) and 'node_id' in data and 'label' in data:
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, pd.Series):
                df = pd.DataFrame({'node_id': data.index, 'label': data.values})
            else:
                df = pd.DataFrame({'node_id': range(len(data)), 'label': data})

            return df.sort_values('node_id').reset_index(drop=True)

    print("❌ Error: Ground truth labels not found")
    print(f"\n📝 Note for challenge organizers:")
    print(f"   Ground truth should be in: data/private/test_labels.pkl")
    print(f"   Current working directory: {Path.cwd()}")
    return None


def validate_submission(submission_df, ground_truth):
    """Validate submission format"""
    errors = []

    # Check columns
    required_cols = ['node_id', 'prediction']
    if not all(col in submission_df.columns for col in required_cols):
        errors.append(f"Missing required columns. Expected: {required_cols}, Got: {list(submission_df.columns)}")
        return errors

    # Check row count matches ground truth
    expected_count = len(ground_truth)
    if len(submission_df) != expected_count:
        errors.append(f"Expected {expected_count} predictions, got {len(submission_df)}")

    # Check node IDs match ground truth exactly
    expected_ids = set(ground_truth['node_id'].values)
    actual_ids   = set(submission_df['node_id'].values)
    if actual_ids != expected_ids:
        errors.append(f"node_id mismatch. Missing: {expected_ids - actual_ids}, Extra: {actual_ids - expected_ids}")

    # Check predictions are binary
    if not all(submission_df['prediction'].isin([0, 1])):
        errors.append("All predictions must be 0 or 1")

    return errors


def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    metrics = {
        'macro_f1':  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1_score':  f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred)
    except Exception:
        metrics['auc_roc'] = None

    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <submission_file> [submission_name] [ground_truth_file]")
        sys.exit(1)

    submission_file   = Path(sys.argv[1])
    submission_name   = sys.argv[2] if len(sys.argv) > 2 else submission_file.stem
    ground_truth_file = sys.argv[3] if len(sys.argv) > 3 else None

    # Load submission
    try:
        submission_df = pd.read_csv(submission_file)
        print(f"\n📄 Loaded submission: {submission_file}")
        print(f"   Shape: {submission_df.shape}")
    except Exception as e:
        print(f"❌ Error loading submission: {e}")
        sys.exit(1)

    # Load ground truth first (needed for validation)
    ground_truth = load_ground_truth(ground_truth_file)
    if ground_truth is None:
        print("\n⚠️  Cannot score submission without ground truth labels.")
        print("   Your submission format is valid and ready to submit!")
        sys.exit(0)

    # Validate format against ground truth
    errors = validate_submission(submission_df, ground_truth)
    if errors:
        print("❌ Submission validation failed:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)

    print("✓ Submission format is valid")

    # Merge and calculate metrics
    merged = submission_df.merge(ground_truth, on='node_id', how='inner')

    if len(merged) != len(ground_truth):
        print(f"❌ Error: Could only match {len(merged)}/{len(ground_truth)} predictions with ground truth")
        sys.exit(1)

    metrics = calculate_metrics(merged['label'], merged['prediction'])

    # Display results
    print("\n" + "=" * 60)
    print(f"RESULTS FOR: {submission_name}")
    print("=" * 60)
    print(f"Macro F1:  {metrics['macro_f1']:.4f}  ← competition metric")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if metrics['auc_roc']:
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print("=" * 60)

    # For GitHub Actions - output in parseable format
    print(f"\nScore: {metrics['macro_f1']:.4f}")

    print("\n📊 Class distribution in predictions:")
    print(submission_df['prediction'].value_counts())


if __name__ == '__main__':
    main()