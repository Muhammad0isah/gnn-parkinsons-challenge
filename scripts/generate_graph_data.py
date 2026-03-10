"""
Generate Graph Data from UCI Parkinson's Dataset
"""

import pandas as pd
import numpy as np
import torch
import dgl
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import os


def download_parkinsons_data():
    print("Downloading UCI Parkinson's Dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    
    try:
        df = pd.read_csv(url)
        print(f"Successfully downloaded dataset with {len(df)} samples")
        return df
    except Exception as e:
        print(f"Error: {e}")
        print(f"Please download manually from: {url}")
        return None


def create_graph_from_features(features, labels, k=5):
    print(f"Creating KNN graph with k={k}...")
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    src, dst, edge_weights = [], [], []
    
    # Create bidirectional edges
    for i in range(len(features)):
        for j in range(1, k+1):
            neighbor_idx = indices[i, j]
            weight = 1.0 / (distances[i, j] + 1e-6)
            
            # Add edge i -> neighbor
            src.append(i)
            dst.append(neighbor_idx)
            edge_weights.append(weight)
            
            # Add reverse edge neighbor -> i
            src.append(neighbor_idx)
            dst.append(i)
            edge_weights.append(weight)
    
    g = dgl.graph((src, dst))
    g.edata['weight'] = torch.FloatTensor(edge_weights)
    
    print(f"Graph created: {g.num_nodes()} nodes, {g.num_edges()} edges")
    return g


def add_subject_connections(g, df):
    print("Adding subject-based connections...")
    subjects = df['name'].str.split('_').str[0]
    unique_subjects = subjects.unique()
    
    src, dst = [], []
    for subject in unique_subjects:
        subject_indices = np.where(subjects == subject)[0]
        for i in subject_indices:
            for j in subject_indices:
                if i != j:
                    src.append(i)
                    dst.append(j)
    
    if len(src) > 0:
        g.add_edges(src, dst)
        print(f"Added {len(src)} subject-based edges")
    
    return g


def prepare_data_splits(df, test_size=0.2, val_size=0.15):
    print("\nPreparing data splits...")
    
    feature_cols = [col for col in df.columns if col not in ['name', 'status']]
    X = df[feature_cols].values
    y = df['status'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    indices = np.arange(len(X))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=y
    )
    
    y_train_val = y[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_train_val
    )
    
    print(f"Train: {len(train_idx)} samples")
    print(f"Validation: {len(val_idx)} samples")
    print(f"Test: {len(test_idx)} samples")
    
    return X, y, train_idx, val_idx, test_idx, scaler


def save_graph_data(g, features, labels, train_idx, val_idx, test_idx, output_dir='../data/public'):
    os.makedirs(output_dir, exist_ok=True)
    
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # ✅ Hide test labels — replace with -1
    safe_labels = labels.clone()
    safe_labels[test_mask] = -1

    train_data = {
        'graph': g,
        'features': features,
        'labels': safe_labels,   # ← only train+val labels visible, test = -1
        'train_mask': train_mask,
        'val_mask': val_mask
    }
    
    with open(os.path.join(output_dir, 'train_graph.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    print(f"\nSaved training data to {output_dir}/train_graph.pkl")

    # Verify hiding worked
    hidden = (safe_labels == -1).sum().item()
    visible = (safe_labels != -1).sum().item()
    print(f"  ✅ Hidden test labels: {hidden}")
    print(f"  ✅ Visible train+val labels: {visible}")
    
    test_data = {
        'graph': g,
        'features': features,
        'node_ids': test_idx
    }
    
    with open(os.path.join(output_dir, 'test_graph.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Saved test data to {output_dir}/test_graph.pkl")
    
    # Save ground truth separately (for scoring only — never share this!)
    ground_truth = {
        'node_ids': test_idx,
        'labels': labels[test_idx].numpy()
    }
    
    # Save ground truth one level up so it's NOT in the public folder
    private_dir = os.path.join(os.path.dirname(output_dir), 'private')
    os.makedirs(private_dir, exist_ok=True)
    with open(os.path.join(private_dir, 'test_labels.pkl'), 'wb') as f:
        pickle.dump(ground_truth, f)
    print(f"Saved test labels to {private_dir}/test_labels.pkl (hidden - do not share!)")


def create_feature_description(output_dir='../data/public'):
    desc = """Parkinson's Disease Voice Measurement Features
==============================================
Source: UCI Machine Learning Repository

Features (22 total):
1. MDVP:Fo(Hz) - Average vocal fundamental frequency
2. MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
3. MDVP:Flo(Hz) - Minimum vocal fundamental frequency
4-8. Jitter measures (variation in fundamental frequency)
9-14. Shimmer measures (variation in amplitude)
15-16. Noise ratios (NHR, HNR)
17-22. Nonlinear measures (RPDE, DFA, spread1, spread2, D2, PPE)

Target: status (0=Healthy, 1=Parkinson's)
"""
    
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write(desc)
    print(f"Saved feature descriptions to {output_dir}/feature_names.txt")


def main():
    print("=" * 70)
    print("GNN Parkinson's Challenge - Data Generation")
    print("=" * 70)
    
    df = download_parkinsons_data()
    if df is None:
        return
    
    X, y, train_idx, val_idx, test_idx, scaler = prepare_data_splits(df)
    g = create_graph_from_features(X, y, k=5)
    g = add_subject_connections(g, df)
    save_graph_data(g, X, y, train_idx, val_idx, test_idx, output_dir='../data/public')
    create_feature_description(output_dir='../data/public')
    
    sample_submission = pd.DataFrame({
        'node_id': test_idx,
        'prediction': [0] * len(test_idx)
    })
    
    os.makedirs('../submissions', exist_ok=True)
    sample_submission.to_csv('../submissions/sample_submission.csv', index=False)
    print(f"\nSaved sample submission to ../submissions/sample_submission.csv")
    
    print("\n" + "=" * 70)
    print("Data generation complete!")
    print("\nFolder structure:")
    print("  data/public/train_graph.pkl  ← share with participants")
    print("  data/public/test_graph.pkl   ← share with participants")
    print("  data/private/test_labels.pkl ← NEVER share (ground truth)")
    print("=" * 70)


if __name__ == '__main__':
    main()