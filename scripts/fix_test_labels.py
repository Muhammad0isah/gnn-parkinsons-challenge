#!/usr/bin/env python3
"""
Fix test labels format to ensure scoring works
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def fix_test_labels():
    """Convert test labels to proper format"""
    
    # Load existing test labels
    test_labels_path = Path('data/test_labels.pkl')
    
    if not test_labels_path.exists():
        print("❌ Error: test_labels.pkl not found!")
        print("Run: python scripts/generate_graph_data.py first")
        return
    
    with open(test_labels_path, 'rb') as f:
        labels = pickle.load(f)
    
    print(f"Original labels type: {type(labels)}")
    print(f"Original labels structure: {labels.keys() if isinstance(labels, dict) else 'Not a dict'}")
    
    # Handle different formats
    if isinstance(labels, dict) and 'node_ids' in labels and 'labels' in labels:
        # Format: {'node_ids': array([...]), 'labels': array([...])}
        node_ids = labels['node_ids']
        label_values = labels['labels']
        
        print(f"\nNode IDs: {node_ids}")
        print(f"Labels: {label_values}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'node_id': node_ids,
            'label': label_values
        })
        
    elif isinstance(labels, dict):
        # Format: {node_id: label, ...}
        df = pd.DataFrame(list(labels.items()), columns=['node_id', 'label'])
        
    elif isinstance(labels, pd.DataFrame):
        if 'node_id' not in labels.columns or 'label' not in labels.columns:
            # Assume index is node_id
            df = pd.DataFrame({
                'node_id': labels.index,
                'label': labels.values.flatten()
            })
        else:
            df = labels
            
    elif isinstance(labels, pd.Series):
        df = pd.DataFrame({
            'node_id': labels.index,
            'label': labels.values
        })
    else:
        # Assume it's a list or array
        df = pd.DataFrame({
            'node_id': range(len(labels)),
            'label': labels
        })
    
    # Ensure proper types
    df['node_id'] = df['node_id'].astype(int)
    df['label'] = df['label'].astype(int)
    
    # Sort by node_id
    df = df.sort_values('node_id').reset_index(drop=True)
    
    # IMPORTANT: The node_ids are the actual graph node indices (e.g., 185, 77, etc.)
    # But for the submission format, we need sequential IDs 0-38
    # So we'll create a mapping
    print(f"\n⚠️  IMPORTANT: Original node IDs range from {df['node_id'].min()} to {df['node_id'].max()}")
    print(f"   But submission format expects node_id 0-38")
    print(f"   Creating mapping...")
    
    # Create sequential mapping 0-38
    df_sequential = df.copy()
    df_sequential['original_node_id'] = df_sequential['node_id']
    df_sequential['node_id'] = range(len(df_sequential))
    
    print(f"\nFixed labels (with sequential IDs):")
    print(df_sequential.head(10))
    print(f"\nShape: {df_sequential.shape}")
    print(f"Sequential Node IDs: {df_sequential['node_id'].min()} to {df_sequential['node_id'].max()}")
    print(f"Original Node IDs: {df_sequential['original_node_id'].min()} to {df_sequential['original_node_id'].max()}")
    print(f"Labels distribution: {df_sequential['label'].value_counts().to_dict()}")
    
    # Save the version with sequential IDs (for scoring)
    with open(test_labels_path, 'wb') as f:
        pickle.dump(df_sequential[['node_id', 'label']], f)
    
    print(f"\n✓ Test labels fixed and saved to: {test_labels_path}")
    
    # Save CSV versions for reference
    csv_path = Path('data/test_labels_reference.csv')
    df_sequential.to_csv(csv_path, index=False)
    print(f"✓ Reference CSV saved to: {csv_path}")
    
    # Also save the mapping for reference
    mapping_path = Path('data/node_id_mapping.csv')
    df_sequential[['node_id', 'original_node_id']].to_csv(mapping_path, index=False)
    print(f"✓ Node ID mapping saved to: {mapping_path}")
    
    print(f"\n📋 Summary:")
    print(f"   - Submission format uses node_id: 0-38")
    print(f"   - These map to original graph nodes: {list(df['node_id'].values)}")
    print(f"   - See node_id_mapping.csv for the full mapping")

if __name__ == '__main__':
    fix_test_labels()