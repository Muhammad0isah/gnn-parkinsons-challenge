"""
convert_to_dgl_free.py
Run ONCE on your dev machine (where DGL works).
Converts DGL graph objects inside pkl files to plain dicts.
"""
import pickle, os, torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data\public")

def convert(in_filename, out_filename):
    path = os.path.join(DATA_DIR, in_filename)
    print(f"Loading {path} ...")
    with open(path, "rb") as f:
        data = pickle.load(f)

    g = data["graph"]
    src, dst = g.edges()

    plain = {
        "features":   data["features"],
        "labels":     data.get("labels"),
        "train_mask": data.get("train_mask"),
        "val_mask":   data.get("val_mask"),
        "node_ids":   data.get("node_ids"),
        "edge_index": torch.stack([src.long(), dst.long()], dim=0),
        "num_nodes":  g.num_nodes(),
        "num_edges":  g.num_edges(),
    }

    # copy any remaining keys (except the DGL graph)
    for k, v in data.items():
        if k != "graph" and k not in plain:
            plain[k] = v

    out_path = os.path.join(DATA_DIR, out_filename)
    with open(out_path, "wb") as f:
        pickle.dump(plain, f)
    print(f"  ✅ Saved → {out_path}")
    print(f"  Keys: {list(plain.keys())}")

convert("train_graph.pkl", "train_graph_free.pkl")
convert("test_graph.pkl",  "test_graph_free.pkl")
print("\nDone! Commit train_graph_free.pkl and test_graph_free.pkl to your repo.")