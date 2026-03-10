#!/usr/bin/env python3
"""
Update leaderboard.json with new submission scores.
This script is called by GitHub Actions after scoring a submission.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def update_leaderboard(participant_name, score, model_type):
    """
    Update leaderboard.json with new submission.
    
    Args:
        participant_name: Name of the participant (filename without .csv)
        score: Macro F1-Score achieved
        model_type: Type of model used (e.g., "GCN", "GAT", "GraphSAGE")
    """
    leaderboard_path = Path("leaderboard.json")
    
    # Load existing leaderboard
    if leaderboard_path.exists():
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"entries": [], "last_updated": ""}
    
    # Create new entry
    new_entry = {
        "participant": participant_name,
        "score": round(float(score), 4),
        "model": model_type,
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    # Check if participant already exists
    existing_entry = None
    for i, entry in enumerate(leaderboard["entries"]):
        if entry["participant"] == participant_name:
            existing_entry = i
            break
    
    if existing_entry is not None:
        # Update existing entry if new score is better
        if new_entry["score"] > leaderboard["entries"][existing_entry]["score"]:
            leaderboard["entries"][existing_entry] = new_entry
            print(f"✅ Updated {participant_name}'s best score: {new_entry['score']}")
        else:
            print(f"ℹ️  {participant_name}'s new score ({new_entry['score']}) is not better than their best ({leaderboard['entries'][existing_entry]['score']})")
    else:
        # Add new entry
        leaderboard["entries"].append(new_entry)
        print(f"✅ Added new participant: {participant_name} with score {new_entry['score']}")
    
    # Sort by score (descending)
    leaderboard["entries"].sort(key=lambda x: x["score"], reverse=True)
    
    # Update timestamp
    leaderboard["last_updated"] = datetime.now().isoformat()
    
    # Save leaderboard
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"✅ Leaderboard updated successfully!")
    print(f"📊 Total participants: {len(leaderboard['entries'])}")
    
    # Print top 3
    print("\n🏆 Top 3:")
    for i, entry in enumerate(leaderboard["entries"][:3], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{medal} {entry['participant']}: {entry['score']} ({entry['model']})")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update leaderboard with new submission")
    parser.add_argument("participant", help="Participant name")
    parser.add_argument("score", type=float, help="Macro F1-Score")
    parser.add_argument("--model", default="GNN", help="Model type (default: GNN)")
    
    args = parser.parse_args()
    
    update_leaderboard(args.participant, args.score, args.model)


if __name__ == "__main__":
    main()