# competition/render_leaderboard.py
import pandas as pd
from datetime import datetime

def render_leaderboard():
    # Read CSV
    df = pd.read_csv('docs/leaderboard.csv')
    
    # Sort by score descending, then by date ascending (earlier submission wins tiebreak)
    df = df.sort_values(['score', 'date'], ascending=[False, True]).reset_index(drop=True)
    
    # Ties share the same rank number (Kaggle-style)
    df['rank'] = df['score'].rank(method='min', ascending=False).astype(int)
    
    # Generate markdown
    with open('leaderboard/leaderboard.md', 'w') as f:
        f.write('# 🏆 Leaderboard\n\n')
        f.write(f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('| Rank | Team | Score | Model | Date |\n')
        f.write('|------|------|-------|-------|------|\n')
        
        for _, row in df.iterrows():
            rank = row['rank']
            medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else ''
            f.write(f"| {medal} {rank} | {row['team']} | {row['score']:.4f} | {row['model']} | {row['date']} |\n")

if __name__ == '__main__':
    render_leaderboard()
