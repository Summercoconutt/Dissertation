import pandas as pd

def load_snapshot_data(file_path="merged_snapshot.csv"):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def get_user_profiles(df):
    user_stats = df.groupby('voter_address').agg({
        'vote_choice': 'count',
        'token_amount': 'sum'
    }).rename(columns={'vote_choice': 'vote_count'})
    return user_stats
