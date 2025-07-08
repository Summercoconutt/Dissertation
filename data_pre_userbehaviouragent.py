import pandas as pd

def preprocess_user_behavior(df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Proposal_Index"] = df.groupby("Proposal Title").ngroup()

    sequences = []
    for voter, group in df.groupby("Voter"):
        group = group.sort_values("Proposal_Index")
        history = list(zip(group["Proposal Title"], group["Choice"]))
        if len(history) > window_size:
            for i in range(len(history) - window_size):
                window = history[i:i+window_size]
                label = history[i+window_size][1]
                sequences.append({
                    "voter": voter,
                    "history": window,
                    "next_choice": label
                })

    return pd.DataFrame(sequences)
