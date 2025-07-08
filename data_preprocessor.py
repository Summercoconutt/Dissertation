import pandas as pd

# -------------------------------
# 1. User Behavior Agent
# -------------------------------
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

# -------------------------------
# 2. Zombie Detection Agent
# -------------------------------
def preprocess_zombie_detection(df: pd.DataFrame) -> pd.DataFrame:
    zombie_df = df.groupby("Voter").agg(
        vote_count=("Proposal Title", "count"),
        choice_diversity=("Choice", "nunique"),
        whale_votes=("Is Whale", "sum"),
        align_rate=("Aligned With Majority", "mean")
    ).reset_index()

    # 自定义 Zombie 标签规则
    zombie_df["zombie_label"] = (
        (zombie_df["vote_count"] <= 2) &
        (zombie_df["choice_diversity"] < 2)
    ).astype(int)

    return zombie_df

# -------------------------------
# 3. Mechanism Simulation Agent
# -------------------------------
def preprocess_mechanism_simulation(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Voter").agg(
        avg_vp=("Voting Power", "mean"),
        max_vp=("Voting Power", "max"),
        vote_count=("Proposal Title", "count")
    ).reset_index()


import pandas as pd
from data_preprocessor import (
    preprocess_user_behavior,
    preprocess_zombie_detection,
    preprocess_mechanism_simulation
)

# 读取数据
df = pd.read_csv("snapshot_votes_with_body.csv")

# 调用处理函数
user_behavior_data = preprocess_user_behavior(df)
zombie_data = preprocess_zombie_detection(df)
simulation_data = preprocess_mechanism_simulation(df)

# 保存结果
user_behavior_data.to_csv("user_behavior_data.csv", index=False)
zombie_data.to_csv("zombie_features.csv", index=False)
simulation_data.to_csv("simulation_user_profile.csv", index=False)
