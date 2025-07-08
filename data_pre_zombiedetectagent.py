def preprocess_zombie_detection(df: pd.DataFrame) -> pd.DataFrame:
    zombie_df = df.groupby("Voter").agg(
        vote_count=("Proposal Title", "count"),
        choice_diversity=("Choice", "nunique"),
        whale_votes=("Is Whale", "sum"),
        align_rate=("Aligned With Majority", "mean")
    ).reset_index()

    # 自定义规则：投票少、选择单一 → Zombie
    zombie_df["zombie_label"] = (
        (zombie_df["vote_count"] <= 2) &
        (zombie_df["choice_diversity"] < 2)
    ).astype(int)

    return zombie_df
