def preprocess_mechanism_simulation(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Voter").agg(
        avg_vp=("Voting Power", "mean"),
        max_vp=("Voting Power", "max"),
        vote_count=("Proposal Title", "count")
    ).reset_index()
