import random
from typing import List, Dict
import pandas as pd
import numpy as np

from data_loader import load_snapshot_data
from agent_base import UserBehaviorAgent, UserVoteHistory
from evaluation import (
    calculate_gini, calculate_hhi,
    calculate_participation_rate,
    calculate_pass_rate,
    calculate_utility
)

# ✅ 投票机制定义模块
def token_voting(tokens: List[float]) -> List[float]:
    return tokens

def quadratic_voting(tokens: List[float]) -> List[float]:
    return [np.sqrt(t) for t in tokens]

def reputation_voting(reputations: List[float]) -> List[float]:
    return [np.log1p(r) for r in reputations]

def hybrid_voting(tokens: List[float], reputations: List[float], alpha=0.5) -> List[float]:
    qv = [np.sqrt(t) for t in tokens]
    rep = [np.log1p(r) for r in reputations]
    return [alpha*q + (1-alpha)*r for q, r in zip(qv, rep)]

# ✅ 单轮仿真函数（模拟一次提案）
def simulate_proposal(agent: UserBehaviorAgent,
                      mechanism: str,
                      voter_pool: pd.DataFrame,
                      participation_rate: float = 0.3,
                      whale_threshold: float = 1000.0) -> Dict:

    # 从 voter_pool 中随机抽取部分人参与投票
    total_users = len(voter_pool)
    num_voters = int(total_users * participation_rate)
    voters = voter_pool.sample(n=num_voters)

    vote_results = []
    voting_power = []

    for _, row in voters.iterrows():
        # 模拟用户行为
        vote_labels = row['history'][-agent.window_size:] if isinstance(row['history'], list) else [0, 1, 0]
        if len(vote_labels) < agent.window_size:
            vote_labels = ([0] * (agent.window_size - len(vote_labels))) + vote_labels

        user_input = UserVoteHistory(
            voter_address=row['voter_address'],
            proposal_ids=["dummy"]*agent.window_size,
            vote_labels=vote_labels
        )

        prediction = agent.predict(user_input)
        vote_results.append((prediction.predicted_vote, row['token_amount'], row['reputation']))

    # 根据投票机制计算 voting power
    tokens = [v[1] for v in vote_results]
    reputations = [v[2] for v in vote_results]

    if mechanism == "token":
        power = token_voting(tokens)
    elif mechanism == "quadratic":
        power = quadratic_voting(tokens)
    elif mechanism == "reputation":
        power = reputation_voting(reputations)
    elif mechanism == "hybrid":
        power = hybrid_voting(tokens, reputations)
    else:
        raise ValueError("Invalid mechanism name.")

    voting_power = power

    # 模拟是否通过（这里简单判断：for 类得分 > against）
    for_votes = sum([p for (v, _, _), p in zip(vote_results, voting_power) if v == 0])
    against_votes = sum([p for (v, _, _), p in zip(vote_results, voting_power) if v == 1])
    proposal_passed = for_votes > against_votes

    # 指标计算
    gini = calculate_gini(voting_power)
    hhi = calculate_hhi(voting_power)
    participation = num_voters / total_users
    utility = calculate_utility(gini, participation)
    
    return {
        "gini": gini,
        "hhi": hhi,
        "participation": participation,
        "passed": proposal_passed,
        "utility": utility
    }

# ✅ 多轮仿真主函数
def run_simulation(agent: UserBehaviorAgent, voter_pool: pd.DataFrame,
                   mechanism: str, num_rounds: int = 100) -> pd.DataFrame:

    results = []
    for i in range(num_rounds):
        result = simulate_proposal(agent, mechanism, voter_pool)
        results.append(result)

    df_results = pd.DataFrame(results)
    df_summary = df_results.mean().to_dict()
    df_summary["mechanism"] = mechanism
    return pd.DataFrame([df_summary])
