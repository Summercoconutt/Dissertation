import pandas as pd
import numpy as np
from typing import List, Dict

# ✅ 计算 Gini 系数（衡量投票权的不平等程度）
def calculate_gini(voting_power: List[float]) -> float:
    array = np.array(voting_power)
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-10  # 避免除零
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

# ✅ 计算 Herfindahl–Hirschman Index (HHI)：衡量集中程度，值越高代表越集中
def calculate_hhi(voting_power: List[float]) -> float:
    power_array = np.array(voting_power)
    squared_shares = (power_array / power_array.sum()) ** 2
    return np.sum(squared_shares)

# ✅ 计算 Proposal 参与率：有多少人投票 / 总用户
def calculate_participation_rate(num_voters: int, total_users: int) -> float:
    if total_users == 0:
        return 0.0
    return num_voters / total_users

# ✅ 计算 Proposal 通过率：有多少提案最终通过
def calculate_pass_rate(proposal_results: List[bool]) -> float:
    return sum(proposal_results) / len(proposal_results) if proposal_results else 0.0

# ✅ 综合效用函数：平衡公平性 vs 效率
def calculate_utility(gini: float, participation: float, alpha=1.0, beta=1.0) -> float:
    """
    U = alpha × participation_rate - beta × gini
    你可以自定义权重
    """
    return alpha * participation - beta * gini
