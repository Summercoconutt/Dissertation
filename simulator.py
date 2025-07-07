from data_loader import load_snapshot_data
from agent_base import UserBehaviorAgent, UserVoteHistory

def main():
    # ✅ Step 1: 加载 Snapshot 投票数据
    print("📥 正在加载治理数据...")
    df = load_snapshot_data("merged_snapshot.csv")
    print(f"✅ 成功加载 {len(df)} 条记录\n")

    # ✅ Step 2: 初始化 Agent 并构建训练数据
    print("🧠 初始化 UserBehaviorAgent...")
    agent = UserBehaviorAgent(window_size=3)
    agent.prepare_training_data(df)
    print("✅ 数据准备完毕，共提取训练样本：", len(agent.X))

    # ✅ Step 3: 训练模型
    agent.train()

    # ✅ Step 4: 测试预测一个用户样本
    print("\n🧪 测试预测结果示例：")
    example = UserVoteHistory(
        voter_address="0xABC123",
        proposal_ids=["p1", "p2", "p3"],
        vote_labels=[0, 1, 0]  # 假设最近为 [支持, 反对, 支持]
    )
    prediction = agent.predict(example)
    print("📊 预测结果：", prediction)

if __name__ == "__main__":
    main()



from evaluation import calculate_gini, calculate_hhi, calculate_participation_rate, calculate_utility

# 假设某次模拟结果如下：
voting_power = [100, 80, 50, 5, 1]  # 模拟投票权重
num_voters = 4
total_users = 10
proposal_results = [True, False, True]  # 三个提案通过情况

gini = calculate_gini(voting_power)
hhi = calculate_hhi(voting_power)
participation = calculate_participation_rate(num_voters, total_users)
pass_rate = calculate_pass_rate(proposal_results)
utility = calculate_utility(gini, participation, alpha=1.5, beta=1.0)

print("📊 Gini:", round(gini, 4))
print("📊 HHI:", round(hhi, 4))
print("📊 Participation:", round(participation, 4))
print("📊 Proposal Pass Rate:", round(pass_rate, 4))
print("💡 Utility Score:", round(utility, 4))





from mechanism_simulation import run_simulation

# 创建 voter_pool（演示用，真实可以从 snapshot 中构造）
df['reputation'] = df.groupby('voter_address')['vote_choice'].transform('count')
df['history'] = df.groupby('voter_address')['vote_label'].transform(lambda x: x.tolist())
voter_pool = df.groupby('voter_address').agg({
    'token_amount': 'mean',
    'reputation': 'mean',
    'history': 'first'
}).reset_index()

print("\n🏁 开始模拟不同投票机制...")

results = []
for mech in ["token", "quadratic", "reputation", "hybrid"]:
    summary = run_simulation(agent, voter_pool, mechanism=mech, num_rounds=100)
    results.append(summary)

final_results = pd.concat(results, ignore_index=True)
print("\n📊 各机制平均指标对比：")
print(final_results[["mechanism", "gini", "hhi", "participation", "passed", "utility"]])
