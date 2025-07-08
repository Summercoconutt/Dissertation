pip install requests pandas

import requests
import pandas as pd
import os
import time

# 目标 DAO 空间名（以 Uniswap DAO 为例）
space_name = "uniswapgovernance.eth"
batch_size = 100
skip = 0
all_proposals = []

print("🚀 开始抓取全部提案（含 Proposal Body）...")

# 分批抓取所有提案
while True:
    proposal_query = f"""
    {{
      proposals(first: {batch_size}, skip: {skip}, where: {{space_in: ["{space_name}"]}}, orderBy: "created", orderDirection: desc) {{
        id
        title
        body
        choices
        author
        state
      }}
    }}
    """
    response = requests.post("https://hub.snapshot.org/graphql", json={"query": proposal_query})
    proposals_batch = response.json().get("data", {}).get("proposals", [])

    print(f"📦 已抓取提案 {skip + 1} ~ {skip + len(proposals_batch)}")

    if not proposals_batch:
        break

    all_proposals.extend(proposals_batch)

    if len(proposals_batch) < batch_size:
        break

    skip += batch_size
    time.sleep(1)

print(f"✅ 提案总数：{len(all_proposals)}")

# 开始抓取投票数据
vote_records = []

for i, p in enumerate(all_proposals):
    proposal_id = p["id"]
    print(f"🔍 [{i+1}/{len(all_proposals)}] 处理提案：《{p['title']}》")

    vote_query = f"""
    {{
      votes(first: 1000, where: {{proposal: "{proposal_id}"}}) {{
        voter
        choice
        vp
      }}
    }}
    """
    vote_response = requests.post("https://hub.snapshot.org/graphql", json={"query": vote_query})
    vote_data = vote_response.json().get("data", {}).get("votes", [])

    if not vote_data:
        print("⚠️ 无投票数据，跳过")
        continue

    total_vp = sum(v["vp"] for v in vote_data)
    if total_vp == 0:
        continue

    # 统计最常出现的 choice
    choice_counter = {}
    for v in vote_data:
        choice_val = v["choice"]
        if isinstance(choice_val, int):
            choice_counter[choice_val] = choice_counter.get(choice_val, 0) + 1
        elif isinstance(choice_val, dict):
            for k, val in choice_val.items():
                if val > 0:
                    choice_counter[int(k)] = choice_counter.get(int(k), 0) + 1

    if not choice_counter:
        continue

    most_common_choice = max(choice_counter.items(), key=lambda x: x[1])[0]

    for v in vote_data:
        vp_ratio = v["vp"] / total_vp
        aligned = False

        if isinstance(v["choice"], int):
            aligned = v["choice"] == most_common_choice
            choice_used = v["choice"]
        elif isinstance(v["choice"], dict):
            aligned = str(most_common_choice) in v["choice"] and v["choice"][str(most_common_choice)] > 0
            choice_used = str(v["choice"])

        vote_records.append({
            "Proposal Title": p["title"],
            "Proposal Body": p["body"].replace('\n', ' ').replace('\r', ' ')[:1000],  # 截断避免过长
            "Voter": v["voter"],
            "Choice": choice_used,
            "Voting Power": v["vp"],
            "VP Ratio (%)": round(vp_ratio * 100, 2),
            "Is Whale": vp_ratio > 0.05,
            "Aligned With Majority": aligned
        })

# 保存为 CSV
df = pd.DataFrame(vote_records)
output_filename = "snapshot_votes_with_body.csv"
file_number = 1
while os.path.exists(output_filename):
    output_filename = f"snapshot_votes_with_body_{file_number}.csv"
    file_number += 1

df.to_csv(output_filename, index=False)
print(f"📁 数据已保存为：{output_filename}，共 {len(df)} 条投票记录")
