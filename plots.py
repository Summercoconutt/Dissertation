import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ✅ 生成柱状图
def plot_bar_comparison(df: pd.DataFrame, metrics: List[str], title: str = "Governance Metrics by Mechanism"):
    mechanisms = df["mechanism"].tolist()
    bar_width = 0.2
    index = np.arange(len(mechanisms))

    plt.figure(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width,
                df[metric],
                width=bar_width,
                label=metric.capitalize())

    plt.xlabel("Voting Mechanism")
    plt.ylabel("Metric Score")
    plt.title(title)
    plt.xticks(index + bar_width * (len(metrics)-1) / 2, mechanisms)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("bar_plot_metrics.png")
    plt.show()


# ✅ 生成雷达图（蜘蛛图）
def plot_radar_chart(df: pd.DataFrame, metrics: List[str], title: str = "Radar Chart - Mechanism Comparison"):
    labels = metrics
    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in df.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['mechanism'])
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title(title, size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.savefig("radar_plot_metrics.png")
    plt.show()
