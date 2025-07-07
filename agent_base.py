from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ✅ Step 1: 定义 Agent 输入格式（用作推理时的输入）
class UserVoteHistory(BaseModel):
    voter_address: str                      # 用户地址（可选）
    proposal_ids: List[str]                # 用户最近参与的提案ID
    vote_labels: List[int]                 # 用户最近n次的投票行为编码（0=for, 1=against, 2=abstain, 3=other）


# ✅ Step 2: 定义 Agent 输出格式
class VotePrediction(BaseModel):
    predicted_vote: int                   # 模型预测的下一次投票行为（0~3）
    confidence: float                     # 对该预测的置信度（概率）


# ✅ Step 3: 构建 Agent 类（包含数据准备、训练、预测方法）
class UserBehaviorAgent(BaseModel):
    window_size: int = 3                           # 滑动窗口大小
    model: RandomForestClassifier = RandomForestClassifier()
    label_map: dict = {'for': 0, 'against': 1, 'abstain': 2, 'other': 3}  # 投票标签映射

    def prepare_training_data(self, df: pd.DataFrame):
        """
        加载原始DataFrame，提取用户的历史行为序列作为训练样本。
        每个样本：窗口行为序列 + 下一步预测
        """
        df = df[['voter_address', 'vote_choice', 'timestamp']].dropna()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['vote_label'] = df['vote_choice'].map(self.label_map)

        samples = []
        for user, group in df.groupby('voter_address'):
            group = group.sort_values('timestamp')
            labels = list(group['vote_label'])

            if len(labels) > self.window_size:
                for i in range(self.window_size, len(labels)):
                    window = labels[i - self.window_size:i]
                    target = labels[i]
                    samples.append(window + [target])

        self.data = pd.DataFrame(samples)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

    def train(self):
        """
        用准备好的 X, y 数据训练分类模型，并打印评估结果
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("✅ 模型训练完成，分类结果如下：")
        print(classification_report(y_test, y_pred))

    def predict(self, history: UserVoteHistory) -> VotePrediction:
        """
        接收一个用户的行为历史，返回对下一步行为的预测结果
        """
        if len(history.vote_labels) != self.window_size:
            raise ValueError(f"vote_labels must be of length {self.window_size}")

        proba = self.model.predict_proba([history.vote_labels])[0]
        prediction = int(np.argmax(proba))
        confidence = float(np.max(proba))

        return VotePrediction(predicted_vote=prediction, confidence=confidence)
