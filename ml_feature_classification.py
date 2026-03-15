import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# ==========================================
# 1. 数据加载与清洗
# ==========================================
df_stat = pd.read_csv('features/features_statistical.csv')

# 直接剔除 Clara Schumann
df_filtered = df_stat[df_stat['composer'] != 'Clara Schumann'].copy()
print(f"[*] 清洗后样本量: {len(df_filtered)}")

# 提取特征 (X) 和 标签 (y)
# 使用 tt, hc, pt, mc 的 mean, std, entropy 共 12 维特征
X_raw = df_filtered.drop(columns=['filename', 'composer']).values
y_raw = df_filtered['composer'].values

# 标签编码: 将名字转为 0, 1, 2
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
print(f"[*] 类别映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

# ==========================================
# 2. 数据预处理 (MLP 必须进行标准化)
# ==========================================
# MLP 对特征尺度非常敏感，必须保证均值为 0，方差为 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 划分训练集和测试集 (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 转为 PyTorch 张量并搬运到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

# 构建 DataLoader
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)


# ==========================================
# 3. 定义 MLP 模型架构
# ==========================================
class MusicFeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MusicFeatureMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_classes)  # 输出 3 个类别的 Logits
        )

    def forward(self, x):
        return self.net(x)


model = MusicFeatureMLP(input_dim=12, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# ==========================================
# 4. 训练循环
# ==========================================
print("[*] 开始训练...")
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {total_loss / len(train_loader):.4f}")

# ==========================================
# 5. 评估结果
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_t)
    predictions = torch.argmax(test_outputs, dim=1)

    print("\n[+] 最终评估报告 (3 分类):")
    print(classification_report(y_test_t.cpu(), predictions.cpu(), target_names=le.classes_))