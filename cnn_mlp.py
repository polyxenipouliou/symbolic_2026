import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report


# ==========================================
# 1. 核心模型架构：1D-CNN (动机提取) + Global Max Pooling + MLP
# ==========================================
class SimpleCNN_MLP(nn.Module):
    def __init__(self, input_dim=20, num_classes=3):
        super().__init__()

        # 卷积层：提取局部序列特征 (相当于观察连续的几个小节)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),  # 50% 丢弃率防止过拟合
            nn.Linear(64, num_classes)
        )

    def forward(self, x, mask):
        # x 形状: [Batch, Seq, 20]
        # PyTorch 的 Conv1d 需要的形状是: [Batch, Channels, Seq]
        x = x.permute(0, 2, 1)

        # 经过卷积提取特征 -> 形状: [Batch, 128, Seq]
        out = self.features(x)

        # === 带掩码的全局最大池化 (Masked Global Max Pooling) ===
        # 忽略掉那些为了对齐而填充 (Padding) 的空白小节
        mask_expanded = mask.unsqueeze(1)  # [Batch, 1, Seq]
        out = out.masked_fill(mask_expanded == 0, -1e9)  # 将无效小节设为极小值

        # 在序列维度 (dim=2) 上取最大值，把一首歌压缩成一个 128 维的向量
        pooled, _ = out.max(dim=2)  # 形状: [Batch, 128]

        # 送入 MLP 分类
        return self.mlp(pooled)


# ==========================================
# 2. 数据预处理流水线 (带标准化 StandardScaler)
# ==========================================
def pad_or_truncate(lst, target_length):
    """强制对齐列表长度，消除 inhomogeneous 报错"""
    if len(lst) >= target_length:
        return lst[:target_length]
    else:
        return lst + [0.0] * (target_length - len(lst))


def prepare_sequential_data(csv_path):
    print("[*] 正在解析并对齐复杂的嵌套序列特征...")
    df = pd.read_csv(csv_path)

    # 剔除 Clara Schumann
    df = df[df['composer'] != 'Clara Schumann'].copy()

    # 标签编码
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['composer'])

    # 按 filename 和 bar 严格排序
    df['bar_idx'] = df['bar'].apply(lambda x: int(x.split(' ')[1]))
    df = df.sort_values(['filename', 'bar_idx'])

    all_sequences = []
    all_labels = []

    for name, group in df.groupby('filename'):
        seq_data = []
        for _, row in group.iterrows():
            tt = pad_or_truncate(ast.literal_eval(row['tt']), 4)
            hc = pad_or_truncate(ast.literal_eval(row['hc']), 4)
            pt = pad_or_truncate(ast.literal_eval(row['pt']), 8)
            mc = pad_or_truncate(ast.literal_eval(row['mc']), 4)

            bar_feat = tt + hc + pt + mc  # 严格 20 维
            seq_data.append(bar_feat)

        all_sequences.append(np.array(seq_data))
        all_labels.append(group['label'].iloc[0])

    # 寻找最大长度
    max_len = max(len(s) for s in all_sequences)
    print(f"[*] 样本总量: {len(all_sequences)}, 最大乐曲长度: {max_len} 小节")

    # 构造填充张量和掩码
    X_padded = np.zeros((len(all_sequences), max_len, 20))
    masks = np.zeros((len(all_sequences), max_len))

    for i, s in enumerate(all_sequences):
        l = len(s)
        X_padded[i, :l, :] = s
        masks[i, :l] = 1

        # === 关键修复：特征归一化 (Standardization) ===
    # 把 [Batch, Seq, 20] 压平为 [Batch*Seq, 20] 交给 Scaler
    flat_X = X_padded.reshape(-1, 20)
    scaler = StandardScaler()
    flat_X_scaled = scaler.fit_transform(flat_X)

    # 变回 3D 形状
    X_padded_scaled = flat_X_scaled.reshape(len(all_sequences), max_len, 20)

    # 将 Padding 部分强行归零（因为 Scaler 会把 0 变成非 0，必须重置）
    X_padded_scaled = X_padded_scaled * masks[:, :, np.newaxis]

    return X_padded_scaled, masks, np.array(all_labels), le


# ==========================================
# 3. 训练执行核心
# ==========================================
if __name__ == "__main__":
    # 请确保路径正确
    csv_file_path = 'features/features_sequential.csv'

    X, M, Y, le = prepare_sequential_data(csv_file_path)

    # 划分数据集 (80/20)
    X_train, X_test, M_train, M_test, y_train, y_test = train_test_split(
        X, M, Y, test_size=0.2, random_state=42, stratify=Y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 当前使用设备: {device}")

    X_train = torch.FloatTensor(X_train).to(device)
    M_train = torch.FloatTensor(M_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)

    X_test = torch.FloatTensor(X_test).to(device)
    M_test = torch.FloatTensor(M_test).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    # 初始化简单的 CNN+MLP 模型
    model = SimpleCNN_MLP(input_dim=20, num_classes=3).to(device)

    # 使用 AdamW 优化器，学习率设为 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"[*] 训练启动... 类别映射: {list(le.classes_)}")

    epochs = 300
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 传入输入和掩码
        outputs = model(X_train, M_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        # 每 10 轮打印一次测试集准确率
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(X_test, M_test).argmax(dim=1)
                acc = (preds == y_test).float().mean()
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

    # ==========================================
    # 4. 最终评估
    # ==========================================
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test, M_test).argmax(dim=1).cpu()

    print("\n[+] Simple CNN + MLP 最终分类报告:")
    print(classification_report(y_test.cpu(), final_preds, target_names=le.classes_))