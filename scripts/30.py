import os
import pandas as pd
import numpy as np
import math
from collections import Counter
from music21 import converter, note, chord
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')


# ==========================================
# 1. 纯手工特征提取函数 (30 个基础维度)
# ==========================================
def extract_30_custom_features(midi_path):
    """
    不依赖特征库，纯手工从乐谱中解析音符，并计算 30 个物理/统计特征。
    """
    try:
        # 解析 MIDI
        score = converter.parse(midi_path)
        notes_data = []

        # 展平乐谱并遍历所有音符和和弦
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                notes_data.append({
                    'pitch': element.pitch.midi,
                    'vel': element.volume.velocity if element.volume.velocity else 64,
                    'dur': float(element.quarterLength),
                    'offset': float(element.offset)
                })
            elif isinstance(element, chord.Chord):
                for p in element.pitches:
                    notes_data.append({
                        'pitch': p.midi,
                        'vel': element.volume.velocity if element.volume.velocity else 64,
                        'dur': float(element.quarterLength),
                        'offset': float(element.offset)
                    })
    except Exception as e:
        print(f"    [!] 解析失败: {midi_path} ({e})")
        return None

    if len(notes_data) < 2:
        return None

    # 转化为 NumPy 数组方便计算
    pitches = np.array([n['pitch'] for n in notes_data])
    vels = np.array([n['vel'] for n in notes_data])
    durs = np.array([n['dur'] for n in notes_data])
    offsets = np.array([n['offset'] for n in notes_data])

    # 按照发声时间排序，用于计算音程和 IOI (发声间隔)
    sort_idx = np.argsort(offsets)
    pitches_sorted = pitches[sort_idx]
    offsets_sorted = offsets[sort_idx]

    # 衍生数据
    pitch_classes = pitches % 12
    iois = np.diff(offsets_sorted)  # Inter-Onset Intervals (音符发声间隔)
    intervals = np.abs(np.diff(pitches_sorted))  # 音程跳跃绝对值

    # --- 开始计算 30 个手工特征 ---
    features = {}

    # 【音高特征 Pitch】 (1-10)
    features['f1_note_count'] = len(pitches)
    features['f2_pitch_mean'] = np.mean(pitches)
    features['f3_pitch_std'] = np.std(pitches)
    features['f4_pitch_range'] = np.max(pitches) - np.min(pitches)
    features['f5_unique_pitches'] = len(np.unique(pitches))
    features['f6_unique_pitch_classes'] = len(np.unique(pitch_classes))

    # 音级分布香农熵
    pc_counts = list(Counter(pitch_classes).values())
    pc_probs = [c / sum(pc_counts) for c in pc_counts]
    features['f7_pitch_class_entropy'] = -sum(p * math.log2(p) for p in pc_probs if p > 0)

    features['f8_most_common_pc_ratio'] = max(pc_probs) if pc_probs else 0
    features['f9_high_pitch_ratio'] = np.sum(pitches > 72) / len(pitches)  # 大于 C5 的高音比例
    features['f10_low_pitch_ratio'] = np.sum(pitches < 48) / len(pitches)  # 小于 C3 的低音比例

    # 【力度/动态特征 Velocity】 (11-15)
    features['f11_vel_mean'] = np.mean(vels)
    features['f12_vel_std'] = np.std(vels)
    features['f13_vel_range'] = np.max(vels) - np.min(vels)
    features['f14_loud_note_ratio'] = np.sum(vels > 80) / len(vels)
    features['f15_soft_note_ratio'] = np.sum(vels < 40) / len(vels)

    # 【节奏/时值特征 Rhythm】 (16-23)
    features['f16_dur_mean'] = np.mean(durs)
    features['f17_dur_std'] = np.std(durs)
    features['f18_dur_max'] = np.max(durs)

    if len(iois) > 0:
        features['f19_ioi_mean'] = np.mean(iois)
        features['f20_ioi_std'] = np.std(iois)
    else:
        features['f19_ioi_mean'] = 0
        features['f20_ioi_std'] = 0

    total_time = np.max(offsets) - np.min(offsets) + 0.001
    features['f21_note_density'] = len(pitches) / total_time  # 每拍音符密度
    features['f22_staccato_ratio'] = np.sum(durs < 0.5) / len(durs)  # 短音/断奏比例
    features['f23_legato_ratio'] = np.sum(durs >= 2.0) / len(durs)  # 长音/连奏比例

    # 【旋律/音程特征 Interval】 (24-30)
    if len(intervals) > 0:
        features['f24_interval_mean'] = np.mean(intervals)
        features['f25_interval_std'] = np.std(intervals)
        features['f26_interval_max'] = np.max(intervals)
        features['f27_unison_ratio'] = np.sum(intervals == 0) / len(intervals)  # 同音重复比例
        features['f28_stepwise_ratio'] = np.sum((intervals == 1) | (intervals == 2)) / len(intervals)  # 级进比例 (小大二度)
        features['f29_leap_ratio'] = np.sum(intervals > 4) / len(intervals)  # 跳进比例 (大于三度)
        features['f30_octave_leap_ratio'] = np.sum(intervals == 12) / len(intervals)  # 八度大跳比例
    else:
        features['f24_interval_mean'] = 0
        features['f25_interval_std'] = 0
        features['f26_interval_max'] = 0
        features['f27_unison_ratio'] = 0
        features['f28_stepwise_ratio'] = 0
        features['f29_leap_ratio'] = 0
        features['f30_octave_leap_ratio'] = 0

    return features


# ==========================================
# 2. 批量提取流水线
# ==========================================
if __name__ == "__main__":
    METADATA_CSV = r"E:\Master\symbolic_2026\dataset\metadata.csv"
    SCORE_DIR = r"E:\Master\symbolic_2026\midi_files"

    print("[*] 读取 Metadata...")
    df = pd.read_csv(METADATA_CSV)
    df = df[df['composer'] != 'Clara Schumann'].copy()


    extracted_data = []

    print(f"[*] 开始纯手工提取 {len(df)} 首曲目的 30 维统计特征...")
    for index, row in df.iterrows():
        midi_filename = row['filename'].replace('.mxl', '.mid')
        file_path = os.path.join(SCORE_DIR, midi_filename)

        if not os.path.exists(file_path):
            continue

        print(f"  -> [{index + 1}/{len(df)}] 提取: {midi_filename}")
        feats = extract_30_custom_features(file_path)

        if feats is not None:
            feats['filename'] = row['filename']
            feats['composer'] = row['composer']
            extracted_data.append(feats)

    # 保存特征
    out_df = pd.DataFrame(extracted_data)
    out_csv = "handmade_30_features.csv"
    out_df.to_csv(out_csv, index=False)
    # 在读取 CSV 之后紧接着加上这行：
    out_df['composer'] = out_df['composer'].replace('Johannes Brahms (1833-1897)', 'Johannes Brahms')
    print(f"\n[+] 提取完成！特征已保存至 {out_csv}")


    print("\n=====================================================")
    print("  手搓 30 维基础特征 vs SVM 分类表现 ")
    print("=====================================================\n")

    # 2. 关键修复：直接踢掉 'filename' 和 'composer'，剩下的全是纯数字特征
    feature_cols = [c for c in out_df.columns if c not in ['filename', 'composer']]
    X = out_df[feature_cols].values

    # 填补可能出现的 NaN (以防有些曲子算不出某些特征)
    X = np.nan_to_num(X)

    y_labels = out_df['composer'].values

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 使用与之前完全相同的 SVM 参数
    svm_model = SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='balanced_accuracy')
    print(f"SVM (Handmade 30D Naive Baseline): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    y_pred = cross_val_predict(svm_model, X_scaled, y, cv=cv)
    print("\n--- 详细分类报告 ---")
    print(classification_report(y, y_pred, target_names=le.classes_))