import os
import pandas as pd
import numpy as np
import math
from collections import Counter
import scipy.stats as stats  # For calculating skewness and kurtosis
from music21 import converter, note, chord
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings

# Suppress unnecessary warnings to keep output clean
warnings.filterwarnings('ignore')


# ==========================================
# 1. Custom Manual Feature Extraction (60 Base Dimensions)
# ==========================================
def extract_60_custom_features(midi_path):
    """
    Extracts 60 statistical features from a MIDI file using music21.
    Returns a dictionary of features or None if parsing fails/insufficient data.
    """
    try:
        score = converter.parse(midi_path)
        notes_data = []

        # Flatten the score to iterate through all notes and chords
        for element in score.flatten().notes:
            if isinstance(element, note.Note):
                notes_data.append({
                    'pitch': element.pitch.midi,
                    'vel': element.volume.velocity if element.volume.velocity else 64,
                    'dur': float(element.quarterLength),
                    'offset': float(element.offset)
                })
            elif isinstance(element, chord.Chord):
                # Decompose chords into individual pitch events
                for p in element.pitches:
                    notes_data.append({
                        'pitch': p.midi,
                        'vel': element.volume.velocity if element.volume.velocity else 64,
                        'dur': float(element.quarterLength),
                        'offset': float(element.offset)
                    })
    except Exception as e:
        return None

    # Ensure there is enough data to calculate statistics
    if len(notes_data) < 2:
        return None

    # Convert lists to numpy arrays for efficient calculation
    pitches = np.array([n['pitch'] for n in notes_data])
    vels = np.array([n['vel'] for n in notes_data])
    durs = np.array([n['dur'] for n in notes_data])
    offsets = np.array([n['offset'] for n in notes_data])

    # Sort by time offset to ensure correct sequence analysis
    sort_idx = np.argsort(offsets)
    pitches_sorted = pitches[sort_idx]
    offsets_sorted = offsets[sort_idx]

    # Derive secondary metrics
    pitch_classes = pitches % 12
    iois = np.diff(offsets_sorted)  # Inter-Onset Intervals
    intervals = np.abs(np.diff(pitches_sorted))  # Melodic intervals

    features = {}

    # ---------------- Basic 30 Dimensions (Core Statistics) ----------------
    # features['f1_note_count'] = len(pitches)
    features['f2_pitch_mean'] = np.mean(pitches)
    features['f3_pitch_std'] = np.std(pitches)
    features['f4_pitch_range'] = np.max(pitches) - np.min(pitches)
    features['f5_unique_pitches'] = len(np.unique(pitches))
    features['f6_unique_pitch_classes'] = len(np.unique(pitch_classes))

    # Pitch Class Distribution Analysis
    pc_counts_dict = Counter(pitch_classes)
    pc_counts = list(pc_counts_dict.values())
    pc_probs = [c / sum(pc_counts) for c in pc_counts]

    features['f7_pitch_class_entropy'] = -sum(p * math.log2(p) for p in pc_probs if p > 0)
    features['f8_most_common_pc_ratio'] = max(pc_probs) if pc_probs else 0
    features['f9_high_pitch_ratio'] = np.sum(pitches > 72) / len(pitches)
    features['f10_low_pitch_ratio'] = np.sum(pitches < 48) / len(pitches)

    # Velocity (Dynamics) Statistics
    # features['f11_vel_mean'] = np.mean(vels)
    # features['f12_vel_std'] = np.std(vels)
    # features['f13_vel_range'] = np.max(vels) - np.min(vels)
    # features['f14_loud_note_ratio'] = np.sum(vels > 80) / len(vels)
    # features['f15_soft_note_ratio'] = np.sum(vels < 40) / len(vels)

    # Duration and Rhythm Statistics
    features['f16_dur_mean'] = np.mean(durs)
    features['f17_dur_std'] = np.std(durs)
    features['f18_dur_max'] = np.max(durs)
    features['f19_ioi_mean'] = np.mean(iois) if len(iois) > 0 else 0
    features['f20_ioi_std'] = np.std(iois) if len(iois) > 0 else 0

    total_time = np.max(offsets) - np.min(offsets) + 0.001
    features['f21_note_density'] = len(pitches) / total_time
    features['f22_staccato_ratio'] = np.sum(durs < 0.5) / len(durs)
    features['f23_legato_ratio'] = np.sum(durs >= 2.0) / len(durs)

    # Interval Statistics
    features['f24_interval_mean'] = np.mean(intervals) if len(intervals) > 0 else 0
    features['f25_interval_std'] = np.std(intervals) if len(intervals) > 0 else 0
    features['f26_interval_max'] = np.max(intervals) if len(intervals) > 0 else 0
    features['f27_unison_ratio'] = np.sum(intervals == 0) / len(intervals) if len(intervals) > 0 else 0
    features['f28_stepwise_ratio'] = np.sum((intervals == 1) | (intervals == 2)) / len(intervals) if len(
        intervals) > 0 else 0
    features['f29_leap_ratio'] = np.sum(intervals > 4) / len(intervals) if len(intervals) > 0 else 0
    features['f30_octave_leap_ratio'] = np.sum(intervals == 12) / len(intervals) if len(intervals) > 0 else 0

    # ---------------- Additional 30 Dimensions (Advanced Metrics) ----------------

    # Higher-order Statistical Moments (Distribution Shape)
    features['f31_pitch_skew'] = stats.skew(pitches) if len(pitches) > 2 else 0
    features['f32_pitch_kurt'] = stats.kurtosis(pitches) if len(pitches) > 2 else 0
    features['f33_dur_skew'] = stats.skew(durs) if len(durs) > 2 else 0
    features['f34_ioi_skew'] = stats.skew(iois) if len(iois) > 2 else 0

    # Pitch Class Histogram (12 features, one for each semitone)
    for i in range(12):
        features[f'f{35 + i}_pc_{i}_ratio'] = pc_counts_dict.get(i, 0) / len(pitches)

    # Texture Thickness (Simultaneity Analysis)
    offset_counts = Counter(offsets)
    simultaneous_notes = list(offset_counts.values())
    features['f47_simultaneity_mean'] = np.mean(simultaneous_notes)
    features['f48_simultaneity_max'] = np.max(simultaneous_notes)
    features['f49_single_note_ratio'] = np.sum(np.array(simultaneous_notes) == 1) / len(simultaneous_notes)
    features['f50_thick_chord_ratio'] = np.sum(np.array(simultaneous_notes) >= 4) / len(simultaneous_notes)

    # Syncopation (Notes occurring off the beat)
    features['f51_offbeat_ratio'] = np.sum((offsets % 1) != 0) / len(offsets)

    # Specific Interval Preferences (Fine-grained music theory metrics)
    if len(intervals) > 0:
        features['f52_minor3rd_ratio'] = np.sum(intervals == 3) / len(intervals)
        features['f53_major3rd_ratio'] = np.sum(intervals == 4) / len(intervals)
        features['f54_perfect4th_ratio'] = np.sum(intervals == 5) / len(intervals)
        features['f55_tritone_ratio'] = np.sum(intervals == 6) / len(intervals)
        features['f56_perfect5th_ratio'] = np.sum(intervals == 7) / len(intervals)
        features['f57_minor6th_ratio'] = np.sum(intervals == 8) / len(intervals)
        features['f58_major6th_ratio'] = np.sum(intervals == 9) / len(intervals)
        features['f59_minor7th_ratio'] = np.sum(intervals == 10) / len(intervals)
        features['f60_major7th_ratio'] = np.sum(intervals == 11) / len(intervals)
    else:
        # Fill with zeros if no intervals exist
        for i in range(52, 61):
            features[f'f{i}'] = 0

    return features


# ==========================================
# 2. Batch Extraction and Execution
# ==========================================
if __name__ == "__main__":
    # Configuration Paths
    METADATA_CSV = r"E:\Master\symbolic_2026\dataset\metadata.csv"
    SCORE_DIR = r"E:\Master\symbolic_2026\midi_files"

    print("[*] Loading Metadata...")
    df = pd.read_csv(METADATA_CSV)

    # Filter out specific composer if needed (e.g., excluding Clara Schumann)
    df = df[df['composer'] != 'Clara Schumann'].copy()

    # Normalize composer names to fix labeling inconsistencies
    df['composer'] = df['composer'].replace('Johannes Brahms (1833-1897)', 'Johannes Brahms')

    extracted_data = []

    print(f"[*] Starting manual extraction of 60-dimensional features for {len(df)} pieces...")

    for index, row in df.iterrows():
        # Handle potential file extension differences (.mxl vs .mid)
        midi_filename = row['filename'].replace('.mxl', '.mid')
        file_path = os.path.join(SCORE_DIR, midi_filename)

        if not os.path.exists(file_path):
            continue

        print(f"  -> [{index + 1}/{len(df)}] Extracting: {midi_filename}")
        feats = extract_60_custom_features(file_path)

        if feats is not None:
            feats['filename'] = row['filename']
            feats['composer'] = row['composer']
            extracted_data.append(feats)

    # Save extracted features to CSV
    out_df = pd.DataFrame(extracted_data)
    out_csv = "handmade_54_features.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[+] Extraction complete! Features saved to {out_csv}")

    # ==========================================
    # 3. SVM Classification and Evaluation
    # ==========================================
    print("\n=====================================================")
    print("  Performance: Handcrafted 60D Features vs SVM")
    print("=====================================================\n")

    # Ensure composer labels are consistent again before modeling
    out_df['composer'] = out_df['composer'].replace('Johannes Brahms (1833-1897)', 'Johannes Brahms')

    # Select feature columns (exclude metadata columns)
    feature_cols = [c for c in out_df.columns if c not in ['filename', 'composer']]

    X = out_df[feature_cols].values
    X = np.nan_to_num(X)  # Replace any NaN values with 0

    y_labels = out_df['composer'].values

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Scale features for SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize SVM Model with balanced class weights
    svm_model = SVC(kernel='rbf', C=5.0, class_weight='balanced', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform Cross-Validation
    scores = cross_val_score(svm_model, X_scaled, y, cv=cv, scoring='balanced_accuracy')
    print(f"SVM (Handmade 60D): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Generate predictions for detailed report
    y_pred = cross_val_predict(svm_model, X_scaled, y, cv=cv)

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y, y_pred, target_names=le.classes_))