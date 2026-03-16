# Composer Fingerprinting: A Multi-Layer Feature Approach for Lieder Authorship Attribution

## Abstract

This project investigates whether individual composer style can be captured through computational analysis of symbolic music representations. We propose a three-layer feature framework combining **tonal tension** (Spiral Array Model), **harmonic complexity** (pitch class entropy), and **pianistic texture** (onset density) to classify Lieder by Franz Schubert, Robert Schumann, and Johannes Brahms. Our experiments demonstrate that carefully designed handcrafted features (54 dimensions, excluding note count and velocity to avoid confounding variables) achieve approximately 67% balanced accuracy using a Support Vector Machine classifier with feature selection. This significantly outperforms both SVM (47.1%) and MLP (45.3%) classifiers on 768-dimensional pretrained transformer embeddings from Adversarial-MidiBERT on our 264-piece corpus.

## Dataset

| Composer | Number of Lieder | Percentage |
|----------|------------------|------------|
| Franz Schubert | 84 | 31.8% |
| Johannes Brahms | 109 | 41.3% |
| Robert Schumann | 71 | 26.9% |
| **Total** | **264** | **100%** |

**Data Sources:**
- [OpenScore Lieder Repository](https://github.com/OpenScore/Lieder)
- [Schubert Winterreise Dataset](https://winterreise.org/)

## Feature Sets

### 1. Statistical Features (12D)
Derived from the three-layer theoretical framework:
- **Tonal Tension** (`tt_mean`, `tt_std`, `tt_entropy`): Euclidean distance in Spiral Array space
- **Harmonic Complexity** (`hc_mean`, `hc_std`, `hc_entropy`): Pitch class distribution entropy
- **Melodic Contour** (`mc_mean`, `mc_std`, `mc_entropy`): Interval succession statistics
- **Pianistic Texture** (`pt_mean`, `pt_std`, `pt_entropy`): Onset density per beat

### 2. Handmade Features (54D)
Comprehensive statistical descriptors across four musical dimensions. **Note:** Note count (piece length proxy) and velocity features were excluded to avoid confounding variables and editorial bias.

| Category | Features | Count | Musical Interpretation |
|----------|----------|-------|----------------------|
| Pitch | f2-f10 | 9 | Range, register preference, pitch class distribution |
| Rhythm/Duration | f16-f23 | 8 | Note density, articulation (staccato/legato) |
| Intervals | f24-f30, f52-f60 | 17 | Melodic motion preferences (stepwise vs. leaps) |
| Texture | f47-f50 | 4 | Chord thickness, simultaneity |
| Higher-order | f31-f34, f35-f46 | 16 | Skewness, kurtosis, pitch class histogram |
| **Total** | | **54** | |

### 3. MidiBERT Embeddings (768D)
Pre-trained transformer representations extracted using [Adversarial-MidiBERT](https://github.com/RS2002/Adversarial-MidiBERT).

### 4. Combined Features (780D)
Concatenation of 12D statistical + 54D handmade + 768D MidiBERT embeddings.

## Installation

### Requirements
```
Python 3.8+
music21>=8.0
pandas>=1.5
numpy>=1.21
scikit-learn>=1.0
scipy>=1.7
matplotlib>=3.5
seaborn>=0.11
torch>=1.9
transformers>=4.0
```

### Install Dependencies
```bash
pip install music21 pandas numpy scikit-learn scipy matplotlib seaborn
pip install torch transformers
```

## Usage

### Feature Extraction

**Extract 54D Handmade Features:**
```bash
python 54.py
```

**Extract MidiBERT Embeddings:**
```bash
cd Adversarial-MidiBERT
python get_feature.py
```

### Classification Experiments

**Statistical Features (12D):**
```bash
python 12.py
```

**Handmade Features (54D):**
```bash
python see_importance.py  # Feature selection + SVM classification
```

**MidiBERT Embeddings (768D) with SVM:**
```bash
python 768classificationmean.py
```

**MidiBERT Embeddings (768D) with MLP:**
```bash
python training.py  # MLP classification
```

**Combined Features (12+54+768=780D):**
```bash
# Features merged in feature_12+54+768.csv
python see_importance.py  # Feature selection + classification
```

### Analysis

**ANOVA Discriminability Analysis:**
```bash
python anova_12.py
```

**Feature Importance & Selection Curve:**
```bash
python see_importance.py
```

## Results Summary

### Classification Performance (Balanced Accuracy)

| Feature Set | Dimensions | Classifier | Balanced Accuracy | Std Dev | Notes |
|-------------|------------|------------|-------------------|---------|-------|
| Statistical (12D) | 12 | SVM | 49.3% | 4.7% | Theory-driven only |
| Handmade (54D) | 54 | SVM | ~63% | ~6% | All features |
| Handmade (20D, top) | 20 | SVM | **~67%** | ~5% | Feature selection |
| MidiBERT (768D) | 768 | SVM | 47.1% | 2.5% | Pretrained embeddings |
| MidiBERT (768D) | 768 | MLP | 45.3% | - | Pretrained embeddings |
| Combined (780D) | 780 | SVM | TBD | TBD | All features |

### Key Findings

1. **Handcrafted features outperform pretrained embeddings** on small datasets (~67% vs 47.1%)
2. **Top discriminative features**: unison ratio (f27), texture std (pt_std), pitch std (f3), stepwise ratio (f28), melodic contour std (mc_std)
3. **Optimal feature subset**: ~20 features achieve peak performance (~67%)
4. **Note count and velocity excluded** to avoid confounding variables and editorial bias
5. **SVM outperforms MLP** on MidiBERT embeddings (47.1% vs 45.3%)

### Top 10 Most Important Features (Random Forest Importance, 54D)

| Rank | Feature | Importance | Category | Musical Meaning |
|------|---------|------------|----------|-----------------|
| 1 | f27_unison_ratio | 0.0308 | Interval | Repeated notes in melody |
| 2 | pt_std | 0.0287 | Texture | Texture variation |
| 3 | f3_pitch_std | 0.0286 | Pitch | Pitch range dispersion |
| 4 | f28_stepwise_ratio | 0.0280 | Interval | Stepwise melodic motion |
| 5 | mc_std | 0.0272 | Melody | Melodic contour variation |
| 6 | f4_pitch_range | 0.0261 | Pitch | Total pitch span |
| 7 | f22_staccato_ratio | 0.0259 | Rhythm | Short note proportion |
| 8 | f24_interval_mean | 0.0243 | Interval | Average melodic interval size |
| 9 | f8_most_common_pc_ratio | 0.0239 | Pitch | Most common pitch class ratio |
| 10 | f25_interval_std | 0.0238 | Interval | Interval size variation |

## Project Structure

```
symbolic_2026/
├── README.md                    # This file
├── 12.py                        # Statistical features classification
├── 30.py                        # 30D handmade feature extraction
├── 54.py                        # 54D handmade feature extraction
├── 768classificationmean.py     # MidiBERT classification with SVM
├── training.py                  # MLP classification
├── conbine_features.py          # Feature merging
├── anova_12.py                  # ANOVA analysis
├── see_importance.py            # Feature importance analysis
├── clean_data.py                # Data preprocessing
├── Adversarial-MidiBERT/        # MidiBERT model & extraction
│   ├── model.py
│   ├── get_feature.py
│   └── Octuple.pkl
├── dataset/                     # Original MusicXML scores
│   └── *.mxl
├── midi_files/                  # Converted MIDI files
│   └── *.mid
├── features/                    # Extracted feature CSVs
│   ├── features_statistical.csv
│   ├── features_sequential.csv
│   └── midibert_768d_features.csv
└── musif_output/                # musif library output
    └── jsymbolic_output/
```

## Musicological Interpretation

### What Features Reveal About Composer Style

**Franz Schubert:**
- Higher stepwise ratio (f28) reflects lyrical, vocal melody writing
- Lower texture density aligns with guitar-like accompaniment patterns
- Moderate harmonic complexity supports text expression

**Robert Schumann:**
- Higher texture variation (pt_std) indicates diverse accompaniment patterns
- Higher staccato ratio (f22) suggests varied articulation
- Complex rhythmic patterns reflect poetic declamation

**Johannes Brahms:**
- Higher pitch range (f4) indicates richer textures
- Conservative interval patterns (f27_unison_ratio) reflect classical influence
- Dense chordal writing shows symphonic piano writing

## Methodological Notes

### Note Count Exclusion
Note count (f1) was initially included as a feature but was removed because it serves as a proxy for piece length, which may confound stylistic analysis with formal/structural choices unrelated to composer-specific musical language.

### Velocity Feature Exclusion
Velocity features (f11-f15) were excluded because MIDI velocity values in symbolic datasets often reflect editorial conventions of score transcribers rather than composer intent. Historical scores from the Romantic era specify dynamics qualitatively (e.g., *p*, *f*) rather than as numerical values (1-127).

## Citation

If you use this code or dataset, please cite:

```bibtex
@unpublished{wu2026composer,
  title={Composer Fingerprinting: A Multi-Layer Feature Approach for Lieder Authorship Attribution},
  author={Wu, Yuhang and Pouliou, Jenny},
  note={Course Project, 2026},
  year={2026}
}
```

## License

This project is for educational and research purposes. Data sources (OpenScore, Winterreise Dataset) have their respective licenses.

## Acknowledgments

- [OpenScore](https://openscore.nl/) for the Lieder corpus
- [Adversarial-MidiBERT](https://github.com/RS2002/Adversarial-MidiBERT) team for the pretrained model
- Herremans & Chew for the Spiral Array Model implementation guidance
