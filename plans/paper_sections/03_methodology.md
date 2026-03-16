# 3. Methodology

**Target Length**: 1.5 pages  
**Format**: Two-column, 10pt Times

---

## 3.1 Dataset

Our dataset comprises 264 German Lieder drawn from two primary sources:

**OpenScore Lieder Repository**: A crowdsourced collection of public domain scores in MusicXML format, providing high-quality digital editions of classical Lieder.

**Schubert Winterreise Dataset**: A specialized scholarly resource containing multiple editions of Schubert's seminal song cycle, offering opportunities for comparative analysis.

### 3.1.1 Corpus Composition

The final dataset includes works by three Romantic-era composers:

| Composer | Pieces | Percentage | Date Range |
|----------|--------|------------|------------|
| Franz Schubert | 84 | 31.8% | 1815-1828 |
| Johannes Brahms | 109 | 41.3% | 1852-1896 |
| Robert Schumann | 71 | 26.9% | 1840-1852 |
| **Total** | **264** | **100%** | **1815-1896** |

*Table 1: Dataset composition by composer.*

The class distribution reflects the relative size of each composer's Lieder output in the source repositories. While imbalanced, this distribution is representative of the available corpus and is addressed through balanced accuracy metrics and class-weighted loss functions.

### 3.1.2 Preprocessing Pipeline

All scores underwent the following preprocessing steps:

1. **Format Conversion**: MusicXML files (.mxl) converted to MIDI (.mid) using MuseScore's command-line batch converter for compatibility with feature extraction tools.

2. **Voice Separation**: Using music21 and partitura, we separated vocal and piano parts based on instrument designation in the score metadata.

3. **Quality Filtering**: Pieces with missing voice parts, incomplete measures, or parsing errors were excluded. A list of error files was maintained for transparency.

4. **Meter Detection**: Time signature extracted from score metadata to determine the metric grid for feature computation (quarter-note level for 2/4, 3/4, 4/4; eighth-note level for 6/8).

5. **Composer Normalization**: Variants of composer names (e.g., "Johannes Brahms (1833-1897)" vs. "Johannes Brahms") were standardized.

---

## 3.2 Feature Design Rationale

Our feature design is grounded in musicological theory about composer style in the German Lieder tradition. Rather than extracting all available features, we selected dimensions that correspond to hypothesized stylistic differences.

### 3.2.1 Tonal Tension: Measuring Harmonic Instability

**Theoretical Foundation**: The Spiral Array Model (Herremans & Chew, 2016, 2019) provides a geometric representation of tonal space that captures relationships between pitches, chords, and keys. The model maps pitch classes onto a 3D helical structure where spatial distance corresponds to tonal relatedness.

**Musical Hypothesis**: Each composer, while working within the shared tonal system, develops individual patterns of harmonic tension and resolution. Schubert's text-driven chromaticism, Schumann's expressive modulations, and Brahms' conservative harmony should manifest as different tonal tension profiles.

**Computational Operationalization**:
- For each metric grid position (beat), collect all sounding pitches
- Map pitches to Spiral Array 3D coordinates
- Compute centroid of pitch distribution
- Calculate Euclidean distance from centroid to each pitch
- Output: mean distance as tonal tension value (higher = more dispersed/unstable)

**Features Extracted**:
- `tt_mean`: Average tonal tension across the piece
- `tt_std`: Variation in tension (dynamic harmonic rhythm)
- `tt_entropy`: Distributional entropy of tension values

### 3.2.2 Harmonic Complexity: Pitch Class Saturation

**Theoretical Foundation**: Pitch class distribution entropy measures the uniformity of chromatic content. Lower entropy indicates concentration around few pitch classes (diatonic writing), while higher entropy suggests chromatic saturation (Kaliakatsos et al., 2011).

**Musical Hypothesis**: Brahms' conservative harmonic language should yield lower entropy than Schumann's chromatic explorations. Schubert's expressive word-painting may show intermediate values with context-dependent variation.

**Computational Operationalization**:
- For each beat, collect all sounding pitch classes (0-11)
- Compute frequency distribution over 12 pitch classes
- Calculate Shannon entropy: H = -Σ pᵢ log₂(pᵢ)
- Normalize by maximum entropy (log₂(12) ≈ 3.58)

**Features Extracted**:
- `hc_mean`: Average harmonic complexity
- `hc_std`: Variation in complexity
- `hc_entropy`: Distributional entropy of complexity values

### 3.2.3 Pianistic Texture: Onset Density

**Theoretical Foundation**: Texture in Lieder accompaniment ranges from sparse (single-line bass) to dense (full chordal writing). Giraud et al. (2014) modeled texture in perceptual layers; we adopt a simpler onset-based approach.

**Musical Hypothesis**: Schubert's guitar-like arpeggiation should yield lower simultaneity than Brahms' chordal writing. Schumann's flowing arpeggios may show intermediate values with high variation.

**Computational Operationalization**:
- For each beat in piano part only, count simultaneous note onsets
- Exclude sustained notes (only count new attacks)
- Compute mean, variance, and distribution of onset counts

**Features Extracted**:
- `pt_mean`: Average onset density
- `pt_std`: Variation in texture (arpeggiation vs. block chords)
- `pt_entropy`: Distributional entropy of texture values

### 3.2.4 Melodic Contour: Interval Succession Patterns

**Theoretical Foundation**: Melodic shape carries significant stylistic information (Pollastri & Simoncelli, 2003). The distribution of interval sizes and directions reflects compositional habits and vocal writing style.

**Musical Hypothesis**: Schubert's lyrical melodies should favor stepwise motion. Schumann's declamatory style may show wider leaps. Brahms' folk-influenced melodies may combine both approaches.

**Computational Operationalization**:
- Extract vocal line as sequence of MIDI pitches
- Compute interval between successive notes (absolute value)
- Categorize intervals: unison (0), step (1-2), leap (3-7), large leap (8+)
- Calculate distribution statistics

**Features Extracted**:
- `mc_mean`: Average interval size
- `mc_std`: Variation in interval sizes
- `mc_entropy`: Entropy of interval distribution

---

## 3.3 Feature Extraction Pipeline

### 3.3.1 Metric Grid Alignment

A critical design decision is the temporal granularity for feature computation. We align feature extraction to the metric grid rather than using fixed time windows:

| Meter | Grid Unit | Rationale |
|-------|-----------|-----------|
| 2/4 | Quarter note | Beat-level harmony |
| 3/4 | Quarter note | Waltz meter beat |
| 4/4 | Quarter note | Standard beat |
| 6/8 | Eighth note | Compound meter subdivision |

This ensures that harmonic changes are captured at musically meaningful boundaries rather than arbitrary time divisions.

### 3.3.2 Bar-Level Representation

Each bar is represented as a matrix of feature values at the metric grid:

```
Bar (3/4 meter) → 3 grid positions → 3×3 feature matrix
                  [tt, hc, pt] at beat 1
                  [tt, hc, pt] at beat 2
                  [tt, hc, pt] at beat 3
```

### 3.3.3 Piece-Level Aggregation

For classification, bar-level matrices are aggregated into piece-level statistics:

- **Mean**: Central tendency of each feature
- **Standard Deviation**: Variation within the piece
- **Entropy**: Distributional complexity

This yields a 12-dimensional feature vector per piece (3 base features × 3 statistics + 3 melodic features × 1 statistic).

---

## 3.4 Handmade Feature Extension (54D)

To complement the theory-driven 12D feature set, we extracted 54 additional statistical features across four musical dimensions:

### 3.4.1 Pitch Features (f2-f10)
- Pitch mean/std/range (f2-f4)
- Unique pitch classes, pitch class entropy (f5-f10)
- Register preferences (high/low pitch ratios)

### 3.4.2 Rhythm/Duration Features (f16-f23)
- Duration statistics
- Inter-onset intervals
- Note density, articulation (staccato/legato ratios)

### 3.4.3 Interval Features (f24-f30, f52-f60)
- Interval statistics (mean, std, max)
- Specific interval ratios (unison, stepwise, leaps)
- Fine-grained interval categories (minor 3rd, perfect 5th, etc.)

### 3.4.4 Texture Features (f47-f50)
- Simultaneity statistics
- Chord thickness measures

### 3.4.5 Higher-Order Statistics (f31-f34, f35-f46)
- Skewness and kurtosis of distributions
- Pitch class histogram (12 bins)

### 3.4.6 Excluded Features

**Note Count (f1)**: Initially included but removed from final analysis. Note count serves as a proxy for piece length, which may confound stylistic analysis with formal/structural choices unrelated to composer-specific musical language. By excluding this feature, we test whether genuine musical features (interval patterns, texture, rhythm) can achieve competitive classification performance.

**Velocity Features (f11-f15)**: MIDI velocity values in symbolic datasets often reflect editorial conventions of score transcribers rather than composer intent, as historical scores from the Romantic era specify dynamics qualitatively (e.g., *p*, *f*) rather than as numerical values (1-127). To ensure our model captures genuine stylistic patterns rather than data source artifacts, we excluded velocity features from our primary analysis.

---

## 3.5 MidiBERT Feature Extraction

For comparison with handcrafted features, we extracted 768-dimensional embeddings using Adversarial-MidiBERT (Zhao, 2025):

1. **Octuple Encoding**: Each note encoded as 8-tuple (TimeSig, Tempo, Bar, Position, Instrument, Pitch, Duration, Velocity)

2. **Model Forward Pass**: Pretrained transformer processes sequence, producing hidden states (batch × seq_len × 768)

3. **Attention Pooling**: Mean pooling over sequence dimension, weighted by attention mask

4. **Piece Embedding**: Single 768-dimensional vector per piece

This approach requires no manual feature engineering but produces uninterpretable representations.

---

## 3.6 Classification Framework

### 3.6.1 Model Selection

We employ two classifiers for comparison:

**Support Vector Machines (SVM)** with RBF kernel:
- **Rationale**: SVMs are effective for high-dimensional data with limited samples
- **Class Balancing**: `class_weight='balanced'` adjusts for imbalanced dataset
- **Hyperparameters**: C=5.0 (determined through preliminary experiments)

**Multi-Layer Perceptron (MLP)**:
- **Architecture**: Input → Hidden1 (128, ReLU, Dropout 0.3) → Hidden2 (64, ReLU, Dropout 0.3) → Output (3, softmax)
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Early stopping (patience=20) to prevent overfitting
- **Rationale**: Assess whether non-linear decision boundaries improve performance

### 3.6.2 Evaluation Protocol

**5-Fold Stratified Cross-Validation**:
- Data split into 5 folds preserving class distribution
- Each fold serves as test set once
- Results averaged across folds

**Balanced Accuracy**:
```
Balanced Accuracy = (Recall_class1 + Recall_class2 + Recall_class3) / 3
```
This metric treats all classes equally regardless of sample size.

### 3.6.3 Feature Importance Analysis

**Random Forest Ranking**:
- Train Random Forest classifier (500 estimators)
- Extract Gini importance for each feature
- Rank features by importance score

**Incremental Validation**:
- Add features in order of importance
- Train SVM at each step
- Plot accuracy curve to identify optimal subset

### 3.6.4 ANOVA Discriminability Analysis

**One-Way ANOVA**:
- Test whether feature means differ significantly across composers
- F-statistic: ratio of between-group to within-group variance
- p-value: probability of observing F under null hypothesis (no difference)

Features with p < 0.05 are considered significantly discriminative.

---

## Methodology Section Writing Notes

### Key Updates from Previous Version:
- Updated feature dimensions: 55D → 54D
- Added note count exclusion rationale (Section 3.4.6)
- Clarified that velocity features are also excluded

### Reproducibility:
- All preprocessing steps documented
- Feature extraction formulas provided
- Hyperparameters specified
- Code available in repository

### Musicological Justification:
- Each feature linked to theoretical hypothesis
- Design decisions explained (metric grid, aggregation)
- Connection to prior literature maintained

### Methodological Rigor:
- Note count exclusion justified (avoid confounding)
- Velocity exclusion justified (avoid editorial bias)
- Both decisions strengthen validity of findings

### Length Management:
- Current draft: ~2.5 pages
- May need to condense feature descriptions
- Consider moving detailed feature tables to appendix

---

## Revision Checklist

- [x] Update feature dimensions (54D not 55D or 60D)
- [x] Add note count exclusion rationale
- [x] Keep velocity exclusion rationale
- [ ] Verify all formulas are correct
- [ ] Ensure hyperparameters match actual experiments
- [ ] Check that feature names match code implementation
- [ ] Confirm table formatting meets ISMIR requirements
- [ ] Review for clarity and completeness
- [ ] Verify word count fits within page limits

---

## Next Steps

1. Complete results section with actual experimental data
2. Ensure results address all methodology components
3. Cross-reference feature definitions with discussion section
4. Prepare figures (feature extraction pipeline, accuracy curves)
