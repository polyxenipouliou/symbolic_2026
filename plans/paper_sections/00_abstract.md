# Abstract

**Target Length**: 150-200 words  
**Position**: Top left column (ISMIR format)

---

## Draft Abstract (Revised - 54D Features, 768D Baselines)

Classifying musical compositions by their composer remains a challenging task in computational musicology, particularly when distinguishing between composers working within the same historical period and genre. This exploratory study investigates whether individual composer style can be captured through computational analysis of symbolic music representations in German Lieder. We propose a three-layer feature framework combining **tonal tension** (computed via the Spiral Array Model), **harmonic complexity** (measured as pitch class entropy), and **pianistic texture** (operationalized as onset density) to classify 264 Lieder by Franz Schubert, Robert Schumann, and Johannes Brahms. Our experiments demonstrate that carefully designed handcrafted features (54 dimensions, excluding note count and velocity to avoid confounding variables) achieve approximately **67% balanced accuracy** using a Support Vector Machine classifier with feature selection (top 20 features). This significantly outperforms both SVM (**47.1%**) and MLP (**45.3%**) classifiers on 768-dimensional pretrained transformer embeddings from Adversarial-MidiBERT on this limited corpus. Feature importance analysis reveals that **unison ratio, texture variation, and pitch standard deviation** are the most discriminative attributes. ANOVA results confirm significant between-composer variance in texture and melody features ($p < 0.05$). These findings suggest that domain-specific, interpretable features encoding musicological knowledge may be more effective than general-purpose pretrained representations when working with small, specialized corpora.

**Keywords**: composer classification, computational musicology, feature extraction, German Lieder, symbolic music analysis

---

## Abstract Writing Notes

### Key Elements Covered:
1. ✅ **Problem**: Composer classification difficulty within same period
2. ✅ **Method**: Three-layer feature framework (54D, excluding note count and velocity)
3. ✅ **Dataset**: 264 Lieder, three composers
4. ✅ **Key Results**: 
   - 54D SVM: ~67% (top 20 features)
   - 768D SVM: 47.1%
   - 768D MLP: 45.3%
5. ✅ **Top Features**: unison ratio, texture std, pitch std
6. ✅ **Implication**: Domain-specific features > pretrained for small corpora

### Word Count Check:
- Current draft: ~200 words
- ISMIR requirement: 150-200 words ✓

### Key Changes from Previous Version:

| Aspect | Previous | Current |
|--------|----------|---------|
| Feature dimensions | 55D | 54D (note count removed) |
| Best accuracy | 65.0% (55D), ~70% (21D) | ~63% (54D), ~67% (20D) |
| Top features | note_count, unison, stepwise | unison, pt_std, pitch_std |
| 768D SVM | Not reported | 47.1% |
| 768D MLP | 45.3% | 45.3% |

### Musicological Framing:
- Emphasizes "exploratory" nature
- Connects features to musical concepts (tonal tension, texture)
- Acknowledges shared tradition (same period/genre)
- Notes exclusions (note count, velocity) with rationale

### Revision Checklist:
- [x] Remove note count from top features
- [x] Update accuracy numbers (~67%, 47.1%, 45.3%)
- [x] Update top features (unison ratio, pt_std, pitch_std)
- [x] Add 768D SVM baseline (47.1%)
- [x] Note both exclusions (note count and velocity)
- [ ] Ensure third-person perspective (double-blind)
- [ ] Verify all statistics match final experimental data
