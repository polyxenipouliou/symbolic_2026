# 1. Introduction

**Target Length**: 1 page (ISMIR 6-page limit)  
**Format**: Two-column, 10pt Times

---

## 1.1 Problem Statement

Classifying musical compositions by their composer has been a longstanding challenge in computational musicology. While significant progress has been made in distinguishing between different historical eras or genres, accurately identifying individual composers working within the same period and stylistic tradition remains difficult even with modern machine learning techniques (Alvarez et al., 2024). This challenge is particularly pronounced in the German Lieder tradition of the Romantic era, where composers such as Franz Schubert (1797-1828), Robert Schumann (1810-1856), and Johannes Brahms (1833-1897) shared a common musical vocabulary while developing distinct individual styles.

The fundamental question underlying this research is whether computational methods can capture and quantify the subtle stylistic markers that musicologists recognize intuitively—the composer's "fingerprint" that distinguishes Schubert's lyrical simplicity from Schumann's poetic intensity and Brahms' classical restraint. Early approaches to this problem relied on statistical characterization of musical surfaces (Youngblood, 1958), while more recent work has employed sophisticated machine learning architectures including Support Vector Machines, Hidden Markov Models, and deep neural networks. However, many of these studies focus on era-level classification or use features that lack direct musicological interpretability.

This project adopts an **exploratory** perspective: rather than claiming definitive composer attribution, we investigate which computational features show promise for capturing stylistic individuality and how these features align with musicological understanding of each composer's approach to the Lieder genre.

---

## 1.2 Research Questions

This exploratory study is guided by three research questions:

**RQ1: Can symbolic features capture composer-specific patterns in German Lieder?**

This foundational question asks whether the computational representation of musical elements—pitch, rhythm, texture, harmony—contains sufficient information to distinguish between composers who worked within the same genre and historical period. A positive answer would support the broader research program of computational stylistics in music.

**RQ2: Which musical dimensions (harmony, texture, melody) are most discriminative for composer classification?**

By comparing the relative importance of different feature categories, we aim to identify which aspects of musical style are most characteristic of individual composers. Our theoretical framework predicts that pianistic texture and melodic contour may be more discriminative than harmonic features, given the shared tonal language of the Romantic era.

**RQ3: How do handcrafted, domain-specific features compare to pretrained transformer representations for composer classification with limited training data?**

This question addresses a practical concern in computational musicology: classical music corpora are often too small to effectively train or fine-tune large pretrained models. We compare our 54-dimensional handcrafted feature set (excluding note count and velocity) against 768-dimensional embeddings from Adversarial-MidiBERT with both SVM and MLP classifiers to determine whether domain knowledge encoded in feature design can compensate for limited data.

---

## 1.3 Contributions

This paper makes the following contributions:

1. **Multi-layer Feature Framework**: We propose a theoretically-grounded approach combining three complementary feature categories—tonal tension (Spiral Array Model), harmonic complexity (pitch class entropy), and pianistic texture (onset density)—with detailed musicological justification for each design choice.

2. **Systematic Comparison**: We present a comprehensive comparison of handcrafted features (12D, 54D) against pretrained transformer embeddings (768D MidiBERT) with both SVM and MLP classifiers, with results showing handcrafted features achieve ~67% balanced accuracy (with feature selection) versus 47.1% (SVM) and 45.3% (MLP) for MidiBERT on our 264-piece corpus.

3. **Feature Importance Analysis**: Using Random Forest importance ranking and ANOVA, we identify the most discriminative features (unison ratio, texture variation, pitch standard deviation) and provide musicological interpretation of why these features capture composer-specific patterns.

4. **Methodological Rigor**: We explicitly exclude note count (confounding variable) and velocity features (editorial bias) from our analysis, demonstrating that classification can achieve meaningful accuracy based on genuine musical features alone.

5. **Reproducible Pipeline**: We release all code, extracted features, and experimental configurations to support reproducible research in computational composer classification.

---

## 1.4 Exploratory Framing

It is important to emphasize the exploratory nature of this research. We do not claim that our feature set provides definitive composer attribution, nor do we suggest that computational analysis can replace musicological expertise. Rather, we position this work as:

- **Hypothesis Generation**: Computational findings can suggest new avenues for musicological investigation (e.g., "Why is unison ratio the most discriminative feature?")

- **Quantitative Complement**: Statistical analysis complements qualitative musicological analysis by providing measurable evidence for stylistic observations

- **Methodological Investigation**: We explore which computational approaches are most promising for future work on larger, more diverse corpora

This framing acknowledges the limitations of our approach while highlighting its potential value for both computational musicology and traditional musicological research.

---

## 1.5 Paper Organization

The remainder of this paper is organized as follows: Section 2 reviews relevant background from both musicological and computational perspectives. Section 3 describes our methodology, including dataset construction, feature design rationale (including exclusions), and classification framework. Section 4 presents experimental results on classification performance and feature discriminability. Section 5 provides musicological interpretation of computational findings. Section 6 discusses exploratory case studies and unexpected findings. Section 7 concludes with directions for future work.

---

## Introduction Writing Notes

### Key Citations to Include:
- Alvarez et al. (2024) - Recent composer classification work
- Youngblood (1958) - Foundational statistical approach
- Simonetta (2025) - Computational framing
- Herremans & Chew (2016, 2019) - Spiral Array Model

### Tone Considerations:
- Use "exploratory," "investigate," "suggest" rather than "prove," "demonstrate"
- Acknowledge limitations while highlighting contributions
- Balance technical and musicological language for ISMIR audience

### Double-Blind Compliance:
- Third-person perspective throughout
- No self-citations to unpublished work
- Anonymous references where necessary

### Key Updates from Previous Version:
- Updated accuracy numbers (~67%, 47.1%, 45.3%)
- Updated feature dimensions (54D not 55D or 60D)
- Added note count exclusion to contributions
- Added MLP classifier comparison
- Updated top features (unison ratio, pt_std, pitch_std)

### Length Management:
- Current draft: ~1.5 pages
- May need to condense for final 6-page limit
- Consider moving some background to Section 2

---

## Revision Checklist

- [x] Update accuracy numbers (~67%, 47.1%, 45.3%)
- [x] Update feature dimensions (54D)
- [x] Add note count exclusion to contributions
- [x] Add MLP classifier comparison
- [x] Update top features in RQ3
- [ ] Verify all citations are in IEEE format
- [ ] Ensure research questions are clearly stated
- [ ] Check that contributions are specific and measurable
- [ ] Confirm exploratory framing is consistent throughout
- [ ] Review for double-blind compliance
- [ ] Verify word count fits ISMIR page limits

---

## Next Steps

1. Complete background section to support introduction claims
2. Ensure methodology section addresses all research questions
3. Verify results section provides answers to RQ1-RQ3
4. Cross-reference contributions with actual experimental findings
