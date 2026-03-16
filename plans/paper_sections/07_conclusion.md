# 7. Conclusion and Future Work

**Target Length**: 0.5 page  
**Format**: Two-column, 10pt Times

---

## 7.1 Summary of Findings

This exploratory study investigated whether computational features can capture individual composer style in German Lieder. Our key findings are:

**Finding 1: Handcrafted features outperform pretrained embeddings.** The 54-dimensional handcrafted feature set (excluding note count and velocity) achieved approximately 67% balanced accuracy with feature selection (top 20 features), significantly outperforming both SVM (47.1%) and MLP (45.3%) classifiers on 768-dimensional MidiBERT embeddings on our 264-piece corpus. This result challenges assumptions about the universal superiority of deep learning approaches and highlights the value of domain-specific feature engineering for small, specialized datasets.

**Finding 2: Interval and texture features are most discriminative.** After removing note count (which was previously the most important feature), unison ratio (f27), texture variation (pt_std), and pitch standard deviation (f3) emerged as the most important features. This suggests that melodic motion patterns and textural variation carry genuine composer-specific information independent of piece length.

**Finding 3: Compact feature subsets are sufficient.** Classification accuracy peaked at approximately 20 features (~67%), with no significant improvement from including additional dimensions. This finding has practical implications for feature selection in similar tasks and suggests that composer style may be captured by a relatively small set of musical attributes.

**Finding 4: Theory-driven features show moderate discriminability.** ANOVA analysis confirmed that multiple features across interval, texture, pitch, and rhythm categories show statistically significant between-composer variance (p < 0.05), validating our feature design approach.

**Finding 5: Methodological exclusions strengthen validity.** By excluding note count (confounding variable) and velocity features (editorial bias), we demonstrate that genuine stylistic markers exist in other musical dimensions, and that classification is based on composer-intentioned patterns rather than artifacts.

---

## 7.2 Contributions Revisited

Returning to the contributions outlined in Section 1:

1. **Multi-layer Feature Framework**: We demonstrated that combining tonal tension, harmonic complexity, and pianistic texture provides a theoretically grounded approach to composer classification, with empirical validation through ANOVA and feature importance analysis.

2. **Systematic Comparison**: Our head-to-head comparison of handcrafted features (54D) against pretrained transformer embeddings (768D MidiBERT) with both SVM and MLP classifiers provides evidence that domain knowledge encoded in feature design can compensate for—and even surpass—large pretrained representations when data is limited.

3. **Feature Importance Analysis**: The identification of unison ratio, texture variation, and pitch standard deviation as most discriminative offers new hypotheses for musicological investigation into composer style.

4. **Methodological Rigor**: We explicitly exclude note count and velocity features to avoid confounding variables and editorial bias, demonstrating that classification can achieve meaningful accuracy based on genuine musical features alone.

5. **Reproducible Pipeline**: All code, features, and experimental configurations are publicly available, supporting reproducible research in computational musicology.

---

## 7.3 Limitations

This study has several important limitations:

**Dataset Size**: With 264 pieces, our corpus is substantial for Lieder analysis but remains small for machine learning applications. Results should be validated on larger datasets as they become available.

**Composer Coverage**: We focused on three canonical composers. Generalizability to other Lieder composers (Wolf, Strauss, Mahler) remains untested.

**Feature Scope**: Our feature set captures pitch, rhythm, and texture but omits chord function, voice-leading quality, and formal structure—dimensions that musicologists consider important for style analysis.

**Symbolic Data Quality**: All features depend on the accuracy of source editions and digital transcriptions. Editorial variations may introduce systematic biases.

**Note Count and Velocity Exclusion**: While methodologically justified, excluding these features means our analysis does not capture piece length preferences or dynamic expression patterns that may carry stylistic information.

**Exploratory Nature**: As an exploratory study, findings should be treated as hypothesis-generating rather than definitive conclusions.

---

## 7.4 Future Work

Based on our findings and limitations, we identify several promising directions for future research:

### 7.4.1 Chord Tonal Distance

Our methodology section noted that chord tonal distance—measuring the harmonic distance between successive chords—could provide additional discriminative information. Implementing this feature requires reliable chord annotation, which may necessitate manual verification or improved automated harmony analysis.

### 7.4.2 Expanded Corpus

Extending analysis to additional composers would test the generalizability of our findings:
- **Hugo Wolf**: Represents late-Romantic, Wagner-influenced Lieder
- **Richard Strauss**: Bridges Lieder and orchestral song traditions
- **Gustav Mahler**: Expands to orchestral accompaniment

A larger corpus would also enable more sophisticated models, including sequential architectures that were ineffective with our current dataset.

### 7.4.3 GroupKFold Evaluation

Future work should implement GroupKFold cross-validation to prevent data leakage from song cycles. Songs from the same cycle (Winterreise, Dichterliebe) share stylistic features, and ensuring complete cycles are kept together during train/test splitting would provide more realistic performance estimates.

### 7.4.4 Multi-Modal Fusion

Combining symbolic features with audio-based representations could capture performance-level stylistic markers not present in score data. Recent work on audio-symbolic pretraining offers promising approaches for such fusion.

### 7.4.5 Cross-Genre Validation

Testing our feature framework on other vocal genres (aria, madrigal, chanson) would assess whether the identified discriminative features generalize beyond Lieder or are genre-specific.

### 7.4.6 Musicological Collaboration

The most promising direction is deeper collaboration between computational and traditional musicologists:
- **Hypothesis refinement**: Musicologists can suggest new features based on theoretical understanding
- **Result interpretation**: Computational findings can be contextualized within broader stylistic discourse
- **Validation**: Expert judgment can assess whether computational classifications align with musicological intuition

---

## 7.5 Broader Implications

This study contributes to ongoing discussions about the role of computational methods in musicology:

**Domain Knowledge vs. Data-Driven Learning**: Our results suggest that for specialized domains with limited data, encoding expert knowledge in feature design remains valuable despite advances in representation learning.

**Interpretability Matters**: Handcrafted features offer direct musicological interpretation, enabling dialogue between computational findings and theoretical understanding. Black-box models, while powerful, obscure this connection.

**Methodological Rigor**: The exclusion of note count and velocity features demonstrates the importance of considering confounding variables and data provenance in computational musicology research.

**Exploratory Computing**: Computational analysis need not provide definitive answers to be valuable. As hypothesis generators, computational methods can suggest new avenues for traditional musicological research.

**Reproducibility**: Making code and data publicly available enables validation, extension, and critique—essential practices for building cumulative knowledge in computational musicology.

---

## 7.6 Concluding Remarks

The question posed in our introduction—whether computational methods can capture the subtle stylistic fingerprints that distinguish Schubert, Schumann, and Brahms—receives a qualified affirmative answer. Our 54-dimensional feature set achieves classification accuracy substantially above chance (~67%), and feature importance analysis reveals patterns that align with (and occasionally challenge) musicological understanding.

However, the goal of this research is not automated attribution but enhanced understanding. By quantifying aspects of musical style, we create new tools for asking old questions: What makes Schubert's melodies distinctive? How does Schumann's piano writing differ from Brahms'? Can we measure the intuitive sense of stylistic identity that performers and scholars recognize?

The answers emerging from this computational exploration are provisional and incomplete. Yet they demonstrate the potential for productive dialogue between quantitative analysis and qualitative interpretation—a dialogue that enriches both computational musicology and traditional musicological inquiry.

---

## Conclusion Writing Notes

### Key Updates from Previous Version:
- Updated accuracy numbers (~67%, 47.1%, 45.3%)
- Updated top features (unison ratio, pt_std, pitch_std)
- Added note count exclusion to methodology discussion
- Added methodological rigor to contributions

### Tone:
- Confident but measured claims
- Acknowledge limitations honestly
- Emphasize exploratory contribution

### Structure:
- Summary → Contributions → Limitations → Future Work → Implications
- Each section builds toward broader significance

### Key Messages:
- Handcrafted features work for small corpora (~67% vs 47.1%/45.3%)
- Interval and texture features are most important
- Methodological exclusions (note count, velocity) strengthen validity
- Interpretability enables musicological dialogue
- Computational methods as hypothesis generators

### Length Management:
- Current draft: ~1.5 pages
- May need to condense future work section
- Consider moving broader implications to discussion section

---

## Revision Checklist

- [x] Update accuracy numbers (~67%, 47.1%, 45.3%)
- [x] Update top features (unison ratio, pt_std, pitch_std)
- [x] Add note count exclusion discussion
- [x] Add methodological rigor to contributions
- [ ] Verify all claims are supported by results section
- [ ] Ensure limitations are comprehensive and honest
- [ ] Check that future work is specific and actionable
- [ ] Review for appropriate hedging language
- [ ] Confirm connection to introduction research questions
- [ ] Verify word count fits within page limits

---

## Next Steps

1. Compile comprehensive references section
2. Review all sections for internal consistency
3. Prepare final figures and tables
4. Create assembly instructions for final paper
