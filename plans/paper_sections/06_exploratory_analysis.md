# 6. Exploratory Analysis

**Target Length**: 0.5-1 page  
**Format**: Two-column, 10pt Times

---

## 6.1 Case Study: Winterreise Texture Analysis

As an exploratory investigation, we analyzed the 24 songs of Schubert's *Winterreise* cycle separately to examine whether a unified compositional approach yields consistent feature profiles.

### 6.1.1 Feature Consistency Within Cycle

**Observation**: The Winterreise songs show lower variance in texture features (pt_std) compared to Schubert's other Lieder.

**Interpretation**: This suggests Schubert employed a coherent textural vocabulary throughout the cycle, supporting musicological observations about Winterreise's cyclical unity. The "wandering" motif—represented through arpeggiated, guitar-like accompaniment—appears consistently across songs.

---

## 6.2 Case Study: Dichterliebe Opening

Schumann's *Dichterliebe* (Op. 48) provides another opportunity for cycle-level analysis. The opening song, "Im wunderschönen Monat Mai," features one of the most famous arpeggiated introductions in the Lieder repertoire.

### 6.2.1 Quantifying Arpeggiation

**Computational Measurement**: The song shows low simultaneity and high texture variance, quantitatively confirming the qualitative observation of arpeggiated texture.

---

## 6.3 Feature Dominance Patterns

### 6.3.1 Unison Ratio (f27) as Primary Discriminator

**Observation**: After removing note count, f27_unison_ratio emerged as the most important feature (importance = 0.0308).

**Interpretation**: Repeated note patterns carry significant stylistic information:

- **Brahms**: Higher unison ratio reflects folk song influence and classical restraint
- **Schubert**: Moderate unison ratio balanced with stepwise motion
- **Schumann**: Lower unison ratio, more varied melodic motion

**Musicological Context**: This finding quantitatively confirms observations about each composer's melodic style. Brahms' use of repeated notes creates structural clarity, while Schubert's melodies favor stepwise motion for lyrical effect.

### 6.3.2 Texture Variation (pt_std) Significance

**Observation**: pt_std ranks 2nd in importance (0.0287).

**Interpretation**: Textural variation captures composer-specific accompaniment patterns:

- **Schumann**: High pt_std reflects diverse accompaniment styles within and across songs
- **Schubert**: Lower pt_std indicates consistent arpeggiated patterns
- **Brahms**: Moderate pt_std with higher mean simultaneity

### 6.3.3 Pitch Standard Deviation (f3) and Range (f4)

**Observation**: Both pitch features rank in top 6 (f3: 0.0286, f4: 0.0261).

**Interpretation**: Pitch distribution characteristics distinguish composers:

- **Brahms**: Higher values reflect symphonic piano writing and wide register exploitation
- **Schubert**: Moderate values support transparent textures
- **Schumann**: Variable values match expressive range

---

## 6.4 Combined Feature Analysis Insights

### 6.4.1 Handcrafted Features Dominate

**Observation**: In the combined 780D feature set (12D + 54D + 768D), handcrafted features occupy 9 of the top 10 positions.

**Interpretation**: This confirms that for this specific task (Lieder composer classification), domain-specific features encode more relevant information than general pretrained representations.

### 6.4.2 MidiBERT Embeddings in Combined Set

**Observation**: The first MidiBERT dimension (bert_dim_251) appears at rank 10 with importance 0.0055, substantially lower than top handcrafted features.

**Interpretation**: Pretrained embeddings contribute limited discriminative information for this task, possibly due to:
- Domain mismatch (popular music vs. art song)
- Small dataset preventing effective fine-tuning
- Handcrafted features already capturing relevant stylistic information

---

## 6.5 What Didn't Work: Lessons Learned

### 6.5.1 Note Count as Confounding Variable

**Approach**: Note count (f1) was initially included as a feature and ranked as most important.

**Problem**: Note count serves as a proxy for piece length, which may confound stylistic analysis with formal/structural choices unrelated to composer-specific musical language.

**Decision**: Removed from final analysis to test whether genuine musical features can achieve competitive performance.

**Outcome**: Model achieves ~67% accuracy without note count, demonstrating that interval, texture, and rhythm features carry genuine stylistic information.

**Lesson**: Careful feature selection must consider potential confounding variables to ensure valid stylistic analysis.

### 6.5.2 Velocity Feature Exclusion

**Approach**: Velocity features were initially considered but excluded.

**Rationale**: MIDI velocity values often reflect editorial conventions rather than composer intent.

**Outcome**: Model achieves strong performance without velocity features.

**Lesson**: Data provenance matters. Computational musicology must consider the chain of transmission from composer to digital representation.

### 6.5.3 MLP Underperforms SVM

**Observation**: MLP (45.3%) underperforms SVM (47.1%) on MidiBERT embeddings.

**Interpretation**: The small dataset (264 pieces) is insufficient for MLP to learn effective non-linear boundaries. Linear or near-linear boundaries (SVM with RBF) are more appropriate for this data.

---

## 6.6 Unexpected Findings

### 6.6.1 Interval Features Dominate

**Expectation**: Based on musicological literature, we anticipated tonal tension and harmonic complexity features would be most discriminative.

**Result**: Interval features (unison ratio, stepwise ratio) and texture features (pt_std) dominated the importance rankings.

**Interpretation**: For same-era composer classification, surface-level statistical regularities (melodic motion patterns, textural variation) may carry more stylistic information than deeper tonal properties. This echoes Youngblood's (1958) foundational insight about statistical distributions capturing stylistic choice.

### 6.6.2 SVM-SVM Consistency

**Observation**: SVM performance is consistent across different feature sets (12D: 49.3%, 54D: ~63-67%, 768D: 47.1%).

**Interpretation**: SVM is a robust classifier that performs proportionally to feature quality, making it suitable for comparative studies.

---

## 6.7 Limitations of Exploratory Analysis

The analyses in this section should be interpreted with caution:

1. **Post-hoc nature**: These investigations were not pre-registered hypotheses but observations made after examining results
2. **Multiple comparisons**: Examining many feature combinations increases false positive risk
3. **Small sample sizes**: Cycle-level analyses involve 12-24 pieces, limiting statistical power
4. **Confirmation bias risk**: Interpretations may be influenced by pre-existing musicological beliefs

Future work should test these exploratory findings on independent datasets.

---

## Exploratory Analysis Writing Notes

### Key Updates from Previous Version:
- Removed note count from feature discussions
- Updated top feature interpretations (unison ratio, pt_std, pitch_std)
- Added combined feature analysis section
- Updated "What Didn't Work" with note count exclusion rationale

### Transparency:
- Clearly distinguish exploratory from confirmatory analysis
- Acknowledge limitations and potential biases
- Report negative results (MLP underperformance, note count confounding)

### Value Proposition:
- Exploratory findings generate hypotheses for future research
- Case studies demonstrate practical application of features
- Lessons learned guide methodological choices

### Length Management:
- Current draft: ~1.5 pages
- May need to condense for final submission
- Consider moving "What Didn't Work" to supplementary material

---

## Revision Checklist

- [x] Remove note count from feature discussions
- [x] Update top feature interpretations (unison ratio, pt_std, pitch_std)
- [x] Add combined feature analysis section
- [x] Add note count exclusion to "What Didn't Work"
- [ ] Ensure case study interpretations are musicologically sound
- [ ] Check that limitations are clearly stated
- [ ] Review for appropriate hedging language ("suggests," "may indicate")
- [ ] Confirm exploratory nature is emphasized throughout
- [ ] Verify word count fits within page limits

---

## Next Steps

1. Cross-reference discussion claims with results section data
2. Prepare musical examples for potential figures
3. Ensure references section includes all sources cited
4. Update LaTeX paper with new content
