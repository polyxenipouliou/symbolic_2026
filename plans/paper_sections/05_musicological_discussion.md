# 5. Musicological Discussion

**Target Length**: 1 page  
**Format**: Two-column, 10pt Times

---

## 5.1 Interpreting Computational Findings

The results presented in Section 4 raise a fundamental question: what do these computational features tell us about the musical style of Schubert, Schumann, and Brahms? This section bridges the gap between statistical patterns and musicological understanding, offering interpretive hypotheses grounded in both domains.

**Important Note**: Note count (piece length proxy) and velocity features were excluded from this analysis to avoid confounding variables and editorial bias. All interpretations are based on interval, texture, pitch, and rhythm features only.

---

## 5.2 What Features Reveal About Schubert

### 5.2.1 High Stepwise Ratio (f28_stepwise_ratio)

**Computational Finding**: Schubert Lieder show higher proportion of stepwise melodic motion (intervals of 1-2 semitones).

**Musicological Interpretation**: This finding aligns with longstanding observations about Schubert's vocal writing. His melodies are renowned for their:

- **Lyrical quality**: Stepwise motion creates singable, memorable lines that mirror natural speech inflection
- **Folk song influence**: Many Schubert melodies evoke *Volkslied* simplicity through conjunct motion
- **Text sensitivity**: Stepwise writing allows clear text declamation without wide leaps disrupting syllable setting

**Example**: The opening of "Gute Nacht" from *Winterreise* moves primarily by step (D-E-F#-E-D), creating the weary walker's plodding motion.

### 5.2.2 Lower Texture Density

**Computational Finding**: Schubert shows lower values for simultaneity features.

**Musicological Interpretation**: This reflects Schubert's characteristic accompaniment patterns:

- **Guitar-like arpeggiation**: Broken chord patterns (as in "Die Forelle") create harmonic support without dense vertical sonorities
- **Bass-melody texture**: Many accompaniments feature single-line bass with sparse inner voices
- **Textural restraint**: Even in dramatic songs, Schubert often maintains transparent textures

**Example**: "Der Lindenbaum" uses flowing triplet arpeggios throughout, rarely employing full chordal writing.

---

## 5.3 What Features Reveal About Schumann

### 5.3.1 High Texture Variation (pt_std)

**Computational Finding**: Schumann Lieder show highest texture standard deviation.

**Musicological Interpretation**: This finding captures Schumann's diverse accompaniment styles:

- **Arpeggiated textures**: Flowing patterns (as in *Dichterliebe*'s opening) create low simultaneity
- **Contrasting sections**: Within-song textural changes produce high variance
- **Piano independence**: Elaborate postludes and interludes vary texture beyond vocal sections

**Example**: The piano postlude of "Ich groll' nicht" transforms from chordal declaration to arpeggiated dissolution.

### 5.3.2 High Staccato Ratio (f22_staccato_ratio)

**Computational Finding**: Schumann shows higher proportion of short notes.

**Musicological Interpretation**: This captures Schumann's expressive articulation:

- **Poetic declamation**: Short notes reflect speech-like rhythmic flexibility
- **Character pieces**: Schumann's background in piano character pieces influences his song writing
- **Emotional volatility**: Articulation changes mirror mood shifts in the poetry

---

## 5.4 What Features Reveal About Brahms

### 5.4.1 High Pitch Range (f4_pitch_range) and Pitch Standard Deviation (f3_pitch_std)

**Computational Finding**: Brahms Lieder employ wider pitch spans and more varied pitch distributions.

**Musicological Interpretation**: These features reflect Brahms' rich musical language:

- **Symphonic piano writing**: Dense textures evoke orchestral sonorities
- **Wide register exploitation**: Piano parts span full keyboard range
- **Continuous motion**: Inner voices maintain rhythmic activity throughout

**Example**: "Wie Melodien zieht es mir" features rich piano texture with continuous sixteenth-note motion.

### 5.4.2 High Unison Ratio (f27_unison_ratio)

**Computational Finding**: Brahms shows higher proportion of repeated notes.

**Musicological Interpretation**: This supports the characterization of Brahms as classically restrained:

- **Folk song influence**: *Volkslied*-inspired melodies often use repeated notes
- **Classical balance**: Repetition creates structural clarity
- **Vocal writing**: Brahms' understanding of vocal limits constrains melodic range

---

## 5.5 Why Handcrafted Features Outperform MidiBERT

### 5.5.1 Domain Specificity

**Finding**: 54D handcrafted features (~67%) significantly outperform 768D MidiBERT embeddings with both SVM (47.1%) and MLP (45.3%).

**Interpretation**: This result illuminates the tension between general and specialized representations:

- **Task mismatch**: MidiBERT trained on diverse MIDI data (popular music, piano pieces) may not capture Lieder-specific features
- **Feature relevance**: Handcrafted features encode musicological hypotheses about what distinguishes composers
- **Signal-to-noise**: 768 dimensions include many features irrelevant to composer style, diluting discriminative signal

### 5.5.2 The Small Corpus Problem

**Finding**: Pretrained model underperforms despite theoretical advantages.

**Interpretation**: The 264-piece corpus is insufficient for effective fine-tuning:

- **Parameter ratio**: 768-dimensional embeddings require thousands of samples for reliable classification
- **Overfitting risk**: High-dimensional space allows model to memorize training data rather than learn generalizable patterns
- **Domain shift**: Pretraining on different musical genres creates representation gap

### 5.5.3 Methodological Rigor: Feature Exclusion

**Finding**: Model achieves ~67% accuracy without note count and velocity features.

**Interpretation**: This validates our methodological decisions:

- **Genuine stylistic markers exist** in interval, texture, pitch, and rhythm dimensions
- **Confounding variables avoided**: Classification based on musical patterns, not piece length or editorial conventions
- **Interpretability maintained**: All features have clear musicological meaning

---

## 5.6 Interval Features as Primary Discriminators

### 5.6.1 Unison Ratio (f27) Dominance

**Finding**: f27_unison_ratio is the most important feature (importance = 0.0308) after note count removal.

**Interpretation**: Repeated note patterns carry significant stylistic information:

- **Brahms**: Higher unison ratio reflects folk song influence and classical restraint
- **Schubert**: Moderate unison ratio balanced with stepwise motion
- **Schumann**: Lower unison ratio, more varied melodic motion

### 5.6.2 Stepwise Ratio (f28) Significance

**Finding**: f28_stepwise_ratio ranks 4th in importance (0.0280).

**Interpretation**: Conjunct melodic motion distinguishes composers:

- **Schubert**: Higher stepwise ratio reflects lyrical, vocal writing
- **Schumann/Brahms**: More varied interval patterns

---

## 5.7 Limitations of Current Approach

### 5.7.1 Binary Texture Model

Our onset density measure captures vertical thickness but not:

- **Voice-leading quality**: How individual voices move independently
- **Articulation patterns**: Staccato vs. legato affects perceived texture
- **Pedaling**: Piano pedaling creates sustained sonorities not captured in symbolic data

### 5.7.2 Absence of Chord Function Analysis

We measure harmonic complexity but not:

- **Functional progression**: Tonic-dominant relationships
- **Modulation patterns**: Key change strategies
- **Chord annotation**: Requires manual or automated harmonic analysis

### 5.7.3 Meter-Level Aggregation

Aggregating to piece-level statistics loses:

- **Local detail**: Moment-to-moment variation in features
- **Structural patterns**: Section-level organization (verse vs. chorus)
- **Temporal evolution**: How features change throughout the piece

---

## 5.8 Synthesis: Computational Musicology as Hypothesis Generator

The value of this computational approach lies not in definitive attribution but in hypothesis generation:

1. **Unison ratio is most discriminative** → Further study of repeated note patterns across composers
2. **Texture variation characterizes Schumann** → Detailed analysis of accompaniment pattern taxonomy
3. **Interval features dominate** → Re-evaluation of melodic style classification approaches

These hypotheses can guide future musicological research, creating a productive dialogue between computational and traditional approaches.

---

## Musicological Discussion Writing Notes

### Key Updates from Previous Version:
- Removed note count discussions
- Updated top feature interpretations (unison ratio, pt_std, pitch_std)
- Added section on interval features as primary discriminators
- Updated methodological rigor discussion

### Balance:
- Equal treatment of all three composers
- Connection between computational findings and musicological literature
- Acknowledgment of interpretive limitations

### Evidence Quality:
- Specific musical examples cited where possible
- Claims grounded in established musicological understanding
- Clear distinction between observation and interpretation

### Critical Perspective:
- Limitations honestly acknowledged
- Alternative interpretations considered
- Avoidance of computational determinism

### Length Management:
- Current draft: ~2 pages
- May need to condense composer-specific sections
- Consider moving detailed examples to appendix

---

## Revision Checklist

- [x] Remove note count discussions
- [x] Update feature interpretations (unison ratio, pt_std, pitch_std)
- [x] Add interval features section
- [ ] Verify musicological claims against scholarly literature
- [ ] Ensure musical examples are accurate and representative
- [ ] Check that computational findings are correctly interpreted
- [ ] Review for balance across composers
- [ ] Confirm limitations section is comprehensive
- [ ] Verify word count fits within page limits

---

## Next Steps

1. Complete exploratory analysis section with case studies
2. Cross-reference discussion claims with results section data
3. Prepare musical examples for potential figures
4. Ensure references section includes all musicological sources cited
