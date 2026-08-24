<!--
DRAFT MANUSCRIPT — Journal of Thoracic Imaging
Style modeled on Joskowicz group clinical OV papers:
  - Olesinski A, Lederman R, Azraq Y, Joskowicz L, Sosna J. "Variability in Mediastinal
    Lymph Node Measurements in Chest CECT: Time to Change the Paradigm?" J Thorac Imaging 2026;41(2):e0859.
  - Joskowicz L, Cohen D, Caplan N, Sosna J. "Inter-observer variability of manual contour
    delineation of structures in CT." Eur Radiol 2019;29(3):1391-1399.

Reader anonymization (internal — remove before submission):
  R1=Avi, R2=Benny, R3=Sigal, R4=Smadar, R5=Nitzan
All numbers below are taken from the 100-pair observer-variability run
(ov_results_main_loo_itamar_plus_lmm5 + ov_results_sq/no_cc_itamar_plus_lmm5_100).
[BRACKETS] mark clinical details still to be filled in by the authors.
-->

# Inter-observer Variability in the Detection of Interval Change on Longitudinal Chest Radiographs: How Many Readers Define the Truth?

**Authors:** [Author list] · **Affiliations:** [School of Computer Science and Engineering, The Hebrew University of Jerusalem; Department of Radiology, Hadassah Hebrew University Medical Center, Jerusalem, Israel]

**Corresponding author:** Prof. Leo Joskowicz — josko@cs.huji.ac.il

---

## Structured Abstract

**Purpose:** Comparison of a current chest radiograph with a prior study to identify interval change is one of the most frequent tasks in thoracic imaging, yet the reliability of this judgment across readers is poorly quantified. We aimed to quantify the inter-observer agreement and variability of radiologists in detecting and characterizing interval change on longitudinal chest radiograph (CXR) pairs, and to determine how many readers are required to establish a stable reference standard.

**Materials and Methods:** Five [board-certified] radiologists independently reviewed 100 longitudinal CXR pairs (a prior and a current image from the same patient) [ICU/inpatient cohort; retrospective; IRB approval and waiver of consent obtained]. For each pair, readers marked every region of interval change and classified it as *appearance* (new finding), *disappearance* (resolved finding), or *persistence* with a change in size and/or intensity (increase/decrease). Findings were grouped into positive change (new or worsening) and negative change (resolved or improving). A region marked by at least *k* readers was assigned consensus level *k*. Agreement was quantified with a Pairwise Agreement Index (PAI) at the level of individual detections and whole pairs, the Human-Matched Detection Rate (HMDR), the number of Unmatched Detections Per Pair (UDPP), and the per-reader leave-one-out sensitivity as a function of consensus level.

**Results:** Readers marked a mean of 1.68 change findings per pair (range across readers, 1.45–1.97; 145–197 findings each). The mean pairwise agreement between two readers was 0.55 at the detection level (range 0.46–0.64) and 0.61 at the pair level (range 0.52–0.70); agreement was higher for positive than for negative change (0.57 vs 0.51). Of 185 distinct positive change findings, 45% (83/185) were reported by a single reader and only 15% (27/185) by all five; for negative change, 40% (55/139) were solo and 9% (13/139) unanimous. Per-reader leave-one-out sensitivity against the remaining panel rose monotonically with consensus level, from 0.46 at consensus level 1 to 0.89 at level 4 for positive change (0.46 to 0.85 for negative). Between 3% and 28% of each reader's findings were unmatched by any peer.

**Conclusions:** Radiologist agreement on interval change on chest radiographs is moderate and is dominated by low-consensus findings: nearly half of all identified changes were seen by only one reader. Agreement rises steeply with consensus, so two or three readers capture only a fraction of the clinically reported change. A reference standard for interval change—and for the evaluation of automated change-detection algorithms—should be built from a multi-reader consensus rather than from any single reader.

**Key Points**
- Inter-observer agreement for interval change on chest radiographs is moderate (mean pairwise detection agreement 0.55) and highest for new/worsening findings.
- Almost half of all change findings (45% positive, 40% negative) were reported by only one of five readers; fewer than one in six were unanimous.
- Reader sensitivity against the panel nearly doubles from single-reader to four-reader consensus, indicating that two or three readers are insufficient to define the full range of reported change.

---

## 1. Introduction

Serial comparison of chest radiographs is among the highest-volume interpretive tasks in radiology, particularly in the intensive-care and inpatient settings, where daily films are used to track lines and tubes, effusions, consolidation, edema, and pneumothorax. Clinical decisions—escalation of care, drainage, antibiotic changes, device repositioning—frequently turn on whether a finding is *new*, *resolved*, *larger*, or *smaller* relative to the prior study.

Despite the ubiquity of this task, the reliability of interval-change judgments across radiologists has received far less attention than the reliability of single-image detection or of quantitative measurement. Prior work from our group and others has shown that inter-observer variability in manual delineation and measurement is large and that two or three observers may not be sufficient to capture its full range [ref: Joskowicz 2019 Eur Radiol; Olesinski 2026 J Thorac Imaging]. Whether the same holds for the fundamentally comparative, detection-level task of interval change on radiographs is unknown, yet it directly determines what should count as "ground truth" when training and evaluating automated longitudinal analysis tools.

In this study we quantify the inter-observer agreement and variability of five radiologists detecting and characterizing interval change on 100 longitudinal CXR pairs. We introduce a consensus-level framework in which every change region is labeled by the number of readers who identified it, and we use group-wise agreement metrics to answer a practical question: *how many readers are needed before the reference standard stabilizes?*

## 2. Materials and Methods

### 2.1 Study population
[Retrospective] analysis of 100 longitudinal chest radiograph pairs from [N patients] imaged at [institution] between [dates]. Each pair comprised a prior and a current [AP/portable] radiograph from the same patient [selection criteria: consecutive ICU studies with an interpretable prior within X days]. [IRB approval number; informed consent waived.] Patient demographics are summarized in [Table S1].

### 2.2 Readers and reading protocol
Five radiologists ([R1–R5]; [years of experience: …]) independently reviewed all 100 pairs on a dedicated annotation workstation, blinded to one another and to any clinical information beyond the two images. For each pair, a reader delineated every region of interval change with an elliptical marker and assigned one of the following labels:

- **Appearance** — a finding present on the current image but not the prior (new).
- **Disappearance** — a finding present on the prior but not the current (resolved).
- **Persistence** — a finding present on both, with an annotated change in **size** (increase/decrease/none) and **intensity** (increase/decrease/none).

For analysis, each marked region was reduced to a signed change: **positive change** (appearance, or persistence with increased size/intensity) or **negative change** (disappearance, or persistence with decreased size/intensity). Persistence without change was excluded.

### 2.3 Reference standard and consensus levels
Reader annotations were rasterized to the coordinate space of the current image and grouped into connected components (findings). For each finding we computed its **consensus level**—the number of readers (1–5) who independently marked an overlapping region. A finding at consensus level *k* was identified by at least *k* readers. This yields a graded reference standard: consensus level 1 is the union of all reader findings, and consensus level 5 comprises only unanimous findings.

### 2.4 Agreement and variability metrics
All metrics were computed separately for positive and negative change.

- **Pairwise Agreement Index (PAI).** For each pair of readers, the per-detection PAI is 2·M ⁄ (2·M + U), where M is the number of matched (spatially overlapping) findings and U the number of unmatched findings—an F1-/Dice-like agreement on detections. The per-pair PAI averages agreement over the 100 image pairs, so that pairs with no findings by either reader count as full agreement.
- **Human-Matched Detection Rate (HMDR).** The fraction of a reader's findings that overlap a finding of at least one other reader (a precision-like measure of how "corroborated" a reader's marks are).
- **Unmatched Detections Per Pair (UDPP).** The mean number of a reader's findings, per image pair, not matched by any other reader (solo findings).
- **Sensitivity at consensus level (leave-one-out).** For each reader, the fraction of the consensus findings of the *other four* readers, at each consensus level, that the reader also marked. Because the reader being scored is excluded from the reference, this is an unbiased, symmetric measure of how well each reader recovers what the rest of the panel agreed upon.
- **Consensus-based specificity.** Among image pairs in which the remaining panel identified no change of a given sign, the fraction in which the reader likewise marked none.

### 2.5 Statistical analysis
Agreement metrics are reported as means with ranges across reader pairs. Sensitivity is reported as a function of consensus level with per-reader curves. [Bland–Altman / 95% CIs / bootstrap resampling over pairs to be added.] Analyses were performed in Python [3.x] with NumPy, SciPy, and scikit-image.

## 3. Results

### 3.1 Finding burden
Across 100 pairs the five readers marked a mean of 1.68 change findings per pair. Per-reader totals ranged from 145 to 197 findings (mean 1.45–1.97 per pair; maximum 4–6 findings on a single pair), a 1.36-fold difference in reporting rate between the least and most prolific reader (Table 1).

**Table 1. Change-finding burden per reader (100 pairs).**

| Reader | Total findings | Mean per pair | SD | Max per pair |
|--------|---------------:|--------------:|----:|-------------:|
| R1 | 148 | 1.48 | 0.77 | 4 |
| R2 | 185 | 1.85 | 1.04 | 6 |
| R3 | 145 | 1.45 | 0.80 | 4 |
| R4 | 166 | 1.66 | 0.95 | 5 |
| R5 | 197 | 1.97 | 0.84 | 4 |

### 3.2 Pairwise agreement
The mean per-detection PAI between two readers was 0.55 (range 0.46–0.64); the mean per-pair PAI was 0.61 (range 0.52–0.70). The highest-agreeing reader pair still concurred on only ~64% of individual detections, and the lowest on ~46% (Table 2, Fig 2). Agreement was consistently higher for positive change (mean per-detection PAI 0.57) than for negative change (0.51), indicating that new or worsening findings are marked more reproducibly than resolving ones.

**Table 2. Pairwise Agreement Index (per detection, positive + negative combined).**

| | R1 | R2 | R3 | R4 | R5 |
|----|----|----|----|----|----|
| R1 | — | 0.47 | 0.59 | 0.51 | 0.59 |
| R2 | 0.47 | — | 0.50 | 0.46 | 0.50 |
| R3 | 0.59 | 0.50 | — | 0.57 | 0.63 |
| R4 | 0.51 | 0.46 | 0.57 | — | 0.64 |
| R5 | 0.59 | 0.50 | 0.63 | 0.64 | — |

### 3.3 Consensus structure and disagreement
Disagreement was dominated by low-consensus findings. Of 185 distinct positive change findings (consensus level ≥1), 101 (55%) reached consensus level 2, 76 (41%) level 3, 52 (28%) level 4, and only 27 (15%) were unanimous; conversely, 45% (83/185) were **solo** findings marked by a single reader. Negative change followed the same pattern: of 139 findings, 84 (60%) reached level 2 and only 13 (9%) were unanimous, with 40% (55/139) solo (Fig 4).

Consistent with this, between 3% and 28% of each reader's findings were unmatched by any peer (HMDR 0.72–0.97 for positive change), and readers contributed a mean of 0.02–0.30 solo findings per pair (UDPP; Table 4). Two readers accounted for most solo findings, while one reader's marks were almost always corroborated (HMDR 0.97).

**Table 4. Corroboration of each reader's findings (positive change).**

| Reader | HMDR (positive) | HMDR (negative) | UDPP (positive) | UDPP (negative) |
|--------|----------------:|----------------:|----------------:|----------------:|
| R1 | 0.83 | 0.85 | 0.15 | 0.09 |
| R2 | 0.72 | 0.78 | 0.30 | 0.17 |
| R3 | 0.97 | 0.88 | 0.02 | 0.08 |
| R4 | 0.89 | 0.88 | 0.11 | 0.08 |
| R5 | 0.77 | 0.83 | 0.26 | 0.15 |

### 3.4 Agreement rises with consensus
Per-reader leave-one-out sensitivity against the remaining panel increased monotonically with consensus level (Fig 3, Table 3). Averaged across readers, sensitivity for positive change rose from 0.46 at consensus level 1 to 0.72 (level 2), 0.80 (level 3), and 0.89 (level 4); for negative change it rose from 0.46 to 0.62, 0.82, and 0.85. In other words, when only one other reader had seen a finding, a given radiologist reproduced it less than half the time; when all four others agreed, they reproduced it in ~85–90% of cases.

**Table 3. Mean per-reader leave-one-out sensitivity by consensus level.**

| Consensus level | Positive | Negative |
|----------------:|---------:|---------:|
| ≥1 | 0.46 | 0.46 |
| ≥2 | 0.72 | 0.62 |
| ≥3 | 0.80 | 0.82 |
| ≥4 | 0.89 | 0.85 |

Consensus-based specificity was high for most readers (several readers marked no change on essentially all pairs the panel deemed unchanged), although the number of change-free pairs was small and this estimate should be interpreted with caution.

## 4. Discussion

In a multi-reader study of interval change on 100 longitudinal chest radiographs, we found that radiologist agreement is moderate and is dominated by low-consensus findings. Nearly half of all change findings were reported by a single reader, and fewer than one in six were unanimous. Agreement climbed steeply with consensus level—reader sensitivity against the panel nearly doubled from single-reader to four-reader consensus—demonstrating that the "truth" of interval change is graded rather than binary.

These results extend to the comparative, detection-level task of interval change the central message of prior inter-observer studies in CT: that two or even three observers may not be sufficient to establish the full range of inter-observer variability [Joskowicz 2019]. They also parallel the recent finding that, for mediastinal lymph-node assessment, disagreements are systematically larger than agreements and warrant a rethinking of dichotomous thresholds [Olesinski 2026]. Here, the analog of that message is that a single-reader reference standard captures only a fraction of what a panel of radiologists collectively identifies as change.

The clinical implications are twofold. First, for **reader practice and quality assurance**, the predominance of solo findings suggests that a substantial share of reported interval change is not reproducibly perceived, and that consensus review (or over-reading) is likely to change the reported picture in a non-trivial fraction of cases. Second, for **automated longitudinal analysis**, our findings argue strongly against evaluating change-detection algorithms against any single reader: an algorithm judged against one radiologist would be penalized for missing that reader's solo findings and rewarded or penalized inconsistently across readers. A graded, consensus-level reference standard—reporting algorithm sensitivity separately at each consensus level—provides a fairer and more informative benchmark. [Optional: In our data, an automated change-detection method reached a panel agreement (per-detection PAI 0.46; leave-one-out sensitivity within the human range at every consensus level) comparable to the lower end of inter-reader agreement, supporting this framing; full model results are reported separately.]

**Limitations.** This was a [single-center, retrospective] study of 100 pairs read by five radiologists; the number of change-free pairs limited the precision of specificity estimates. Reader experience was [not stratified/…], and annotations used elliptical region markers rather than pixel-wise contours, which may merge or split adjacent findings. The cohort was enriched for [ICU studies], which may over-represent devices and rapidly evolving findings and limit generalizability to outpatient chest radiography. Finally, the reference standard is itself defined by the readers; consensus level is a pragmatic surrogate for truth, not an independent gold standard.

## 5. Conclusions

Radiologist agreement on interval change on chest radiographs is moderate and graded: nearly half of all identified changes were seen by only one of five readers, and agreement rose sharply with the number of concurring readers. Two or three readers are insufficient to define the full range of reported change. Reference standards for interval change—and the evaluation of automated change-detection tools—should be built from multi-reader consensus and reported as a function of consensus level.

---

## Figure legends

**Figure 1.** Example longitudinal chest radiograph pair (prior, current) with the change annotations of the five readers overlaid; green = positive (new/worsening) change, red = negative (resolved/improving) change. Note both a unanimously marked finding and one or more solo findings.

**Figure 2.** Pairwise Agreement Index heatmaps among the five readers (and an automated method, optional), per detection and per pair, for positive, negative, and combined change.

**Figure 3.** Per-reader leave-one-out sensitivity as a function of consensus level (1–4), for positive and negative change, showing the monotonic rise in agreement with consensus.

**Figure 4.** Distribution of change findings by consensus level (fraction of findings marked by exactly 1, 2, 3, 4, and 5 readers), illustrating the predominance of low-consensus (solo) findings.

## Tables
Table 1 — finding burden per reader. Table 2 — PAI matrix (per detection). Table 3 — mean sensitivity by consensus level. Table 4 — HMDR/UDPP per reader. [Supplementary: per-reader positive/negative split; per-pair PAI matrix; consensus-count table.]

---

*Draft prepared for internal review. Source metrics: `python_files/Sahar_work/files/ov_results_main_loo_itamar_plus_lmm5/leave_one_out_metrics.json` and `python_files/Sahar_work/files/ov_results_sq/no_cc_itamar_plus_lmm5_100/`.*
