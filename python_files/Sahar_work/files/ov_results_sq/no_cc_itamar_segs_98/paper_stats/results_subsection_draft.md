## Results: Inter-Physician Observer Variability (98 Pairs)

Observer variability was evaluated across five physicians on 98 longitudinal CXR pairs using human-human (H-H) agreement metrics for positive, negative, and combined labels. Pairwise agreement among physicians was moderate to high overall, with the highest agreement observed when combining positive and negative labels.

For H-H per-pair agreement, the mean (±SD) was 0.5566 ± 0.0714 for positive labels, 0.5969 ± 0.0726 for negative labels, and 0.6149 ± 0.0597 for all labels combined. The corresponding 95% bootstrap confidence intervals for the mean were [0.5108, 0.5986], [0.5525, 0.6416], and [0.5778, 0.6517], respectively. Agreement ranges were 0.4374-0.6367 (positive), 0.4983-0.7065 (negative), and 0.5250-0.6998 (all), indicating measurable heterogeneity across physician pairs.

For H-H per-detection agreement, the mean (±SD) was 0.5755 ± 0.0557 for positive labels, 0.5108 ± 0.0719 for negative labels, and 0.5477 ± 0.0584 for all labels. The corresponding 95% bootstrap confidence intervals were [0.5405, 0.6096], [0.4679, 0.5562], and [0.5117, 0.5839], respectively.

Paired inferential testing between positive and negative agreement showed a significant difference for both analysis levels. For per-pair agreement, positive-minus-negative mean difference was -0.0403 (exact sign-flip p=0.0273), indicating higher negative agreement. For per-detection agreement, positive-minus-negative mean difference was +0.0647 (p=0.0039), indicating higher positive agreement at the detection level.

Annotation burden varied across physicians. Mean labels per pair (all labels) were: Avi 1.4898, Benny 1.8673, Sigal 1.4490, Smadar 1.6633, and Nitzan 1.9592. This spread suggests physician-specific labeling thresholds and likely contributes to pairwise variability.

Consensus sensitivity increased with stricter consensus thresholds for positive labels (0.4144 at level 1 to 0.8889 at level 5). For negative labels, sensitivity increased from 0.4676 (level 1) to 0.7742 (level 4), with a slight decrease at level 5 (0.7692). These trends indicate improved physician-consensus alignment at higher agreement levels.

### Suggested In-Text Figure/Table Callouts

- Pairwise and per-detection summary statistics: Table 1 ([agreement_summary_human_human.csv](tables/agreement_summary_human_human.csv))
- Physician annotation burden: Table 2 ([annotation_burden_by_physician.csv](tables/annotation_burden_by_physician.csv))
- Consensus sensitivity by level: Table 3 ([consensus_sensitivity_summary.csv](tables/consensus_sensitivity_summary.csv))
- Positive vs negative inferential comparisons: Table 4 ([positive_vs_negative_tests.csv](tables/positive_vs_negative_tests.csv))
- H-H agreement heatmaps (pair-level): Figure 1A-C ([heatmap_per_pair_positive_hh.png](plots/heatmap_per_pair_positive_hh.png), [heatmap_per_pair_negative_hh.png](plots/heatmap_per_pair_negative_hh.png), [heatmap_per_pair_all_hh.png](plots/heatmap_per_pair_all_hh.png))
- H-H agreement heatmaps (detection-level): Figure 2A-C ([heatmap_per_detection_positive_hh.png](plots/heatmap_per_detection_positive_hh.png), [heatmap_per_detection_negative_hh.png](plots/heatmap_per_detection_negative_hh.png), [heatmap_per_detection_all_hh.png](plots/heatmap_per_detection_all_hh.png))
- Distribution plots: Figure 3A-B ([boxplot_pairwise_hh.png](plots/boxplot_pairwise_hh.png), [boxplot_per_detection_hh.png](plots/boxplot_per_detection_hh.png))
- Consensus trend plot: Figure 4 ([consensus_sensitivity_positive_negative.png](plots/consensus_sensitivity_positive_negative.png))
- Annotation burden bar plot: Figure 5 ([annotation_burden_by_physician.png](plots/annotation_burden_by_physician.png))

### Reporting Note

Panel-level multi-rater reliability metrics (for example, Fleiss' kappa or Krippendorff's alpha) are not recoverable exactly from these aggregate outputs alone and require case-level rater assignment data.

## Equations Used for Each Table and Plot

Let $x_1,\dots,x_N$ denote the human-human agreement values for one metric/condition (here $N=10$ physician pairs), and let $L=98$ denote number of longitudinal pairs.

### Table 1: agreement_summary_human_human.csv

For each row (pairwise/per-detection; positive/negative/all), the following were computed:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

$$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i-\mu)^2}
$$

$$
	ext{median} = Q_{0.5}(x),\quad Q_1 = Q_{0.25}(x),\quad Q_3 = Q_{0.75}(x),\quad \text{IQR}=Q_3-Q_1
$$

$$
	ext{range}=\max(x)-\min(x)
$$

95% bootstrap CI for the mean (resampling the $N$ values with replacement):

$$
	ext{CI}_{95\%}(\mu)=\left[Q_{0.025}(\mu^{\ast}),\;Q_{0.975}(\mu^{\ast})\right]
$$

where $\mu^{\ast}$ are bootstrap sample means.

### Table 2: annotation_burden_by_physician.csv

For physician $r$ and label type $t\in\{\text{pos},\text{neg},\text{all}\}$, with per-case counts $c_{r,t,1},\dots,c_{r,t,L}$:

$$
	ext{Total}_{r,t}=\sum_{\ell=1}^{L} c_{r,t,\ell},\quad
	ext{Mean}_{r,t}=\frac{1}{L}\sum_{\ell=1}^{L} c_{r,t,\ell},\quad
	ext{SD}_{r,t}=\sqrt{\frac{1}{L}\sum_{\ell=1}^{L}(c_{r,t,\ell}-\text{Mean}_{r,t})^2}
$$

$$
	ext{Min}_{r,t}=\min_{\ell} c_{r,t,\ell},\quad
	ext{Max}_{r,t}=\max_{\ell} c_{r,t,\ell}
$$

### Table 3: consensus_sensitivity_summary.csv

For consensus level $k$ and class $t\in\{\text{pos},\text{neg}\}$:

$$
	ext{Sensitivity}_{k,t}=\frac{\text{Detected}_{k,t}}{\text{Total}_{k,t}}
$$

where Detected and Total are the numerator/denominator reported from the OV aggregation.

### Table 4: positive_vs_negative_tests.csv

For paired vectors $x_i$ (positive) and $y_i$ (negative):

$$
\Delta = \frac{1}{N}\sum_{i=1}^{N}(x_i-y_i)
$$

Exact sign-flip two-sided p-value:

$$
p = \frac{1}{2^N}\sum_{s\in\{-1,+1\}^N} \mathbf{1}\!\left(\left|\frac{1}{N}\sum_{i=1}^{N} s_i(x_i-y_i)\right|\ge |\Delta|\right)
$$

### Table 5: raw_hh_pair_values.csv

This table is the raw $x_i$ values used above (no further transformation):

$$
\{x_i\}_{i=1}^{N}\text{ for each condition/metric}
$$

### Figure 1A-C and Figure 2A-C: heatmaps

Heatmap entries are agreement matrices $A\in\mathbb{R}^{5\times 5}$ (human-human block):

$$
A_{ij}=\text{agreement score between physician }i\text{ and }j
$$

Diagonal entries satisfy:

$$
A_{ii}=1
$$

Color encodes matrix value $A_{ij}$ in $[0,1]$.

### Figure 3A-B: boxplots

Each boxplot summarizes the same distribution used in Table 1:

$$
\{x_i\}_{i=1}^{N}=\{A_{ij}:1\le i<j\le5\}
$$

with median line, box spanning $[Q_1,Q_3]$, and whiskers from the plotted boxplot rule.

### Figure 4: consensus_sensitivity_positive_negative.png

Curve points at level $k$ are:

$$
\left(k,\;\text{Sensitivity}_{k,\text{pos}}\right),\qquad
\left(k,\;\text{Sensitivity}_{k,\text{neg}}\right)
$$

using the sensitivity formula from Table 3.

### Figure 5: annotation_burden_by_physician.png

Bar heights are physician-wise means from Table 2:

$$
h_{r,t}=\text{Mean}_{r,t}=\frac{1}{L}\sum_{\ell=1}^{L} c_{r,t,\ell}
$$

for $t\in\{\text{pos},\text{neg},\text{all}\}$.