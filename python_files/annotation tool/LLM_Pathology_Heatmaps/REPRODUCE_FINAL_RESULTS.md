# Reproducing the final “Why not use an LLM?” results

This compact archive keeps the raw LLM responses, prompts, run manifests,
final metric tables, coordinate audits, slide-used collages, canonical
presentation source, and latest presentation. Large rendered NIfTI/PNG outputs
and full collage sets were removed because they are deterministic products of
the retained responses and code.

Run commands from the repository root with the repository virtual environment.

## 1. Render the retained Claude responses

```powershell
$harness = '.\python_files\annotation tool\llm_semantic_diff_experiment.py'
$root = '.\python_files\annotation tool\LLM_Pathology_Heatmaps\claude_sonnet_5_full_experiment'

& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm ellipses --responses-root "$root\responses_ellipses" `
  --annotator-name Claude_Sonnet_5
& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm heatmap --responses-root "$root\responses_heatmap" `
  --annotator-name Claude_Sonnet_5
& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm heatmap --rendered-name heatmap_precise `
  --responses-root "$root\responses_heatmap_precise" `
  --annotator-name Claude_Sonnet_5
```

## 2. Render the retained GPT-5.4 responses

```powershell
$harness = '.\python_files\annotation tool\llm_semantic_diff_experiment.py'
$root = '.\python_files\annotation tool\LLM_Pathology_Heatmaps\gpt_5_4_full_experiment'

& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm ellipses --responses-root "$root\responses_ellipses" `
  --annotator-name GPT_5_4
& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm heatmap --responses-root "$root\responses_heatmap" `
  --annotator-name GPT_5_4
& '.\.venv\Scripts\python.exe' $harness --output-root $root render `
  --arm heatmap --rendered-name heatmap_precise `
  --responses-root "$root\responses_heatmap_precise" `
  --annotator-name GPT_5_4
```

## 3. Recompute observer metrics and coordinate audits

```powershell
$claude = '.\python_files\annotation tool\LLM_Pathology_Heatmaps\claude_sonnet_5_full_experiment'
$gpt = '.\python_files\annotation tool\LLM_Pathology_Heatmaps\gpt_5_4_full_experiment'

& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\compare_llm_annotator_experiment.py' `
  --experiment-root $claude --out-dir "$claude\comparison" `
  --annotator-label Claude
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\compare_llm_annotator_experiment.py' `
  --experiment-root $gpt --out-dir "$gpt\comparison" `
  --annotator-label GPT-5.4

& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\make_llm_experiment_collages.py' `
  --experiment-root $claude --output-root "$claude\collages_precision" `
  --annotator-label Claude --annotator-file-name Claude_Sonnet_5
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\make_llm_experiment_collages.py' `
  --experiment-root $gpt --output-root "$gpt\collages_precision" `
  --annotator-label GPT-5.4 --annotator-file-name GPT_5_4
```

The collage commands recreate all 100 collages. Only the slide-used examples
need to be retained after verification.

## 4. Recompute Claude/GPT comparison and mask-size statistics

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\compare_claude_gpt_experiments.py'
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\compute_llm_segmentation_size_stats.py'
```

The second command reproduces:

- `claude_gpt_comparison\segmentation_size\summary.json`
- `claude_gpt_comparison\segmentation_size\per_pair_coverage.csv`

## 5. Recompute repeatability results

The 60 fresh responses, 12 frozen baseline responses, five run prompts, input
snapshots, and protocol are retained.

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\analyze_llm_repeatability_study.py'
```

This recreates rendered repeatability maps, all metrics, and all 12 collages.
Only pair 47 is used by the final presentation.

## 6. Rebuild the latest presentation

`Why_Not_Use_LLM_Experiment.pptx` is the canonical edited layout source. Apply
the computed segmentation-size statistics:

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\add_segmentation_size_stats_to_deck.py'
```

This creates `Why_Not_Use_LLM_With_Size_Stats.pptx`, the latest presentation.

## Evidence integrity

`FINAL_EVIDENCE_MANIFEST.json` records every retained artifact, its role, byte
size, and SHA-256 hash. The cleanup itself is reproducible and defaults to a
dry run:

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\cleanup_llm_experiment_artifacts.py'
```

Add `--apply` only when intentionally performing the documented compact
cleanup.
