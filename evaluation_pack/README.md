# Evaluation Pack for MindBridge

This pack adds a standalone evaluation folder structure you can drop into your project.

## Suggested repo placement

```text
mindbridge-main/
  evaluation/
    README.md
    quantitative/
      01_performance_measurements.md
      02_baseline_comparison.md
      03_statistical_significance.md
      04_ablation_studies.md
    qualitative/
      01_case_studies.md
      02_user_study_optional.md
      03_error_analysis.md
      04_limitations_future_work.md
    templates/
      results_table_template.md
      case_study_template.md
    scripts/
      evaluation_checklist.md
```

## How to use

1. Copy the whole `evaluation_pack` folder into your repo and rename it to `evaluation`.
2. Fill in your actual numbers from `evaluate.py` outputs.
3. Keep the structure separate so your TA can see you covered every rubric item.
