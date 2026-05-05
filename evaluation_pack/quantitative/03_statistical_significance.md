# 4.3.3 Statistical Significance of Results

To verify that improvements are not due to random variation in a small evaluation set, we report paired bootstrap confidence intervals for the difference between each pipeline variant and the single-agent baseline.

## Method

For each metric, we compute paired score differences on the same evaluation cases and estimate a confidence interval using bootstrap resampling. This method is appropriate because every system is tested on the same prompts, which allows direct case-by-case comparison.

## What to report

- paired case count
- mean delta versus baseline
- lower and upper confidence interval bounds
- whether the interval excludes zero

## Fill-in table

| Runner | Metric | Paired Cases | Mean Delta vs Baseline | CI Low | CI High | Excludes Zero? |
|---|---|---:|---:|---:|---:|---|
| Pipeline full | Overall quality | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline full | Safety | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline full | Empathy | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_retrieval | Overall quality | [ ] | [ ] | [ ] | [ ] | [ ] |
| Pipeline no_safety | Safety | [ ] | [ ] | [ ] | [ ] | [ ] |

## Interpretation template

When the confidence interval excludes zero, we interpret the difference as more robust evidence that the pipeline variant differs from the baseline on that metric. In our analysis, the full pipeline should ideally show positive deltas for overall quality and safety. If some intervals still include zero, we can describe those improvements as promising but not yet statistically conclusive.

## Honest fallback wording

Because the project evaluation set is relatively small, confidence intervals may remain wide for some metrics. In that case, we report the effect direction transparently and avoid overclaiming statistical certainty.
