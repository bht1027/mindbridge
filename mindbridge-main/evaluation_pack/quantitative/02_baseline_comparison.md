# 4.3.2 Comparative Analysis with Baselines

We compare our proposed multi-agent architecture against a simpler single-agent baseline. The baseline directly responds to the user with one model call, while the proposed system decomposes the task into input analysis, retrieval, empathy generation, strategy planning, safety checking, coordination, critique, and revision.

## Comparison goal

The purpose of the baseline comparison is to test whether architectural decomposition provides measurable benefits over a simpler design. If the pipeline outperforms the baseline, we can argue that role specialization and self-reflection are useful design choices for supportive dialogue.

## Suggested narrative

Compared with the single-agent baseline, the full pipeline tends to produce more structured answers, stronger emotional acknowledgment, and safer practical suggestions. The baseline often gives broadly supportive but generic replies, whereas the pipeline can combine context analysis, retrieval support, and multi-stage refinement to produce more deliberate responses.

This comparison is especially important because a supportive-dialogue system can sound acceptable in isolated examples while still failing to consistently balance empathy, usefulness, and safety. The baseline therefore serves as a reference point that makes the value of the full architecture easier to demonstrate.

## Side-by-side qualitative summary

| Aspect | Single-agent baseline | Multi-agent pipeline |
|---|---|---|
| Architecture | one direct response generator | staged specialist pipeline |
| Strength | faster, simpler | more structured and safer |
| Weakness | generic advice, less controllable | slower and more complex |
| Expected benefit | minimal engineering cost | better quality and interpretability |

## Result summary paragraph template

The baseline comparison shows that the multi-agent design is not only a conceptual architecture choice but also an empirically useful one. Relative to the single-agent baseline, the full pipeline provides more balanced supportive responses and better preserves safety constraints. This supports our hypothesis that modular reasoning and reflection improve supportive-dialogue quality.
