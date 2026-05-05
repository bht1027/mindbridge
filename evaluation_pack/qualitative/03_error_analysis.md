# 4.3.7 Error Analysis

We analyze representative failure modes of the system to understand current limitations.

## Error type 1: overly generic empathy

In some low-information inputs such as "I am sad," the system may produce a compassionate but highly generic response. This can happen when the retrieval signal is weak or when the model prioritizes rapport-building over concrete intervention.

## Error type 2: retrieval underuse

Even when the knowledge base contains relevant support items, retrieval may not strongly influence the final answer if the matched content is not sufficiently emphasized during coordination. This can make the response sound less grounded than intended.

## Error type 3: prompt dependence

The system depends heavily on prompt wording and stage design. Small prompt changes can alter tone, action timing, and the balance between reflection and advice.

## Error type 4: latency-cost tradeoff

The full pipeline is more expensive and slower than a single-agent baseline because it uses several sequential and parallel subcalls. This complexity is useful for analysis and quality control, but it reduces efficiency.

## Suggested analysis paragraph

Our error analysis shows that the current system is strongest when the user message contains enough signal for routing and retrieval, but weaker on very short or ambiguous inputs. The pipeline also remains prompt-sensitive, meaning that some behaviors are shaped more by prompt design than by stable reasoning capability. These findings help explain both the strengths and the current limitations of the architecture.
