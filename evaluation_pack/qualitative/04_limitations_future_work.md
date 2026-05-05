# 4.3.8 Limitations and Future Work

## Limitations

1. **Knowledge base scale**: our retrieval component uses a small handcrafted support knowledge base rather than a large external corpus.
2. **Lightweight retrieval method**: hybrid lexical/vector retrieval is easy to inspect, but it still uses a small local knowledge base rather than a large embedding index or cross-encoder reranker.
3. **LLM judge dependence**: automatic quality scoring depends on a judge model, which may introduce bias.
4. **Prompt sensitivity**: system behavior remains highly dependent on prompt design and stage instructions.
5. **Not a clinical system**: this project is a research prototype for supportive dialogue, not a production mental-health service.

## Future work

1. Expand the support knowledge base with richer and more diverse cases.
2. Replace the lightweight TF-IDF vector stage with embedding-based semantic retrieval and a stronger reranker.
3. Add stronger memory modeling for longer conversations.
4. Conduct human evaluation with side-by-side comparisons.
5. Improve routing so short inputs like "I am sad" can trigger more grounded support.
6. Add better demo transparency by surfacing retrieved items and agent traces.

## Report-ready conclusion paragraph

Overall, the current system demonstrates that a multi-agent architecture can improve supportive-dialogue quality compared with a simpler baseline, especially in empathy, grounding, and safety control. At the same time, the project still has clear technical limitations in retrieval strength, prompt robustness, and evaluation scale. These limitations make the system an appropriate research prototype and a strong foundation for future iteration, rather than a finished real-world support product.
