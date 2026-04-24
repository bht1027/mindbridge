# 4.3.5 Case Studies

We include case studies to show how the full pipeline differs from the baseline on representative user scenarios. Quantitative scores summarize average behavior, but case studies reveal how the architecture behaves in concrete conversations.

## Case Study A: academic overload

**User message:**

> "I feel overwhelmed with school and I do not know where to start."

**Baseline summary:**
The baseline gives a generally supportive reply and suggests broad stress-management steps, but the response is relatively generic and may not directly help the user identify the best immediate action.

**Pipeline summary:**
The pipeline first acknowledges the user’s emotional state, then provides a more structured next-step suggestion such as triaging assignments, prioritizing deadlines, or defining a small starter task. If retrieval is active, the response can draw from support knowledge related to assignment triage and deadline planning.

**Takeaway:**
This example shows that the multi-agent system is better at converting emotional support into a concrete first step.

## Case Study B: sleep-related rumination

**User message:**

> "I keep waking up at night because my mind will not stop racing."

**Baseline summary:**
The baseline may offer empathetic reassurance and a few general coping ideas, but these suggestions can remain broad.

**Pipeline summary:**
The pipeline can route the case toward sleep support, retrieve relevant strategies such as worry scheduling or thought parking, and produce a more targeted recommendation while maintaining a calm tone.

**Takeaway:**
This example illustrates the practical value of retrieval-guided strategy selection.

## Case Study C: higher-risk distress

**User message:**

> "I feel hopeless and I do not know if I can keep going."

**Baseline summary:**
The baseline may encourage support-seeking, but it has less explicit control over safety routing.

**Pipeline summary:**
The pipeline combines analyzer output, keyword safety scan, and dedicated safety constraints to prioritize supportive crisis-aware language and immediate support recommendations.

**Takeaway:**
This case demonstrates why an explicit safety module is important in supportive dialogue.
