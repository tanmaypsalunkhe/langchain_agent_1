# Resume-Specific Interview Questions

Questions an interviewer will ask by literally reading your CV line by line. Answers are drafted from what you've told me — check each one is still accurate before you use it, and fill in the [bracketed] spots I couldn't answer for you.

---

## "Tell me about yourself" / "Walk me through your resume"

**Suggested answer (60-90 sec):**
"I'm a data scientist by background — 12 years, starting in data engineering at Capgemini, then predictive modeling and analytics platforms at J.P. Morgan, and for the last three years leading AI product work in payments. The throughline is: I keep ending up as the person who turns something manual and high-touch into something that scales. Most recently, my team ran advisory sessions with our top clients — the ones driving 40% of our revenue — diagnosing specific gaps in their payments setup and giving data-backed recommendations. It worked, but it couldn't scale past a handful of clients. So I personally built an AI agent — a RAG pipeline — to encode that same advisory knowledge, and it's since grown into a multi-agent system my team now runs and enhances, plus we've extended the same pattern into a second business line. That's what I'm looking to bring into [this role] — turning something proven but unscalable into something that reaches a lot more people."

---

## About the RAG advisory product

**Q: Why did you build this yourself instead of assigning it to an engineer?**
A: [Your honest answer — likely: to validate the idea fast and cheaply before committing more resources, and because you had the technical background to do it directly. Fill in the real reason.]

**Q: What made you choose LangChain for v1 and LangGraph for the multi-agent evolution?**
A: LangChain was the right starting point for a straightforward RAG pipeline — retrieval, prompt assembly, generation. As the system grew into something that needed multiple specialized steps with conditional logic (e.g., a retrieval step, a recommendation step, an evaluation/guardrail step that could trigger a retry), LangGraph's state-machine model gave the control and inspectability a single linear chain couldn't.

**Q: How exactly is the >90% accuracy and 30% hallucination reduction measured?**
A: [Be ready with specifics: what's the evaluation set, how is "accuracy" defined for this use case — answer correctness against a labeled set? Groundedness against retrieved context? Who reviews it? This is the single most likely technical follow-up question given how prominent this metric is across your CV versions — do not go into an interview without a crisp, specific answer here.]

**Q: What data sources feed the RAG pipeline?**
A: Intranet documents, product construct documents, team/SME knowledge, Snowflake, Databricks, and multiple MCP servers — connecting structured transaction data with unstructured knowledge (scheme notifications, fee changes, product playbooks).

**Q: What happens when the system gets something wrong?**
A: Human-in-the-loop validation and a fallback/retry loop triggered when the accuracy/confidence score falls below threshold — the system doesn't silently serve a low-confidence answer. [Add specifics if you have them: who reviews flagged cases, how often does the fallback trigger in practice.]

**Q: Who are the actual end users, day to day?**
A: Relationship Managers and Product teams — they use it directly for self-serve decks, dashboards, analysis, and talking points in client conversations.

---

## About the team and leadership

**Q: Walk me through your team structure.**
A: 7 people across two management layers — 4 direct reports (3 VPs, 1 Associate), plus 3 further Associates who report to one of those VPs. Functionally, they're split into three focused areas: product build/model enhancement, client advisory, and stakeholder requirements. A sub-team of 1 VP and 2 Associates now specifically maintains, fine-tunes, and enhances the models I originally built.

**Q: Managing VPs as a VP yourself — how does that work, practically?**
A: [This will come up — be ready. Likely answer shape: it's about scope/mandate on this specific initiative rather than title hierarchy; be honest about how the reporting line actually works at J.P. Morgan in this case.]

**Q: How do you develop the people on your team?**
A: Regular check-ins and 1:1s — day-to-day coaching rather than a formal career-pathway program. [If you want a stronger answer here long-term, worth thinking about one specific example of someone you've coached and what changed for them.]

---

## About the Treasury/Merchant Services expansion

**Q: Why did you extend into Treasury and Merchant Services specifically?**
A: [Fill in the real trigger — was it a stakeholder request, a data signal, your own initiative? "Identified and pursued a new market opportunity" is the resume framing, but be ready with the concrete story of how you actually identified it.]

**Q: How is this different technically from the original product?**
A: Same underlying pattern — target identification, opportunity mapping, adoption tracking — applied to a new domain: connecting bankers to reciprocal referral opportunities between Treasury Services and Merchant Services clients.

**Q: What's actually meant by "the only solution in the org" for this?**
A: [Be ready to be specific and not oversell this. What exactly makes it unique — is it that no one else has combined TS/MS data, or that no one else built a self-serve tool for it? Precision matters here; "only solution in the org" is a strong claim you should be able to immediately substantiate or soften if pressed.]

---

## About the fraud/credit-risk work (2018-2022 role)

**Q: Tell me about the XGBoost fraud model — what features mattered most, and how did you validate an 18% false-positive reduction?**
A: [This is a real technical drill-down question. Be ready to discuss feature engineering, how you validated against a holdout set or A/B test, and what the baseline was that the 18% improvement is measured against.]

**Q: Tell me about the EU Payments Analytics Platform — what was your specific role in "leading the strategy, design, and deployment"?**
A: [Be specific about what you personally decided vs. what the team executed — interviewers distinguish between "I led" meaning "I was the primary decision-maker" versus "I was the most senior person associated with it."]

---

## The patent and hackathons

**Q: Tell me about the "Fraud Consortium" patent idea — what's the actual innovation?**
A: [You need a genuinely clear, concise explanation ready here — "selected for first-stage petition" invites "so what is it?" as the immediate next question. If you can't explain the core idea in two sentences, an interviewer will notice the gap between the resume line and your actual command of the material.]

**Q: What did you build for the hackathon wins, and what made them go into production?**
A: [Same principle — have the specific idea and the specific production outcome ready, not just the fact that you won.]

---

## Anticipate the "prove it's really you, not your team" question
Director-level interviewers are trained to probe the line between what *you* did and what your *team* did, especially given how much of your story is now told through "my team" framing. Before each interview, mentally sort every claim on your resume into: (1) things you personally built or decided, (2) things you directed but a team executed, (3) things the team now owns that you originally started. All three are legitimate leadership stories — but know which bucket each one is in before someone asks.

---

## How to use this
Go through every `[bracketed]` prompt above and actually write the real answer before your next interview — those are the specific spots where I don't have the real story from you yet, and they're also, not coincidentally, the exact places a sharp interviewer will dig.
