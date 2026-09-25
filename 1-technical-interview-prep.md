# Technical Interview Prep
### For: Principal AI Engineer / AI-ML Engineer / AI Product Manager (technical rounds) / Data Analytics Director

Organized by topic. Each has: core Q&A, then "go deeper" sources if you want more than the answer gives you. Practice saying every answer out loud — reading it silently is not the same skill as producing it under pressure.

---

## 1. RAG (Retrieval-Augmented Generation)

**Q: Walk me through a RAG pipeline end to end.**
A: Ingest documents → split into chunks → embed each chunk into a vector → store vectors in an index (vector DB) → at query time, embed the user's question → retrieve the top-k most similar chunks → optionally rerank them → pass the retrieved chunks + question to the LLM as context → LLM generates a grounded answer. The value of RAG is that the LLM answers from your specific documents instead of only its training data, and you can cite/verify the source.

**Q: Why RAG instead of just fine-tuning the model on your documents?**
A: RAG is cheaper and faster to update — when a document changes, you re-index it, not retrain the model. It's also more auditable (you can show which chunk the answer came from) and reduces hallucination because the model is grounded in retrieved text rather than relying purely on parametric memory. Fine-tuning is better when you need to change the model's *behavior/style/format*, not just give it new facts.

**Q: What's chunking and why does chunk size matter?**
A: Chunking splits documents into smaller pieces before embedding, because embedding a whole document loses granularity (a 50-page PDF embedded as one vector can't be matched precisely to a specific question). Too small a chunk loses context (a sentence fragment without surrounding meaning); too large a chunk dilutes relevance (the important sentence gets averaged out among irrelevant ones) and wastes context-window budget at generation time. Common approach: chunk by semantic boundary (paragraph/section) with overlap (e.g. 50-100 tokens) so a fact split across a chunk boundary isn't lost entirely.

**Q: What is hybrid search / BM25 + vector search, and why combine them?**
A: BM25 is a classical keyword/lexical search algorithm (ranks documents by term frequency and rarity — TF-IDF style). Vector/semantic search finds chunks with similar *meaning* even if the exact words differ. Combining them (hybrid search) catches both cases: BM25 wins when the user searches an exact term, code, ID, or acronym that a semantic embedding might blur; vector search wins when the user asks conceptually ("how do I reduce declines" matching a chunk that never uses the word "declines"). You typically run both, then merge/normalize the two score lists — a common method is Reciprocal Rank Fusion (RRF).

**Q: What is reranking, and where does it sit in the pipeline?**
A: After initial retrieval (which is optimized for speed over a large corpus, e.g. top 50-100 candidates), a reranker — usually a heavier cross-encoder model — re-scores that smaller candidate set with a more accurate but slower model, and you keep only the top-k (e.g. top 5) to send to the LLM. It's a two-stage retrieve-then-rerank pattern: cheap/broad first pass, expensive/precise second pass.

**Q: What is MMR (Maximal Marginal Relevance) and why would you use it?**
A: MMR selects retrieved results balancing relevance against redundancy — instead of returning the 5 most similar chunks (which might all say nearly the same thing), MMR penalizes chunks that are too similar to ones already selected, so the final set covers more distinct information. Useful when you want diverse supporting evidence rather than five near-duplicate chunks.

**Q: What's the difference between a vector database like pgvector vs. OpenSearch?**
A: pgvector is a PostgreSQL extension — vectors live inside your existing relational database, so you get one system for structured + vector data, simpler ops, and easy joins between vector similarity and SQL filters. OpenSearch is a dedicated search/analytics engine — built for scale, has native hybrid search (BM25 + k-NN vector) and reranking pipelines out of the box, and is the better choice for very large, unstructured, text-heavy corpora. A common real pattern: use pgvector for structured/transactional data joined with embeddings, OpenSearch for large unstructured document search, then merge results (exactly what "hybrid retrieval" across both systems means in practice).

**Q: What's session memory / PostgreSQL as memory, and why does it matter for cost?**
A: Instead of re-sending the full conversation history (all prior tokens) with every request — which multiplies token cost and latency as a session grows — you store a compressed/summarized representation of the session state (facts established, prior retrieved context, user preferences) in a database like PostgreSQL, and retrieve/inject only what's relevant for the current turn. This cuts inference cost because you're not re-processing the entire growing context every single call.

**Go deeper:**
- RAG Tutorial 2026: Complete Introduction — https://www.youtube.com/watch?v=vAK0iqA6-QI
- OpenSearch vector search tutorials (official) — https://docs.opensearch.org/2.19/tutorials/vector-search/index
- pgvector vs OpenSearch comparison — https://www.instaclustr.com/education/vector-database/pgvector-vs-opensearch-for-vector-databases-5-differences-and-how-to-choose/
- AWS sample notebook: hybrid pgvector + OpenSearch + Bedrock + LangChain EnsembleRetriever (Reciprocal Rank Fusion) — https://github.com/aws-samples/hybrid-search-postgres-opensearch-bedrock

---

## 2. Agentic Application Architecture (LangChain, LangGraph, Multi-Agent, MCP)

**Q: What's the difference between a simple LLM chain and an agent?**
A: A chain is a fixed sequence of steps (prompt → LLM → parse → done). An agent can *reason about what to do next* — it decides which tool to call, whether to call another tool after seeing the result, and when it has enough information to answer. The key difference is a loop with decision-making, not a straight line.

**Q: What is LangGraph and why use it over a basic LangChain agent?**
A: LangGraph models an agent as a state machine/graph — explicit nodes (steps) and edges (transitions), with shared state passed between them. This gives you control a simple agent loop doesn't: conditional branching, cycles (retry loops), persistence of state between steps, and the ability to have multiple specialized agents pass control between each other (multi-agent orchestration). It's the difference between "the LLM freely decides everything" and "the LLM makes decisions within a structured, inspectable, controllable flow" — important for production reliability and debugging.

**Q: What is multi-agent orchestration, and when do you actually need multiple agents instead of one with more tools?**
A: Multi-agent orchestration splits a complex task across specialized agents (e.g., a retrieval agent, a reasoning/recommendation agent, an evaluation agent) that hand off to each other, rather than one agent trying to do everything with a huge prompt and tool list. You need it when: a single agent's prompt/tool list gets too large and starts making mistakes from context overload; different sub-tasks need genuinely different strategies or models; or you want a separation of concerns for auditability (e.g., a dedicated eval/guardrail agent that checks the primary agent's output before it reaches the user).

**Q: What is MCP (Model Context Protocol) and why does it matter?**
A: MCP is an open standard (analogous to "USB-C for AI applications") for connecting an AI application to external tools and data sources — databases, file systems, APIs — in a standardized way, instead of every integration being custom-built. An MCP server exposes a set of tools/resources; any MCP-compatible client (an agent) can discover and call them the same way, regardless of who built the server. In a real system, this is how an agent might query Snowflake, Databricks, or an internal knowledge base without bespoke integration code for each one.

**Q: What is client-server architecture, in the context of an AI application?**
A: The client (e.g., the agent or the front-end) sends requests; the server (the model API, the retrieval service, the MCP tool server) processes and returns a response. In an agentic system this usually means: the orchestrating agent is a client to several servers — the LLM API, the vector DB, the MCP tool servers, the evaluation service — each with its own API contract, and the agent coordinates calls across all of them.

**Go deeper:**
- Krish Naik — Agentic AI playlist (LangGraph, orchestration) — https://www.youtube.com/playlist?list=PLZoTAELRMXVMBr14UQ30AFlnlQ7eL5wjl
- Krish Naik — MCP explained (25 min) — search "Krish Naik All You Need To Know About MCP" on YouTube
- MCP official docs — https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro
- MCP Tutorial: client-server architecture explained (16 min, Dan Vega) — search "MCP Explained Dan Vega" on YouTube

---

## 3. Fine-Tuning (LoRA, PEFT, RLHF)

**Q: What is PEFT and why not just fully fine-tune the model?**
A: PEFT (Parameter-Efficient Fine-Tuning) updates only a small subset of a model's parameters instead of all of them. Full fine-tuning of a large model requires huge GPU memory and storage per fine-tuned version (you're duplicating billions of parameters). PEFT methods freeze the base model and train small additional components, making fine-tuning feasible on far less hardware and letting you store many small "adapters" instead of many full model copies.

**Q: What is LoRA specifically?**
A: LoRA (Low-Rank Adaptation) freezes the original model weights and injects small, trainable low-rank matrices into specific layers (commonly attention layers). Instead of updating a full weight matrix, you learn two much smaller matrices whose product approximates the needed change. This drastically cuts the number of trainable parameters (often <1% of the full model) while achieving comparable task performance to full fine-tuning for many use cases.

**Q: What is QLoRA, and how is it different from LoRA?**
A: QLoRA combines LoRA with quantization — the frozen base model is loaded in a compressed, lower-precision format (e.g. 4-bit) to cut memory usage further, while the LoRA adapters are still trained in higher precision. This lets you fine-tune much larger models on consumer/limited GPU hardware that couldn't otherwise hold the full model in memory.

**Q: What is RLHF, in plain terms?**
A: Reinforcement Learning from Human Feedback. After a base model is trained, humans rank multiple model outputs for the same prompt (which response is better). Those rankings train a separate "reward model" that predicts human preference. The base model is then fine-tuned using reinforcement learning to produce outputs the reward model scores highly — aligning the model's behavior with what humans actually prefer (helpfulness, safety, tone), not just what's statistically likely from training data.

**Q: When would you actually reach for fine-tuning over prompt engineering or RAG?**
A: When the problem is about *changing the model's behavior/format/style consistently* (e.g., always output a specific JSON schema, adopt a specific domain tone, follow a specific reasoning pattern) rather than giving it new facts (that's RAG's job) or steering a single interaction (that's prompting's job). Fine-tuning is the heaviest, most expensive lever — reach for it last, after prompting and RAG have been tried and fall short.

**Go deeper:**
- freeCodeCamp — LLM Fine-Tuning Course (12 hrs): PEFT, LoRA, QLoRA, RLHF, DPO — https://www.youtube.com/watch?v=CcrC5zSv1iA (code: https://github.com/sunnysavita10/Complete-LLM-Finetuning)
- Fine-Tuning an LLM with LoRA and QLoRA: A Hands-On Guide (written) — https://medium.com/@aminfadaeinejad.edu/fine-tuning-an-llm-with-lora-and-qlora-a-hands-on-guide-441ea09360c5

---

## 4. Evaluation, Guardrails, and Safety

**Q: How do you evaluate an LLM/RAG system in production?**
A: Split into retrieval metrics and generation metrics. Retrieval: recall@k (did the right chunk appear in the top k results?), MRR (Mean Reciprocal Rank — how high up was the first correct result?). Generation: faithfulness/groundedness (does the answer only state what's actually supported by the retrieved context — checks against hallucination), answer relevance (does it actually address the question), and for classification-style tasks, precision/recall/F1. You also want an automated "LLM-as-judge" setup for scale, backed by periodic human spot-checks, since human eval alone doesn't scale.

**Q: Explain F1 score, and why not just use accuracy.**
A: F1 is the harmonic mean of precision and recall: F1 = 2 × (precision × recall) / (precision + recall). Precision = of everything you flagged positive, how much was actually positive. Recall = of everything actually positive, how much did you catch. Accuracy is misleading on imbalanced data — e.g., fraud detection where 99% of transactions are legitimate: a model that predicts "not fraud" every time gets 99% accuracy while catching zero fraud. F1 forces you to balance catching true positives (recall) against not flooding the system with false alarms (precision), which is why it's the standard metric for imbalanced classification problems like fraud.

**Q: What are guardrails, concretely — what do they actually check?**
A: Guardrails are checks applied to model input and/or output before it reaches the user or a downstream system. Typically: input guardrails (block prompt injection, off-topic requests, PII in the query), output guardrails (block hallucinated claims not grounded in retrieved context, block unsafe/toxic content, enforce output format/schema, block disclosure of sensitive internal data). Guardrails can be rule-based (regex, keyword blocklists, schema validators) or model-based (a smaller classifier model scoring the output for safety/faithfulness).

**Q: What is human-in-the-loop validation, and where does it fit?**
A: A point in the pipeline where a human reviews and can approve, correct, or reject the model's output before it's finalized — either on every case (high-stakes/low-volume) or a sampled subset (for ongoing quality monitoring and to generate labeled data for retraining/eval). It's a deliberate trade-off: slower/costlier per case, but a safety net for cases where full automation risk is too high, and a valuable source of ground-truth feedback for improving the system over time.

**Q: What's a fallback/retry loop tied to an accuracy threshold?**
A: If the automated evaluation score for a given response falls below a defined confidence/accuracy threshold, instead of returning that answer to the user, the system triggers a fallback: re-run with adjusted retrieval, escalate to a human reviewer, or return a safe default/"I don't have enough information" response rather than a low-confidence answer. This prevents low-quality outputs from silently reaching production users.

**Q: What's "deterministic output design" mean for an LLM system, which is inherently probabilistic?**
A: Constraining the LLM's output to a fixed, predictable structure — e.g., forcing JSON-schema-conformant output, using low/zero temperature for tasks needing consistency, or having the LLM select from a constrained set of valid options rather than free-form text — so that downstream systems consuming the output can rely on its shape even though the underlying model is probabilistic. It's about controlling variance in *format and structure*, not claiming the content itself is deterministic.

**Go deeper:**
- Bedrock course — AI Safety, Evaluation & Watermark Detection module — https://www.udemy.com/course/complete-aws-bedrock-generative-ai-course-projects/
- StatQuest — Precision, Recall, F1 explained simply — search "StatQuest precision recall" on YouTube

---

## 5. Classical ML & Data Science (XGBoost, Anomaly Detection, Regression)

**Q: How does XGBoost work, at a level you could explain to a non-technical stakeholder?**
A: It builds an ensemble of decision trees sequentially — each new tree is trained specifically to correct the errors (residuals) of the trees built before it ("boosting"). This is different from Random Forest, which builds many trees independently and averages them ("bagging"). Boosting tends to reduce bias and often outperforms on structured/tabular data, which is why it's a go-to for fraud detection, credit risk, and other tabular business problems.

**Q: How would you approach anomaly detection for fraud, and why not just use a simple threshold rule?**
A: Fraud is rare and evolving, so pure rule-based thresholds miss novel patterns and go stale quickly. A model-based approach (e.g., anomaly-scoring or classification models trained on labeled fraud/non-fraud transactions) can learn complex, non-obvious combinations of features that indicate fraud, and can be retrained as patterns shift. In practice, the strongest systems combine both: deterministic rules for known, well-understood fraud patterns (fast, explainable, no false-negative risk on known cases) plus ML models for catching novel/complex patterns rules would miss.

**Q: Explain the bias-variance trade-off in one or two sentences.**
A: Bias is error from a model being too simple to capture the real pattern (underfitting); variance is error from a model being too sensitive to the specific training data, capturing noise instead of signal (overfitting). You're always trading one against the other — regularization, cross-validation, and ensemble methods are tools for finding the right balance.

**Q: Why did you choose Ridge/Linear Regression for portfolio forecasting instead of a more complex model?**
A: (Tailor to your actual answer, but the general principle:) When the relationship between features and target is genuinely close to linear, and interpretability/explainability matters to stakeholders (e.g., regulators, finance partners who need to understand *why* a forecast moved), a simpler, well-understood model is often the right choice over a black-box model that offers marginal accuracy gains but no interpretability. Regularization (Ridge) also helps when you have correlated features, preventing unstable coefficient estimates.

**Go deeper:**
- StatQuest with Josh Starmer — the best free source for all of the above, visual and fast (10-20 min each) — https://www.youtube.com/@statquest/playlists
- StatQuest — Gradient Boost & XGBoost — https://m.youtube.com/playlist?list=PLZ5DHV9_5h9vQwAImmNi1RfoTtSuOUjwM

---

## 6. Cloud Infrastructure & Production Deployment (AWS, Docker, CI/CD)

**Q: Walk me through your production deployment architecture.**
A: (Tailor to your real setup, but the general shape to hit:) Application code containerized with Docker for consistent, portable deployment across environments. CI/CD pipeline (e.g., Jules, Spinnaker) automates build → test → deploy, including staged rollouts (canary/blue-green) so a bad deploy doesn't hit 100% of traffic at once. AWS Bedrock for model access, S3 for document/artifact storage, IAM for least-privilege access control between services, Lambda for event-driven/serverless compute where appropriate, and CloudWatch for logging, metrics, and tracing — including tracking accuracy-related metrics over time so you can detect model drift or degradation in production, not just at deploy time.

**Q: What's the difference between canary and blue-green deployment?**
A: Blue-green: you run two full environments (old "blue," new "green"), test the new one, then switch all traffic over at once — fast rollback (just switch back) but an all-or-nothing cutover. Canary: you gradually shift a small percentage of traffic to the new version, monitor for problems, and ramp up — slower but catches issues before they affect all users, at the cost of running mixed versions simultaneously for a period.

**Q: Why use IAM roles instead of hardcoded credentials for service-to-service access?**
A: IAM roles grant temporary, scoped permissions to a service (e.g., "this Lambda function can read from this specific S3 bucket and call this specific Bedrock model, nothing else") without embedding long-lived credentials in code, which is both a security risk (credentials can leak) and an operational one (rotating hardcoded secrets is painful). It's the least-privilege principle applied to infrastructure.

**Q: What does CloudWatch actually give you that application logs alone don't?**
A: Centralized, queryable logs and metrics across distributed services, with alerting (trigger a notification when a metric crosses a threshold, e.g., error rate spike or latency spike) and tracing (following a single request across multiple services to find where it slowed down or failed) — essential once you have more than one service, since local logs on individual machines don't give you the cross-system picture.

**Go deeper:**
- AWS Lambda for Absolute Beginners (2026) — https://www.youtube.com/watch?v=QuMy8acc6eQ
- What is AWS Bedrock — full course for beginners — https://www.youtube.com/watch?v=slAD0iuwFEE
- AWS Skill Builder — official AI Practitioner learning plan — https://skillbuilder.aws/learning-plan/G8ENMJ5QBE/aws-artificial-intelligence-practitioner-learning-plan/SU2A1EJM1A
- freeCodeCamp — AWS Cloud Project Bootcamp (S3, IAM, full stack) — search on youtube.com

---

## How to use this before an interview
1. Don't just read — for each Q, close the doc and answer out loud from memory, then check yourself against the written answer.
2. For every topic, have a "my project" version ready: how does BM25/reranking/MMR, or the eval/guardrails setup, or the CI/CD pipeline, map to *your actual RAG advisory product*? Interviewers will ask you to apply the concept to your own work, not recite the definition.
3. Prioritize by your own gaps: if fine-tuning (LoRA/PEFT/RLHF) and MCP are the areas you're least hands-on with, spend disproportionate time there — the classical ML and cloud infra sections are likely closer to what you already know cold.
