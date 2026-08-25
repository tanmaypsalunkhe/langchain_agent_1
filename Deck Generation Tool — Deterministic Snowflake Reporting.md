# Deck Generation Tool — Deterministic Snowflake Reporting

## 1. Purpose

The system includes a Deck Generation Tool as one of the tools available to the multi-agent system.

The deck is NOT generated from the agent's analysis, reasoning, recommendations, or free-form LLM output.

Instead, the deck is based on a **pre-defined PowerPoint/template structure** and retrieves specific approved fields directly from Snowflake.

The agent's role is only to determine whether the deck-generation tool should be invoked and, where applicable, provide the parameters required to generate the deck.

The actual business values displayed in the deck must come directly from Snowflake.

---

# 2. Core Principle

The architecture must maintain a strict separation:

USER QUERY
    ↓
AGENTS
    ↓
ANALYSIS / RECOMMENDATION

and independently:

DECK TOOL
    ↓
PREDEFINED DECK TEMPLATE
    ↓
SNOWFLAKE
    ↓
APPROVED DATA FIELDS
    ↓
POWERPOINT

The agent must NEVER:

- Invent deck values
- Calculate or modify business KPIs unless explicitly defined by the deck logic
- Copy its own generated analysis into the deck
- Substitute an LLM-generated value for a Snowflake value
- Generate arbitrary slide structures
- Change the predefined business logic of the deck

---

# 3. Updated Architecture

The overall architecture should be:

                         USER
                           │
                           ▼
                     FastAPI API
                           │
                           ▼
                   Pre-Query Guardrails
                           │
                           ▼
                 LangGraph Supervisor
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
        RAG Agent     Snowflake Agent   Deck Tool
             │             │              │
             │             │              │
             │             │         Direct Snowflake
             │             │              │
             ▼             ▼              ▼
          Evidence      Analysis     Predefined
             │             │          Deck Template
             │             │              │
             └───────┬─────┘              │
                     ▼                    │
               Recommendation             │
                     │                    │
                     │                    │
                     └──────────┐         │
                                ▼         ▼
                           Final Answer  Deck
                                │         │
                                └────┬────┘
                                     ▼
                             Post-Response
                              Guardrails
                                     │
                                     ▼
                              LangSmith Eval
```

The key point is that the **Deck Tool is an independent branch**.

It does not consume the Recommendation Agent's output.

---

# 4. Deck Tool Architecture

The deck tool should follow:

Agent
  ↓
Deck Tool Invocation
  ↓
Deck Generator Service
  ↓
Predefined Deck Template
  ↓
Snowflake Data Retrieval
  ↓
Populate Approved Fields
  ↓
Generate PowerPoint
  ↓
Validate Deck
  ↓
Return Deck Artifact

Example:

generate_deck(
    merchant_id,
    reporting_period,
    deck_type
)

The tool then independently performs:

1. Validate input parameters.
2. Identify the predefined deck template.
3. Execute approved Snowflake queries.
4. Retrieve the required fields.
5. Populate the corresponding PowerPoint placeholders.
6. Generate charts/tables defined by the template.
7. Validate the generated presentation.
8. Return the PowerPoint artifact.

---

# 5. Agent Responsibility

The agent should NOT decide what values go into individual slides.

For example, the agent may determine:

"The user is asking for the Merchant Performance Review deck."

It can then invoke:

generate_merchant_performance_deck(
    merchant_id="12345",
    period="Q2-2026"
)

The deck tool itself knows:

- Which Snowflake tables to query
- Which columns to retrieve
- Which calculations are permitted
- Which slide receives each value
- Which chart to generate
- Which template to use

This makes the deck deterministic.

---

# 6. Snowflake Access

The Deck Generation Tool should use the approved Snowflake access mechanism.

If the enterprise architecture requires Snowflake MCP:

Agent
  ↓
Deck Tool
  ↓
Snowflake MCP
  ↓
Snowflake

Alternatively, if the deck-generation service has an approved direct data-access mechanism, follow that existing architecture.

Do not create a second uncontrolled Snowflake connection.

The implementation should follow the enterprise security model.

---

# 7. Predefined Data Contract

Create a strict data contract between Snowflake and the deck.

For example:

{
    "merchant_name": "...",
    "reporting_period": "...",
    "revenue": 1234567,
    "authorization_rate": 0.923,
    "transaction_count": 123456,
    "growth_rate": 0.12,
    "top_segment": "...",
    "risk_indicator": "...",
    "performance_status": "..."
}

The deck template consumes this structured object.

The LLM should not modify these values.

---

# 8. Template Mapping

Create an explicit mapping between Snowflake fields and deck placeholders.

Example:

SNOWFLAKE FIELD
    ↓
merchant_name
    ↓
Slide 1 / Title Placeholder

SNOWFLAKE FIELD
    ↓
revenue
    ↓
Slide 2 / KPI Card

SNOWFLAKE FIELD
    ↓
authorization_rate
    ↓
Slide 3 / KPI + Chart

SNOWFLAKE FIELD
    ↓
segment_performance
    ↓
Slide 4 / Table

This mapping should be configuration-driven where possible rather than embedded throughout the application.

---

# 9. No LLM in the Data Population Path

The LLM should NOT sit between Snowflake and the PowerPoint template.

Avoid:

Snowflake
 ↓
LLM
 ↓
PowerPoint

Use:

Snowflake
 ↓
Validated Data Contract
 ↓
Template Engine
 ↓
PowerPoint

This ensures that:

- Numbers remain accurate.
- Reproducibility is maintained.
- The same Snowflake data produces the same deck.
- The deck is auditable.
- Hallucination risk is eliminated from the deck population process.

---

# 10. Deck Validation

After generating the deck, validate:

### Data validation

- Required fields exist.
- Values match Snowflake.
- No unexpected nulls.
- Correct reporting period.
- Correct merchant/entity.
- Correct units.

### Template validation

- Correct template used.
- Expected slides exist.
- Expected placeholders populated.
- Charts generated correctly.
- No empty required sections.

### Presentation validation

- No text overflow.
- No broken images.
- No missing charts.
- No duplicate slides.
- No corrupted PowerPoint objects.

The generated deck should contain metadata identifying:

- Data extraction timestamp
- Reporting period
- Data source
- Deck template version
- Generation request ID

---

# 11. Agent Interaction Example

User:

"Generate the Q2 merchant performance deck for Merchant ABC."

LangGraph Supervisor:

1. Understand request.
2. Identify deck-generation intent.
3. Extract:
   - Merchant = ABC
   - Period = Q2
   - Deck type = Merchant Performance
4. Invoke:

generate_merchant_performance_deck(
    merchant="ABC",
    period="Q2"
)

The Deck Tool:

1. Loads the predefined Merchant Performance template.
2. Queries Snowflake.
3. Retrieves approved fields.
4. Populates template.
5. Generates charts/tables according to predefined logic.
6. Validates values.
7. Creates PowerPoint.
8. Returns artifact.

The final response:

"Your Q2 Merchant Performance deck has been generated."

The deck itself contains only approved Snowflake-derived information.

---

# 12. Important Separation from the AI Analysis

The system may simultaneously perform AI analysis.

For example:

User:

"Analyze Merchant ABC's Q2 performance and generate the Q2 performance deck."

The system performs TWO independent workflows:

### AI Analysis

Snowflake
+
S3 RAG
↓
LangGraph
↓
Analysis Agent
↓
Recommendation Agent
↓
AI Answer

### Deck

Snowflake
↓
Predefined queries
↓
Approved data contract
↓
Predefined template
↓
PowerPoint

The AI answer and the deck can be returned together, but they are generated independently.

The deck must not inherit numbers from the LLM-generated answer.

---

# 13. Why This Architecture Is Preferred

This design provides:

- Strong data governance
- Reproducibility
- Auditability
- Lower hallucination risk
- Consistent executive reporting
- Separation of AI reasoning from deterministic reporting
- Easier testing
- Easier regulatory/control review
- Clear ownership between agent logic and reporting logic

The deck becomes a **trusted reporting artifact**, while the agent remains the **intelligence and recommendation layer**.

---

# 14. Final System Capability

The completed system should therefore support two distinct outcomes.

## Outcome A — AI Decision Support

Snowflake
+
S3 Knowledge Base
↓
Multi-Agent LangGraph
↓
Analysis
↓
Recommendations
↓
Grounded Answer

## Outcome B — Trusted Executive Reporting

Agent determines deck request
↓
Deck Generation Tool
↓
Predefined Template
↓
Direct Approved Snowflake Data
↓
Validated PowerPoint

The system should clearly maintain these two paths throughout the architecture and implementation.
