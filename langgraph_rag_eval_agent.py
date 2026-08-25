"""
LangGraph skeleton: RAG (S3/Bedrock KB) + database retrieval + recommendation + eval loop.

Wire your existing pieces in where marked TODO:
- your Bedrock Knowledge Base retriever (the "basic RAG" you already built)
- your Bedrock LLM for generation

Already filled in below:
- retrieve_db: LangChain SQL query chain against a read-only DB connection
- eval_check: Bedrock Guardrails ApplyGuardrail API, contextual grounding check

pip install langgraph langchain langchain-aws langchain-community boto3 psycopg2-binary
"""

from typing import Annotated, TypedDict, Literal
import operator

import boto3
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------------
# kb_context / db_context use an "append" reducer (operator.add) because the
# router can send execution down both branches in parallel, and both branches
# write to state before the merge node runs. Without the reducer, LangGraph
# would raise an "update conflict" when two parallel nodes touch the same key.

class AgentState(TypedDict):
    query: str
    route: Literal["kb", "db", "both"]
    kb_context: Annotated[list[str], operator.add]
    db_context: Annotated[list[str], operator.add]
    merged_context: str
    recommendation: str
    eval_score: float
    eval_passed: bool
    eval_reason: str
    retry_count: int


MAX_RETRIES = 2
EVAL_THRESHOLD = 0.75  # tune against your own eval dataset

GUARDRAIL_ID = "your-guardrail-id"          # from bedrock.create_guardrail()
GUARDRAIL_VERSION = "DRAFT"                  # or a published version number


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------

def route_query(state: AgentState) -> dict:
    """Classify whether the query needs the KB, the DB, or both.

    Keep this cheap — a small/fast Bedrock model call, or even a rules-based
    classifier if your query patterns are predictable (e.g. "recommend me a
    plan based on my usage" -> both; "what does the policy say about X" -> kb).
    """
    query = state["query"]

    # TODO: replace with a real classifier call
    route = "both"  # kb | db | both

    return {"route": route}


def retrieve_kb(state: AgentState) -> dict:
    """Pull relevant chunks from your existing Bedrock Knowledge Base / S3 vector store."""
    if state["route"] not in ("kb", "both"):
        return {"kb_context": []}

    # TODO: call your existing retriever, e.g.
    # docs = kb_retriever.invoke(state["query"])
    # return {"kb_context": [d.page_content for d in docs]}
    return {"kb_context": ["<TODO: KB retrieval result>"]}


def retrieve_db(state: AgentState) -> dict:
    """Pull relevant structured records from your database.

    Uses LangChain's SQLDatabase + an LLM to turn the natural-language query
    into SQL, run it, and return the rows as context. Swap in your own
    connection string / read-only DB user. Two safety notes:
      - Use a DB role with SELECT-only permissions for this connection.
        Never point an LLM-generated-SQL path at a role that can write.
      - `db.run()` returns a string; for larger result sets you may want
        `SQLDatabaseToolkit` instead so the agent can page through rows
        rather than getting one giant blob back.
    """
    if state["route"] not in ("db", "both"):
        return {"db_context": []}

    from langchain_community.utilities import SQLDatabase
    from langchain.chains import create_sql_query_chain
    from langchain_aws import ChatBedrock

    # TODO: move this connection setup out of the node (build once at module
    # load / app startup) rather than reconnecting on every call.
    db = SQLDatabase.from_uri(
        "postgresql+psycopg2://readonly_user:***@your-db-host:5432/your_db"
    )

    llm = ChatBedrock(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")

    # create_sql_query_chain prompts the LLM with your schema and the
    # question, and returns just the SQL string.
    sql_query_chain = create_sql_query_chain(llm, db)
    sql_query = sql_query_chain.invoke({"question": state["query"]})

    try:
        result = db.run(sql_query)
    except Exception as exc:
        # Don't let a bad generated query kill the whole graph run — surface
        # it as empty-ish context so eval_check correctly scores this branch
        # as ungrounded rather than the node crashing.
        return {"db_context": [f"<DB query failed: {exc}>"]}

    return {"db_context": [f"SQL: {sql_query}\nResult: {result}"]}


def merge_and_generate(state: AgentState) -> dict:
    """Combine KB + DB context and generate the recommendation."""
    merged = "\n".join(state["kb_context"] + state["db_context"])

    # TODO: call your Bedrock LLM here, e.g.
    # response = bedrock_llm.invoke(build_prompt(state["query"], merged))
    recommendation = "<TODO: generated recommendation>"

    return {"merged_context": merged, "recommendation": recommendation}


def eval_check(state: AgentState) -> dict:
    """Score the recommendation for groundedness using Bedrock Guardrails'
    contextual grounding check (ApplyGuardrail API).

    Requires a guardrail already created with a contextualGroundingPolicyConfig,
    e.g. via bedrock.create_guardrail(...) with GROUNDING + RELEVANCE filters
    and thresholds set. This node just calls it against the live output.
    """
    bedrock_runtime = boto3.client("bedrock-runtime")

    response = bedrock_runtime.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="OUTPUT",  # we're validating the generated response
        content=[
            # the retrieved context the answer should be grounded in
            {"text": {"text": state["merged_context"], "qualifiers": ["grounding_source"]}},
            # the original user query, used for the relevance score
            {"text": {"text": state["query"], "qualifiers": ["query"]}},
            # the thing actually being scored
            {"text": {"text": state["recommendation"]}},
        ],
        outputScope="FULL",  # returns the numeric scores, not just pass/fail
    )

    grounding_score = 0.0
    relevance_score = 0.0
    action = response.get("action", "NONE")

    for assessment in response.get("assessments", []):
        grounding_policy = assessment.get("contextualGroundingPolicy", {})
        for f in grounding_policy.get("filters", []):
            if f.get("type") == "GROUNDING":
                grounding_score = f.get("score", 0.0)
            elif f.get("type") == "RELEVANCE":
                relevance_score = f.get("score", 0.0)

    # Use the lower of the two as your pass/fail signal — a response can be
    # highly "grounded" in irrelevant context and still be a bad answer.
    score = min(grounding_score, relevance_score)
    passed = action != "GUARDRAIL_INTERVENED" and score >= EVAL_THRESHOLD

    return {
        "eval_score": score,
        "eval_passed": passed,
        "eval_reason": (
            f"grounding={grounding_score:.2f} relevance={relevance_score:.2f} "
            f"action={action}"
        ),
    }


def return_answer(state: AgentState) -> dict:
    """Terminal node: eval passed, hand back the recommendation as-is."""
    return {}


def fallback(state: AgentState) -> dict:
    """Terminal node: retries exhausted, return a safe non-answer instead of
    a low-confidence recommendation."""
    return {
        "recommendation": (
            "I don't have enough grounded information to make a confident "
            "recommendation on this yet. Could you clarify the request or "
            "point me to more specific data?"
        )
    }


def increment_retry(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# 3. Conditional routing
# ---------------------------------------------------------------------------

def after_eval(state: AgentState) -> str:
    if state["eval_passed"]:
        return "return_answer"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retry"
    return "fallback"


# ---------------------------------------------------------------------------
# 4. Build the graph
# ---------------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("route_query", route_query)
graph.add_node("retrieve_kb", retrieve_kb)
graph.add_node("retrieve_db", retrieve_db)
graph.add_node("merge_and_generate", merge_and_generate)
graph.add_node("eval_check", eval_check)
graph.add_node("return_answer", return_answer)
graph.add_node("fallback", fallback)
graph.add_node("increment_retry", increment_retry)

graph.add_edge(START, "route_query")

# fan-out: both retrieval nodes run in parallel, LangGraph waits for both
# before running merge_and_generate (they share it as their only downstream node)
graph.add_edge("route_query", "retrieve_kb")
graph.add_edge("route_query", "retrieve_db")
graph.add_edge("retrieve_kb", "merge_and_generate")
graph.add_edge("retrieve_db", "merge_and_generate")

graph.add_edge("merge_and_generate", "eval_check")

graph.add_conditional_edges(
    "eval_check",
    after_eval,
    {
        "return_answer": "return_answer",
        "retry": "increment_retry",
        "fallback": "fallback",
    },
)

# retry loop: bump the counter, then go back to routing (you may instead want
# to re-route straight to retrieval with a rewritten query — adjust as needed)
graph.add_edge("increment_retry", "route_query")

graph.add_edge("return_answer", END)
graph.add_edge("fallback", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 5. Run it
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = app.invoke({
        "query": "What plan should I upgrade to based on my usage history?",
        "route": "both",
        "kb_context": [],
        "db_context": [],
        "merged_context": "",
        "recommendation": "",
        "eval_score": 0.0,
        "eval_passed": False,
        "eval_reason": "",
        "retry_count": 0,
    })
    print(result["recommendation"])
