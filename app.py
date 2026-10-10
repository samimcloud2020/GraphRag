import os
import logging
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_neo4j import Neo4jVector, Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langgraph.graph import StateGraph, END

load_dotenv(override=True)
logging.getLogger("neo4j").setLevel(logging.CRITICAL)

URI, USER, PWD = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "0bd407b0")


# ═══════════════════════════════════════════════════════════════
# State schema
# ═══════════════════════════════════════════════════════════════
class State(TypedDict):
    question: str
    route: str              # "semantic" | "aggregation"
    docs: list              # Document objects (semantic path)
    context: str            # formatted context for the LLM
    cypher: str             # generated Cypher (aggregation path)
    answer: str             # final answer
    error: str              # error message if any node failed


# ═══════════════════════════════════════════════════════════════
# Semantic path components
# ═══════════════════════════════════════════════════════════════
RETRIEVAL_QUERY = """
WITH node AS article, score
OPTIONAL MATCH (article)-[:WRITTEN_BY]->(a:Author)
WITH article, score, collect(DISTINCT a.name)[..10] AS authors
OPTIONAL MATCH (article)-[:COVERS_TOPIC]->(t:Topic)
WITH article, score, authors, collect(DISTINCT t.name)[..5] AS topics
RETURN
    article.title AS text,
    score,
    {title: article.title, authors: authors, topics: topics} AS metadata
"""

vector_store = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url=URI, username=USER, password=PWD, database=DB,
    index_name="article_embeddings",
    text_node_property="title",
    retrieval_query=RETRIEVAL_QUERY,
)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

graph = Neo4jGraph(
    url=URI, username=USER, password=PWD, database=DB,
    driver_config={
        "max_connection_lifetime": 300,
        "keep_alive": True,
    },
)

GRAPH_QUERY = """
UNWIND $titles AS title
MATCH (start:Article {title: title})
OPTIONAL MATCH (start)-[:WRITTEN_BY|COVERS_TOPIC*1..2]-(neighbor:Article)
WHERE neighbor.title <> start.title
WITH neighbor, count(*) AS connections
ORDER BY connections DESC
LIMIT $limit
OPTIONAL MATCH (neighbor)-[:WRITTEN_BY]->(au:Author)
WITH neighbor, collect(DISTINCT au.name)[..10] AS authors
OPTIONAL MATCH (neighbor)-[:COVERS_TOPIC]->(t:Topic)
RETURN neighbor.title AS title, authors, collect(DISTINCT t.name)[..5] AS topics
"""


class GraphRetriever(BaseRetriever):
    store: object
    db_graph: object
    start_k: int = 3
    max_results: int = 20

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        titles = [d.page_content for d in self.store.similarity_search(query, k=self.start_k)]
        if not titles:
            return []
        rows = self.db_graph.query(
            GRAPH_QUERY, params={"titles": titles, "limit": self.max_results}
        )
        return [
            Document(
                page_content=r["title"],
                metadata={"authors": r["authors"], "topics": r["topics"]},
            )
            for r in rows
        ]


ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, GraphRetriever(store=vector_store, db_graph=graph)],
    weights=[0.7, 0.3],
)

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever,
)


# ═══════════════════════════════════════════════════════════════
# Aggregation path components
# ═══════════════════════════════════════════════════════════════
CYPHER_PROMPT = PromptTemplate.from_template("""
Task: Generate a Cypher query to answer the question.

Schema:
{schema}

Critical rules:
- (:Article)-[:WRITTEN_BY]->(:Author)   direction: Article → Author
- (:Article)-[:COVERS_TOPIC]->(:Topic)  direction: Article → Topic
- To join authors and topics, use TWO MATCH clauses from the same Article.
- Never use size() on a pattern. Use COUNT {{ (a)-[:REL]->(:Label) }}.

Examples:

Q: How many articles did Alice Smith write?
A: MATCH (a:Article)-[:WRITTEN_BY]->(au:Author {{name: 'Alice Smith'}})
   RETURN COUNT(DISTINCT a) AS count

Q: Count topics under author Bob Jones
A: MATCH (a:Article)-[:WRITTEN_BY]->(au:Author {{name: 'Bob Jones'}})
   MATCH (a)-[:COVERS_TOPIC]->(t:Topic)
   RETURN t.name AS topic, COUNT(DISTINCT a) AS count
   ORDER BY count DESC

Q: Which authors work on Graph RAG?
A: MATCH (a:Article)-[:COVERS_TOPIC]->(t:Topic {{name: 'Graph RAG'}})
   MATCH (a)-[:WRITTEN_BY]->(au:Author)
   RETURN DISTINCT au.name

Q: How many articles per topic?
A: MATCH (a:Article)-[:COVERS_TOPIC]->(t:Topic)
   RETURN t.name AS topic, COUNT(a) AS count
   ORDER BY count DESC

Q: What is the average number of authors?
A: MATCH (a:Article)
   WITH a, COUNT {{ (a)-[:WRITTEN_BY]->(:Author) }} AS author_count
   RETURN AVG(author_count) AS average_authors_per_article

Q: Find articles by Bob Jones
A: MATCH (a:Article)-[:WRITTEN_BY]->(au:Author {{name: 'Bob Jones'}})
   RETURN a.title

Question: {question}
Cypher:""")

QA_PROMPT = PromptTemplate.from_template("""
You are answering a question using the result of a Neo4j Cypher query.

If the result is a list with a single row and a single value (for example
{{count: 20}}), return that value in a natural sentence.

If the result is a list of rows with multiple fields, present them as a
table or bullet list, one row per line.

If the result is empty, say "No results found."

Never make up values. Use only what is in the result.

Question: {question}
Cypher result: {context}

Answer:""")

cypher_chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    graph=graph,
    cypher_prompt=CYPHER_PROMPT,
    qa_prompt=QA_PROMPT,
    verbose=False,
    allow_dangerous_requests=True,   # required by the class; safety comes from reader role
    return_intermediate_steps=True,
)

SEMANTIC_PROMPT = ChatPromptTemplate.from_template(
    "Answer using ONLY the context below.\n"
    "Enumerate ALL rows. Use canonical names.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)
semantic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
semantic_chain = SEMANTIC_PROMPT | semantic_llm | StrOutputParser()


# ═══════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════
AGGREGATION_WORDS = (
    "how many", "count", "number of", "total", "average",
    "how much", "sum", "how often", " by ", "written by",
)

def classify(state: State) -> State:
    q = state["question"].lower()
    state["route"] = "aggregation" if any(w in q for w in AGGREGATION_WORDS) else "semantic"
    return state


def semantic_retrieve(state: State) -> State:
    try:
        state["docs"] = retriever.invoke(state["question"])
        state["context"] = "\n".join(
            f"- {d.page_content} | Authors: {', '.join(d.metadata.get('authors', [])) or '—'} "
            f"| Topics: {', '.join(d.metadata.get('topics', [])) or '—'}"
            for d in state["docs"]
        )
        state["error"] = ""
    except Exception as e:
        state["docs"] = []
        state["error"] = f"Semantic retrieval failed: {e}"
    return state


def cypher_generate(state: State) -> State:
    try:
        result = cypher_chain.invoke({"query": state["question"]})
        state["cypher"] = result["intermediate_steps"][0]["query"]
        state["answer"] = result["result"]
        state["error"] = ""
    except Exception as e:
        state["cypher"] = ""
        state["error"] = f"Cypher generation failed: {e}"
    return state


def respond_semantic(state: State) -> State:
    if state.get("error"):
        state["answer"] = state["error"]
        return state
    state["answer"] = semantic_chain.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    return state


def respond_aggregation(state: State) -> State:
    if state.get("error"):
        state["answer"] = state["error"]
    return state


# ═══════════════════════════════════════════════════════════════
# Edges
# ═══════════════════════════════════════════════════════════════
def route_after_classify(state: State) -> Literal["semantic_retrieve", "cypher_generate"]:
    return "cypher_generate" if state["route"] == "aggregation" else "semantic_retrieve"


# ═══════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════
wf = StateGraph(State)
wf.add_node("classify", classify)
wf.add_node("semantic_retrieve", semantic_retrieve)
wf.add_node("cypher_generate", cypher_generate)
wf.add_node("respond_semantic", respond_semantic)
wf.add_node("respond_aggregation", respond_aggregation)

wf.set_entry_point("classify")
wf.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "semantic_retrieve": "semantic_retrieve",
        "cypher_generate": "cypher_generate",
    },
)
wf.add_edge("semantic_retrieve", "respond_semantic")
wf.add_edge("cypher_generate", "respond_aggregation")
wf.add_edge("respond_semantic", END)
wf.add_edge("respond_aggregation", END)

app = wf.compile()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Graph RAG — LangGraph state machine. 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            q = input("❓ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting.")
            break

        if q.lower() in ("quit", "exit", "q"):
            print("👋 Exiting.")
            break
        if not q:
            continue

        state: State = {
            "question": q,
            "route": "",
            "docs": [],
            "context": "",
            "cypher": "",
            "answer": "",
            "error": "",
        }

        try:
            out = app.invoke(state)

            print(f"\n🛣️  Route: {out['route']}")

            if out["docs"]:
                print(f"\n📄 Retrieved {len(out['docs'])} documents:")
                for d in out["docs"]:
                    authors = ", ".join(d.metadata.get("authors", [])) or "—"
                    topics = ", ".join(d.metadata.get("topics", [])) or "—"
                    print(f"   • {d.page_content}")
                    print(f"     Authors: {authors}")
                    print(f"     Topics : {topics}")

            if out["cypher"]:
                print(f"\n🔧 Generated Cypher:\n{out['cypher']}")

            print(f"\n💡 {out['answer']}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")
