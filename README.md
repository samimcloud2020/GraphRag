## GraphRag with Neo4j DB.

              ┌────────────────┐
              │ User Question  │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │   classify     │  keyword scan
              └───────┬────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
    semantic                     aggregation
        │                            │
        ▼                            ▼
┌───────────────────┐      ┌───────────────────┐
│ semantic_retrieve │      │  cypher_generate  │
│  ───────────────  │      │  ───────────────  │
│  vector retriever │      │  LLM → Cypher     │
│  graph retriever  │      │  validate         │
│  RRF fusion       │      │  execute          │
│  cross-encoder    │      │  LLM → answer     │
│  → top 5 docs     │      │  → answer         │
└─────────┬─────────┘      └─────────┬─────────┘
          │                          │
          ▼                          ▼
┌───────────────────┐      ┌───────────────────┐
│ respond_semantic  │      │respond_aggregation│
│  LLM → answer     │      │  pass-through     │
└─────────┬─────────┘      └─────────┬─────────┘
          │                          │
          └─────────────┬────────────┘
                        ▼
                       END
                        │
                        ▼
                  Final Answer
