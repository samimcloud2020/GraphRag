"""ingest.py — one-shot setup + data load."""
import os, csv
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)
URI, USER, PWD = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "0bd407b0")
CSV, VEC, DIM = "articles.csv", "article_embeddings", 1536

SETUP = [
    "CREATE CONSTRAINT article_title IF NOT EXISTS FOR (a:Article) REQUIRE a.title IS UNIQUE",
    "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE FULLTEXT INDEX ft_article IF NOT EXISTS FOR (n:Article) ON EACH [n.title]",
    f"CREATE VECTOR INDEX {VEC} IF NOT EXISTS FOR (a:Article) ON (a.embedding) "
    f"OPTIONS {{ indexConfig: {{`vector.dimensions`: {DIM}, `vector.similarity_function`: 'cosine'}} }}",
]

INGEST = """
UNWIND $rows AS row
MERGE (art:Article {title: row.title})
MERGE (au:Author   {name:  row.author_name})
MERGE (t:Topic     {name:  row.topic_name})
MERGE (art)-[:WRITTEN_BY]->(au)
MERGE (art)-[:COVERS_TOPIC]->(t)
"""

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    emb = OpenAIEmbeddings()
    with driver.session(database=DB) as s:
        for stmt in SETUP:
            s.run(stmt)
        rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
        s.run(INGEST, rows=rows)
        print(f"Ingested {len(rows)} rows")
        titles = [r["title"] for r in s.run("MATCH (a:Article) RETURN a.title AS title")]
        for t in titles:
            s.run("MATCH (a:Article {title: $t}) SET a.embedding = $v",
                  t=t, v=emb.embed_query(t))
        print(f"Embedded {len(titles)} articles")
    driver.close()

if __name__ == "__main__":
    main()
