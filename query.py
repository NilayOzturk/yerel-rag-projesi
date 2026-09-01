import json
import math
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "knowledge.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_top_chunks(query, top_k=3):
    config = Configuration(app_name="local_rag_query")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model = manager.catalog.get_model(EMBEDDING_MODEL)

    print("Model kontrol ediliyor...")
    model.download(
        lambda p: print(f"\rModel durumu: %{p:.1f}", end="", flush=True)
    )
    print("\nModel yükleniyor...")
    model.load()

    client = model.get_embedding_client()

    response = client.generate_embedding(query)
    query_embedding = response.data[0].embedding

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT source, content, embedding FROM documents")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for source, content, embedding_json in rows:
        document_embedding = json.loads(embedding_json)
        score = cosine_similarity(
            query_embedding,
            document_embedding,
        )
        scored.append((score, source, content))

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    model.unload()
    return scored[:top_k]


if __name__ == "__main__":
    results = get_top_chunks("Python hakkında bilgi ver")
    for score, source, content in results:
        print("\nSCORE:", round(score, 4))
        print("SOURCE:", source)
        print(content)