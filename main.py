import json
import math
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "knowledge.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "qwen2.5-0.5b"


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(query, embedding_client, top_k=3):
    response = embedding_client.generate_embedding(query)
    query_embedding = response.data[0].embedding
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT source, content, embedding FROM documents")
    rows = cur.fetchall()
    conn.close()

    scored = []
    for source, content, embedding_json in rows:
        document_embedding = json.loads(embedding_json)
        score = cosine_similarity(query_embedding, document_embedding)
        scored.append((score, source, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def main():
    config = Configuration(app_name="local_rag_app")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Embedding modeli hazırlanıyor...")
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
    embedding_model.download(
        lambda p: print(f"\rEmbedding Model: %{p:.1f}", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    print("Chat modeli hazırlanıyor...")
    chat_model = manager.catalog.get_model(CHAT_MODEL)
    chat_model.download(
        lambda p: print(f"\rChat Model: %{p:.1f}", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nLocal RAG hazır (Retrieval Log Aktif).")
    print('Çıkmak için "q" yaz.\n')

    while True:
        question = input("Sorunuz: ").strip()
        if question.lower() == "q":
            break
        if not question:
            print("Lütfen bir soru yaz.\n")
            continue

        results = retrieve(question, embedding_client, top_k=3)

        # 12.2 Retrieval Log Çıktısı
        print("\n--- RETRIEVED CONTEXT ---")
        for score, source, content in results:
            print("SCORE:", round(score, 4))
            print("SOURCE:", source)
            print("TEXT:", content[:250])
            print("-------------------------")

        context = "\n\n".join(
            f"[Kaynak: {source}]\n{content}" for _, source, content in results
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document Q&A assistant. "
                    "Use only the supplied context. "
                    "If the context is insufficient, "
                    "say you do not have enough information. "
                    "Do not invent facts.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]

        response = chat_client.complete_chat(messages)
        answer = response.choices[0].message.content

        print("\nAssistant:")
        print(answer)
        print("\n" + "=" * 50 + "\n")

    embedding_model.unload()
    chat_model.unload()


if __name__ == "__main__":
    main()