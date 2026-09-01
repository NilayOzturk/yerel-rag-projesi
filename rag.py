import sqlite3
import numpy as np

from foundry_local_sdk import Configuration, FoundryLocalManager


CHAT_MODEL = "qwen2.5-1.5b"  # Çok daha mantıklı ve akıcı yanıtlar verir
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

DATABASE_PATH = "database/language_learning.db"


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0

    return np.dot(a, b) / denominator


def search_documents(embedding_client, query, top_k=3):
    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT filename, content, embedding
        FROM documents
        """
    )

    rows = cursor.fetchall()
    connection.close()

    results = []

    for filename, content, embedding_blob in rows:
        document_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        score = cosine_similarity(query_embedding, document_embedding)

        results.append((score, filename, content))

    results.sort(key=lambda x: x[0], reverse=True)

    return results[:top_k]


def main():
    print("Foundry Local başlatılıyor...")

    # SDK başlatma
    config = Configuration(app_name="language-learning-ai")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)
    chat_model = manager.catalog.get_model(CHAT_MODEL)

    if not embedding_model.is_cached:
        embedding_model.download()

    if not chat_model.is_cached:
        chat_model.download()

    embedding_model.load()
    chat_model.load()

    embedding_client = embedding_model.get_embedding_client()
    chat_client = chat_model.get_chat_client()

    while True:
        print("\n-----------------------------")

        question = input("Sorunuz (çıkmak için 'q'): ")

        if question.lower() == "q":
            break

        results = search_documents(embedding_client, question, top_k=3)

        context_parts = []

        print("\nBulunan kaynaklar:")

        for score, filename, content in results:
            print(f"- {filename} (score={score:.3f})")
            context_parts.append(content)

        context = "\n\n".join(context_parts)

        # Modele verilen strict (sıkı) Türkçe prompt
        prompt = f"""
Sen Türkçe konuşan bir dil öğretmenisin. 
Aşağıdaki öğrenme materyalini kullanarak kullanıcının sorusunu yanıtla.

Öğrenme materyali:
{context}

Kullanıcı sorusu:
{question}

Kurallar:
1. Yanıtı HER ZAMAN Türkçe olarak ver.
2. Sadece sağlanan öğrenme materyalindeki bilgilere dayanarak cevap ver.
3. Eğer materyalde sorunun cevabı tam olarak yoksa strictly "Aradığınız bilgi verilen öğrenme materyalinde bulunmamaktadır." yaz.
4. Bilgi uydurma veya metin dışına çıkma.
5.Cevabın içine soruyu tekrar yazma.
"""

        response = chat_client.complete_chat(
            [
                {
                    "role": "system",
                    "content": "Sen sadece verilen öğrenme materyaline sadık kalan ve kesinlikle Türkçe cevap veren bir dil öğretmenisin.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        answer = response.choices[0].message.content

        print("\nAI:")
        print(answer)

    embedding_model.unload()
    chat_model.unload()

    print("\nProgram kapatıldı.")


if __name__ == "__main__":
    main()