import os
import sqlite3
import numpy as np

from foundry_local_sdk import Configuration, FoundryLocalManager


CHAT_MODEL = "qwen2.5-0.5b"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"

DATA_DIR = "data"
DATABASE_DIR = "database"
DATABASE_PATH = os.path.join(DATABASE_DIR, "language_learning.db")


def create_database():
    os.makedirs(DATABASE_DIR, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)

    connection.commit()
    return connection


def create_embedding(client, text):
    # SDK sürümüne uygun şekilde positional argument kullanımı
    response = client.generate_embedding(text)
    return response.data[0].embedding


def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def main():
    print("===================================")
    print("LANGUAGE LEARNING AI")
    print("DOCUMENT INGESTION")
    print("===================================\n")

    print("Foundry Local başlatılıyor...")

    # Güncel initialize mimarisi
    config = Configuration(app_name="language-learning-ai")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    print("Foundry Local hazır.\n")

    print("Embedding modeli seçiliyor...")
    embedding_model = manager.catalog.get_model(EMBEDDING_MODEL)

    print(f"Model: {EMBEDDING_MODEL}")
    print("Model indiriliyor/yükleniyor...")

    if not embedding_model.is_cached:
        embedding_model.download()

    embedding_model.load()

    print("Embedding modeli hazır.\n")

    embedding_client = embedding_model.get_embedding_client()
    print("Embedding client hazır.\n")

    connection = create_database()
    cursor = connection.cursor()

    cursor.execute("DELETE FROM documents")
    connection.commit()

    if not os.path.exists(DATA_DIR):
        print(f"HATA: {DATA_DIR} klasörü bulunamadı.")
        connection.close()
        return

    files = [
        filename
        for filename in os.listdir(DATA_DIR)
        if filename.endswith(".txt")
    ]

    print(f"{len(files)} dosya bulundu.\n")

    total_chunks = 0

    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        print(f"İşleniyor: {filename}")

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = split_text(text)
        print(f"  {len(chunks)} chunk oluşturuldu.")

        for index, chunk in enumerate(chunks):
            print(
                f"  Embedding oluşturuluyor "
                f"{index + 1}/{len(chunks)}..."
            )

            embedding = create_embedding(embedding_client, chunk)

            embedding_array = np.array(embedding, dtype=np.float32)

            cursor.execute(
                """
                INSERT INTO documents
                (filename, content, embedding)
                VALUES (?, ?, ?)
                """,
                (
                    filename,
                    chunk,
                    embedding_array.tobytes(),
                ),
            )

            total_chunks += 1

    connection.commit()
    connection.close()

    embedding_model.unload()

    print("\n===================================")
    print("INGESTION TAMAMLANDI")
    print("===================================")

    print(f"Toplam chunk: {total_chunks}")
    print(f"Database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()