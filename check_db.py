import json
import sqlite3
conn = sqlite3.connect("knowledge.db")
cur = conn.cursor()
cur.execute(
    "SELECT id, source, content, embedding FROM documents"
)
rows = cur.fetchall()
print("Kayıt sayısı:", len(rows))
for row in rows[:3]:
    record_id, source, content, embedding_json = row
    vector = json.loads(embedding_json)
    print("\nID:", record_id)
    print("Source:", source)
    print("Content:", content[:100])
    print("Embedding boyutu:", len(vector))
conn.close()
