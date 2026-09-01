import sqlite3
DB_PATH = "knowledge.db"
def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute(
    '''
    CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL
    )
    '''
    )
    conn.commit()
    conn.close()
    print("Veritabanı hazır.")
if __name__ == "__main__":
    main()
