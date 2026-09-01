# Local RAG Application (Foundry Local SDK & SQLite)

Microsoft AI Foundry Local SDK, Python ve SQLite kullanılarak geliştirilmiş **%100 yerel (offline) çalışan Retrieval-Augmented Generation (RAG)** uygulaması.

Bu proje; harici API'lere veya bulut servislerine bağımlı olmadan, yerel makine üzerinde doküman ingestion (vektörleştirme), vektör benzerlik araması (Retrieval) ve yerel dil modeli ile yanıt üretme (Generation) adımlarını uçtan uca yürütür.

---

## 🏗 Mimari ve Akış

Proje, standart RAG mimarisini 3 ana katmanda uygular:

1. **Ingestion (Veri İşleme):** `data/` klasöründeki metin dokümanları okunur, parçalara (chunk) ayrılır ve `qwen3-embedding-0.6b` modeli ile vektör haline getirilerek `knowledge.db` (SQLite) veritabanına kaydedilir.
2. **Retrieval (Arama Katmanı):** Kullanıcının sorduğu soru vektörleştirilir. Cosine Similarity (Kosinüs Benzerliği) algoritması kullanılarak veritabanındaki en alakalı `Top-K` doküman parçaları çekilir.
3. **Generation (Cevap Üretimi):** Elde edilen bağlam (context) ve kullanıcı sorusu, `qwen2.5-0.5b` yerel dil modeline yönlendirilir. Model, *System Prompt* kısıtlamalarına uyarak yalnızca verilen bağlama dayalı Türkçe yanıt üretir.

---

## 🚀 Öne Çıkan Özellikler

- **Sıfır Dış Bağımlılık:** İnternet bağlantısı olmadan tamamen yerel donanımda çalışır.
- **Hafif Vektör Depolama:** Ekstra vektör veritabanı kurulumu gerektirmeden SQLite ve JSON formatında vektör takibi.
- **Güvenli Prompt Mühendisliği:** Hallucination (uydurma) riskini önleyen kısıtlayıcı System Prompt mimarisi.
- **Etkileşimli CLI:** Terminal üzerinden anlık soru-cevap arayüzü ve debug/retrieval log desteği.

---

## 📁 Proje Yapısı

```text
yerel-rag-projesi/
│
├── data/                  # RAG için kullanılacak kaynak metin dosyaları
│   ├── doc1.txt
│   └── doc2.txt
│
├── venv/                  # Python Sanal Ortamı
├── knowledge.db           # SQLite veritabanı (Vektörler ve dokümanlar)
│
├── ingest.py              # Dokümanları vektörleştirip DB'ye kaydeden betik
├── query.py               # Vektör arama (Retrieval) test betiği
├── chat_test.py           # Yerel LLM bağlantı test betiği
├── main.py                # Uçtan uca RAG CLI uygulaması
└── README.md              # Proje dokümantasyonu