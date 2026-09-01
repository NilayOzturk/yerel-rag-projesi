import os
import random
import streamlit as st
from foundry_local_sdk import Configuration, FoundryLocalManager
from rag import search_documents

CHAT_MODEL = "qwen2.5-1.5b"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"


@st.cache_resource
def load_models():
    try:
        manager = FoundryLocalManager.instance
        if manager is None:
            config = Configuration(app_name="language-learning-ai")
            FoundryLocalManager.initialize(config)
            manager = FoundryLocalManager.instance
    except Exception:
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

    return embedding_client, chat_client


@st.cache_data
def load_vocabulary():
    vocab_file = os.path.join("data", "french_a1.txt")
    pairs = []
    if os.path.exists(vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        fr = parts[0].strip()
                        tr = parts[1].strip()
                        pairs.append((fr, tr))
    return pairs


def reset_quiz_session():
    st.session_state["current_set"] = st.session_state.get("current_set", 1)
    st.session_state["question_number"] = 1
    st.session_state["score_correct"] = 0
    st.session_state["score_wrong"] = 0
    generate_new_question()


def generate_new_question():
    vocab = load_vocabulary()
    if len(vocab) < 4:
        return

    target = random.choice(vocab)
    fr_word, correct_tr = target

    wrong_candidates = [tr for fr, tr in vocab if tr != correct_tr]
    wrong_options = random.sample(
        wrong_candidates, min(3, len(wrong_candidates))
    )

    options = wrong_options + [correct_tr]
    random.shuffle(options)

    st.session_state["quiz_question"] = fr_word
    st.session_state["quiz_correct"] = correct_tr
    st.session_state["quiz_options"] = options
    st.session_state["quiz_answered"] = False
    st.session_state["quiz_selected"] = None


def generate_new_flashcard():
    vocab = load_vocabulary()
    if vocab:
        target = random.choice(vocab)
        st.session_state["flashcard_fr"] = target[0]
        st.session_state["flashcard_tr"] = target[1]
        st.session_state["flashcard_flipped"] = False


st.set_page_config(
    page_title="Language Learning AI", page_icon="🌍", layout="wide"
)

st.title("🌍 Fransızca Öğrenim Asistanı")
st.write("Yerel AI destekli dil öğrenim rehberi.")

language = st.selectbox("Seçilen Dil", ["Fransızca"])

menu = st.sidebar.selectbox(
    "Özellikler",
    ["AI Öğretmen", "Test (Quiz)", "Flashcards", "Cümle Düzeltme"],
)

# ---------------------------------------------------------
# 1. AI ÖĞRETMEN MODÜLÜ
# ---------------------------------------------------------
if menu == "AI Öğretmen":
    st.header("💬 AI Öğretmen")
    question = st.text_input("Öğrenme materyaliniz hakkında bir soru sorun:")

    if st.button("AI'ya Sor"):
        if not question.strip():
            st.warning("Lütfen bir soru girin.")
        else:
            with st.spinner("Yanıt hazırlanıyor..."):
                embedding_client, chat_client = load_models()
                results = search_documents(
                    embedding_client, question, top_k=2
                )
                context = "\n".join(result[2] for result in results)

                prompt = f"""
Sadece aşağıdaki kaynak metni kullanarak sorulan kelimenin veya cümlenin Türkçe karşılığını ver. 
Ekstra yorum yapma, paragraf yazma.

Kaynak Metin:
{context}

Soru: {question}
Cevap:
"""
                response = chat_client.complete_chat(
                    [
                        {
                            "role": "system",
                            "content": "Sen sadece verilen metinden sorunun Türkçe karşılığını aktaran bir sözlük botusun.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

                answer = response.choices[0].message.content
                st.subheader("🤖 AI")
                st.write(answer)

                with st.expander("📚 Kaynaklar"):
                    for score, filename, content in results:
                        st.write(
                            f"**{filename}** — benzerlik skoru: {score:.3f}"
                        )
                        st.write(content)

# ---------------------------------------------------------
# 2. TEST (QUIZ) MODÜLÜ
# ---------------------------------------------------------
elif menu == "Test (Quiz)":
    st.header("📝 Fransızca Kelime Testi")

    if "question_number" not in st.session_state:
        reset_quiz_session()

    q_num = st.session_state["question_number"]
    current_set = st.session_state["current_set"]

    if q_num > 20:
        correct = st.session_state["score_correct"]
        wrong = st.session_state["score_wrong"]
        success_rate = (correct / 20) * 100

        st.success(f"🎊 **Set {current_set} Tamamlandı!**")
        st.write(f"* **Doğru Cevap:** {correct}")
        st.write(f"* **Yanlış Cevap:** {wrong}")
        st.write(f"* **Başarı Oranı:** %{success_rate:.1f}")

        st.write("---")
        if st.button(f"➡️ Set {current_set + 1}'e Geç"):
            st.session_state["current_set"] += 1
            reset_quiz_session()
            st.rerun()

    else:
        st.caption(f"**Set {current_set}** | Soru {q_num} / 20")
        st.progress(q_num / 20)

        st.subheader(
            f'🇫🇷 **"{st.session_state["quiz_question"]}"** kelimesinin Türkçe karşılığı nedir?'
        )

        options = st.session_state["quiz_options"]
        correct = st.session_state["quiz_correct"]

        for option in options:
            if st.button(
                option,
                key=f"btn_{option}",
                disabled=st.session_state["quiz_answered"],
            ):
                st.session_state["quiz_answered"] = True
                st.session_state["quiz_selected"] = option
                if option == correct:
                    st.session_state["score_correct"] += 1
                else:
                    st.session_state["score_wrong"] += 1
                st.rerun()

        if st.session_state["quiz_answered"]:
            selected = st.session_state["quiz_selected"]
            if selected == correct:
                st.success(f"🎉 **Tebrikler! Doğru cevap:** {correct}")
            else:
                st.error(
                    f"❌ **Yanlış cevap!** Doğrusu: **{correct}** olacaktı."
                )

            st.write("---")
            next_label = (
                "Sonuçları Gör"
                if q_num == 20
                else f"➡️ Soru {q_num + 1}'e Geç"
            )
            if st.button(next_label):
                st.session_state["question_number"] += 1
                if st.session_state["question_number"] <= 20:
                    generate_new_question()
                st.rerun()

# ---------------------------------------------------------
# 3. FLASHCARDS MODÜLÜ
# ---------------------------------------------------------
elif menu == "Flashcards":
    st.header("🎴 Kelime Kartları (Flashcards)")
    st.write("Kelimenin anlamını tahmin edin ve kartı çevirerek kontrol edin.")

    if "flashcard_fr" not in st.session_state:
        generate_new_flashcard()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.info(f"### 🇫🇷 Fransızca\n# **{st.session_state['flashcard_fr']}**")

        if st.session_state["flashcard_flipped"]:
            st.success(
                f"### 🇹🇷 Türkçe Anlamı\n# **{st.session_state['flashcard_tr']}**"
            )

        st.write("---")
        b_col1, b_col2 = st.columns(2)

        with b_col1:
            if st.button("🔄 Kartı Çevir", use_container_width=True):
                st.session_state["flashcard_flipped"] = (
                    not st.session_state["flashcard_flipped"]
                )
                st.rerun()

        with b_col2:
            if st.button("➡️ Sonraki Kart", use_container_width=True):
                generate_new_flashcard()
                st.rerun()

# ---------------------------------------------------------
# 4. CÜMLE DÜZELTME MODÜLÜ
# ---------------------------------------------------------
elif menu == "Cümle Düzeltme":
    st.header("✍️ Fransızca Cümle Düzeltme ve Gramer Analizi")
    st.write("Kontrol ettirmek istediğiniz Fransızca cümleyi yazın.")

    user_sentence = st.text_area(
        "Fransızca Cümle:", placeholder="Örn: Je suis suis öğrenci..."
    )

    if st.button("Cümleyi İncele"):
        if not user_sentence.strip():
            st.warning("Lütfen analiz edilecek bir cümle yazın.")
        else:
            with st.spinner("Gramer ve yazım kontrol ediliyor..."):
                _, chat_client = load_models()

                prompt = f"""
Aşağıdaki Fransızca cümleyi incele. Gramer veya yazım hatası varsa düzelt ve Türkçe olarak hatayı açıkla. 
Cümle zaten doğruysa Türkçe olarak doğru olduğunu belirt.

Girilen Cümle: {user_sentence}

Çıktı Formatı:
Düzeltilmiş Cümle: [Cümlenin doğru hali]
Açıklama: [Nelerin düzeltildiğine dair kısa Türkçe açıklama]
"""

                response = chat_client.complete_chat(
                    [
                        {
                            "role": "system",
                            "content": "Sen yardımsever ve nazik bir Fransızca dil bilgisi öğretmenisin.",
                        },
                        {"role": "user", "content": prompt},
                    ]
                )

                st.subheader("🔍 AI Analiz Sonucu")
                st.write(response.choices[0].message.content)