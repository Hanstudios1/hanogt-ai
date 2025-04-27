import os
import json
import wikipedia
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from tinydb import TinyDB
import speech_recognition as sr
import pyttsx3
from langchain.llms import OpenAI

# --------------------
# Dosyalar
knowledge_file = "knowledge.json"
db = TinyDB(knowledge_file)

# --------------------
# Varsayılan Bilgiler
default_knowledge = {
    "merhaba": "Merhaba! Sana nasıl yardımcı olabilirim?",
    "nasılsın": "İyiyim, teşekkür ederim! Sen nasılsın?",
    "selam": "Selam! Bugün sana nasıl yardımcı olabilirim?",
    "günaydın": "Günaydın! Harika bir gün seni bekliyor!",
    "iyi akşamlar": "İyi akşamlar! Umarım günün güzel geçmiştir.",
    "ne yapıyorsun": "Seninle sohbet ediyorum ve öğreniyorum!",
    "adın ne": "Benim adım Hanogt AI!",
    "seni kim yaptı": "Beni Hanogt tarafından geliştirildim!",
    "c# dilinde sınıf nasıl tanımlanır": "C# dilinde sınıf `class` anahtar kelimesiyle tanımlanır. Örnek: `class MyClass { }`",
    "c# dilinde interface nasıl tanımlanır": "C# dilinde interface `interface` anahtar kelimesi ile tanımlanır. Örnek: `interface IExample { void Method(); }`",
    "c# dilinde for döngüsü nasıl yazılır": "`for (int i = 0; i < 10; i++) { Console.WriteLine(i); }` şeklinde yazılır."
}

# --------------------
# Başlangıçta Yükleme
def load_knowledge():
    if len(db) == 0:
        save_knowledge(default_knowledge)
    data = db.all()
    return {item['question']: item['answer'] for item in data}

def save_knowledge(knowledge):
    db.truncate()
    for question, answer in knowledge.items():
        db.insert({'question': question, 'answer': answer})

# --------------------
# NLP: Anlamlı Soru Bulma
# Modeli sadece bir kere yükleyelim
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def find_best_match(user_input, knowledge, threshold=0.7):
    if not knowledge:
        return None, None

    sentences = list(knowledge.keys())
    embeddings = model.encode(sentences, convert_to_tensor=True)
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embedding, embeddings)

    best_score, best_idx = cosine_scores.max(), cosine_scores.argmax()

    if best_score > threshold:
        return sentences[best_idx], None
    else:
        # Eğer yeterli skor yoksa en yakın 3 öneriyi getir
        top_results = cosine_scores.squeeze().topk(3)
        suggestions = [sentences[i] for i in top_results.indices]
        return None, suggestions

# --------------------
# Chatbot Cevabı
def chatbot_response(user_input, knowledge):
    match, suggestions = find_best_match(user_input, knowledge)
    if match:
        return knowledge[match]
    else:
        return suggestions  # Artık öneri listesi döner

# --------------------
# Web'den Öğrenme
def learn_from_web(query):
    try:
        wikipedia.set_lang("tr")
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception:
        try:
            search_query = query.replace(" ", "+") + "+site:learn.microsoft.com"
            google_url = f"https://www.google.com/search?q={search_query}"

            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(google_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select(".tF2Cxc a")

            if links:
                first_link = links[0]['href']
                return f"Daha fazla bilgi için buraya bakabilirsin: {first_link}"
            else:
                return None
        except Exception:
            return None

# --------------------
# Özetleyici
def summarize_web_info(info):
    try:
        llm = OpenAI(temperature=0.5, max_tokens=200)
        summary = llm(f"Şunu Türkçe özetle ve doğal bir cümle haline getir: {info}")
        return summary
    except Exception:
        return info

# --------------------
# Sesli Asistan
def listen_to_microphone():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Dinliyorum...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()