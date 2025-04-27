import json
import os
import wikipedia
import requests
from bs4 import BeautifulSoup

knowledge_file = "knowledge_base.json"

default_knowledge = {
    "merhaba": "Merhaba! Sana nasıl yardımcı olabilirim?",
    "nasılsın": "İyiyim, teşekkür ederim! Sen nasılsın?",
    "selam": "Selam! Bugün sana nasıl yardımcı olabilirim?",
    "günaydın": "Günaydın! Harika bir gün seni bekliyor!",
    "iyi akşamlar": "İyi akşamlar! Umarım günün güzel geçmiştir.",
    "ne yapıyorsun": "Seninle sohbet ediyorum ve öğreniyorum!",
    "adın ne": "Benim adım Hanogt AI!",
    "seni kim yaptı": "Beni Hanogt tarafından geliştirildim!"
}

def load_knowledge():
    if not os.path.exists(knowledge_file):
        save_knowledge(default_knowledge)
    with open(knowledge_file, "r", encoding="utf-8") as f:
        return json.load(f)

def save_knowledge(knowledge):
    with open(knowledge_file, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)

def chatbot_response(user_input, knowledge):
    user_input = user_input.lower()
    return knowledge.get(user_input)

def learn_from_web(query):
    try:
        # Önce Wikipedia'da aramaya çalış
        wikipedia.set_lang("tr")
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception:
        try:
            # Wikipedia'da bulunamadıysa Google'dan learn.microsoft.com sitesi üzerinde ara
            search_query = query.replace(" ", "+") + "+site:learn.microsoft.com"
            google_url = f"https://www.google.com/search?q={search_query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            response = requests.get(google_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select(".tF2Cxc a")

            if links:
                first_link = links[0]['href']
                return f"Daha fazla bilgi için buraya bakabilirsin: {first_link}"
            else:
                return None
        except Exception as e:
            return None