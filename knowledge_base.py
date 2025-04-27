import json
import os
import wikipedia

knowledge_file = "knowledge_base.json"

# Başlangıç için temel bilgiler
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
        wikipedia.set_lang("tr")
        search_query = query.replace(" ", "+") + "+C# site:learn.microsoft.com"
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception as e:
        return None