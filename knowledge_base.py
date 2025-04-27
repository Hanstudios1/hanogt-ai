# knowledge_base.py

import json
import os
import wikipedia

# Türkçe wikipedia kullanalım
wikipedia.set_lang("tr")

def load_knowledge():
    if os.path.exists("knowledge.json"):
        with open("knowledge.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def save_knowledge(knowledge):
    with open("knowledge.json", "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)

def chatbot_response(user_input, knowledge):
    user_input = user_input.lower()
    for question in knowledge:
        if question in user_input:
            return knowledge[question]
    return None

def learn_from_web(query):
    try:
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except Exception:
        return None