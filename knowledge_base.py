# knowledge_base.py

import json
import os
import requests
from bs4 import BeautifulSoup

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
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Google arama sonucu özetini çekiyoruz
        answer_box = soup.find("div", class_="BNeawe s3v9rd AP7Wnd")
        if answer_box:
            return answer_box.text.strip()
        else:
            return None
    except Exception as e:
        return None