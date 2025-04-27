import streamlit as st
import wikipedia
import json
import os

# Wikipedia dili Türkçe
wikipedia.set_lang("tr")

def load_knowledge():
    try:
        if os.path.exists("knowledge.json"):
            with open("knowledge.json", "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Bilgi veritabanı yüklenemedi: {e}")
        return {}

def save_knowledge(knowledge):
    try:
        with open("knowledge.json", "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"Bilgi veritabanı kaydedilemedi: {e}")

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
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Birden fazla sonuç bulundu: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Wikipedia'da böyle bir sayfa bulunamadı."
    except Exception as e:
        return f"Hata oluştu: {e}"