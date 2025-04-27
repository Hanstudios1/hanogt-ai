import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
from knowledge_base import load_knowledge, chatbot_response
import os
import time
import json
import random

# --- Yardımcı Fonksiyonlar ---

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_to_microphone():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("Dinliyorum...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return None

def learn_from_web(query):
    try:
        wikipedia.set_lang("tr")
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except:
        try:
            search_query = query.replace(" ", "+")
            url = f"https://www.google.com/search?q={search_query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.find_all('a', href=True)
            for link in links:
                href = link['href']
                if "url?q=" in href:
                    real_link = href.split("url?q=")[1].split("&")[0]
                    return f"Daha fazla bilgi için bu kaynağa bakabilirsin: {real_link}"
            return None
        except:
            return None

def save_chat_history(history):
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False)

def load_chat_history():
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return []

# --- Yaratıcı Fonksiyonlar ---

def creative_response(prompt):
    styles = [
        "Bunu düşündüğümde aklıma gelen ilk şey şudur: {}",
        "Belki de şöyle hayal edebiliriz: {}",
        "Eğer bunu bir hikaye gibi düşünürsek: {}",
        "Bu konuda şunu söyleyebilirim: {}",
        "Bence {} olabilir."
    ]
    comment = random.choice(styles).format(generate_new_idea(prompt))
    return comment

def generate_new_idea(seed):
    topics = ["zaman yolculuğu", "mikro evrenler", "dijital rüyalar", "ışık hızında düşünce", "ses dalgalarıyla iletişim"]
    verbs = ["oluşturur", "dönüştürür", "yok eder", "yeniden inşa eder", "hızlandırır"]
    seed = seed.lower()

    idea = f"{seed} {random.choice(verbs)} ve {random.choice(topics)} ile birleşir."
    return idea.capitalize()

def advanced_word_generator(base_word):
    vowels = "aeiouüöı"
    consonants = "bcçdfgğhjklmnprsştvyz"
    mutation = ''.join(random.choice(consonants + vowels) for _ in range(3))
    suffixes = ["sal", "vari", "matik", "nistik", "gen", "goloji", "nomi"]

    new_word = base_word[:len(base_word)//2] + mutation + random.choice(suffixes)
    return new_word.capitalize()

# --- Sayfa Ayarları ---
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:", layout="centered")

# --- CSS ve Temalar ---
theme = st.sidebar.selectbox("Tema Seçin", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
        <style>
        body, .block-container {
            background-color: #0e1117;
            color: white;
        }
        .stButton>button {
            background-color: #262730;
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #262730;
            color: white;
        }
        .css-1v3fvcr {
            background-color: #0e1117;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .block-container {
            background-color: #ffffff;
            color: black;
        }
        .stButton>button {
            background-color: #f0f2f6;
            color: black;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: black;
        }
        .css-1v3fvcr {
            background-color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)

# --- Avatar ve Başlık ---
st.markdown("""
    <style>
    .avatar-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
    }
    .avatar {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        animation: pulse 2s infinite;
        object-fit: cover;
    }
    @keyframes pulse {
        0% { transform: rotate(0deg) scale(1); box-shadow: 0 0 0px rgba(255,255,255,0.4); }
        50% { transform: rotate(180deg) scale(1.05); box-shadow: 0 0 30px rgba(255,255,255,0.7); }
        100% { transform: rotate(360deg) scale(1); box-shadow: 0 0 0px rgba(255,255,255,0.4); }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="avatar-container">
    <img src="https://i.imgur.com/NySv35d.png" class="avatar" alt="AI Avatar">
    <h1 style="color:#FF4B4B;">Hanogt AI</h1>
</div>
""", unsafe_allow_html=True)

# --- Uygulama ---
knowledge = load_knowledge()
chat_history = load_chat_history()

st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Mod Seçin:", ["Sohbet Botu", "Sesli Sohbet", "Yaratıcı Mod"])

# --- Sohbet Botu ---
if app_mode == "Sohbet Botu":
    st.header("Yazılı Sohbet")

    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
        with st.spinner('Hanogt AI cevap üretiyor...'):
            result = chatbot_response(user_input, knowledge)

        if isinstance(result, str) and result.strip() != "":
            st.success(f"Hanogt AI: {result}")
            chat_history.append(("Sen", user_input))
            chat_history.append(("Hanogt AI", result))
        else:
            wiki_result = learn_from_web(user_input)
            if wiki_result:
                st.success(f"Hanogt AI (Wikipedia'dan öğrendi): {wiki_result}")
                chat_history.append(("Sen", user_input))
                chat_history.append(("Hanogt AI (Wikipedia'dan)", wiki_result))
            else:
                st.error("Üzgünüm, bu konuda bilgi bulamadım.")

        save_chat_history(chat_history)

    if chat_history:
        st.subheader("Geçmiş Konuşmalar:")
        for sender, message in chat_history:
            st.write(f"**{sender}:** {message}")

# --- Sesli Sohbet ---
elif app_mode == "Sesli Sohbet":
    st.header("Sesli Sohbet")

    if st.button("Konuşmaya Başla"):
        user_text = listen_to_microphone()

        if user_text:
            st.write(f"Sen: {user_text}")

            with st.spinner('Hanogt AI cevap üretiyor...'):
                result = chatbot_response(user_text, knowledge)

            if isinstance(result, str) and result.strip() != "":
                st.success(f"Hanogt AI: {result}")
                chat_history.append(("Sen", user_text))
                chat_history.append(("Hanogt AI", result))
            else:
                wiki_result = learn_from_web(user_text)
                if wiki_result:
                    st.success(f"Hanogt AI (Wikipedia'dan öğrendi): {wiki_result}")
                    chat_history.append(("Sen", user_text))
                    chat_history.append(("Hanogt AI (Wikipedia'dan)", wiki_result))
                else:
                    st.error("Bu konuda bir şey bulamadım.")

            save_chat_history(chat_history)

        else:
            st.error("Sesi anlayamadım. Lütfen tekrar deneyin.")

    if chat_history:
        st.subheader("Geçmiş Konuşmalar:")
        for sender, message in chat_history:
            st.write(f"**{sender}:** {message}")

# --- Yaratıcı Mod ---
elif app_mode == "Yaratıcı Mod":
    st.header("Yaratıcı Mod")

    creative_prompt = st.text_input("Bir kelime veya fikir gir:", key="creative_input")
    
    if creative_prompt:
        with st.spinner('Yaratıcı cevap üretiliyor...'):
            creative_result = creative_response(creative_prompt)
            new_word = advanced_word_generator(creative_prompt)

        st.success(f"**Yaratıcı Cevap:** {creative_result}")
        st.info(f"**Yeni Kelime:** {new_word}")