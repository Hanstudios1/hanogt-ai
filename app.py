# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
import random
import os
import json
import time
from PIL import Image, ImageDraw
from io import BytesIO
from knowledge_base import load_knowledge, chatbot_response

# Sayfa yapılandırması
st.set_page_config(page_title="Hanogt AI", page_icon="https://i.imgur.com/NySv35d.png", layout="wide")

# --- Yükleniyor Animasyonu ---
@st.cache_resource
def load_logo():
    return "https://i.imgur.com/NySv35d.png"

with st.spinner('Yapay Zekâ Yükleniyor...'):
    time.sleep(2)  # 2 saniye beklet

logo_url = load_logo()

# --- Üstte Logo ve Başlık ---
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="{logo_url}" width="120"/>
        <h1 style='color: #4CAF50; font-family: Arial;'>Hanogt AI</h1>
        <h4>Akıllı Asistan & Yaratıcı Yapay Zekâ</h4>
    </div>
    <hr style='border:1px solid #eee;'/>
    """,
    unsafe_allow_html=True
)

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
                    return f"Daha fazla bilgi için: {real_link}"
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
        "Bunu düşündüğümde aklıma gelen şey: {}",
        "Şöyle hayal edebiliriz: {}",
        "Bir hikaye gibi düşünürsek: {}",
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

# --- Görsel Üretici ---
def generate_fake_image(prompt):
    img = Image.new('RGB', (512, 512), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    d = ImageDraw.Draw(img)
    d.text((20, 250), prompt, fill=(255, 255, 255))
    return img

# --- Bilgi Yükle ---
knowledge = load_knowledge()
chat_history = load_chat_history()

# --- Sidebar Menü ---
st.sidebar.title("Menü")
app_mode = st.sidebar.radio("Bir Mod Seçin:", ["🧠 Yazılı Sohbet", "🎤 Sesli Sohbet", "✨ Yaratıcı Mod", "🖼️ Görsel Üretici"])
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Hanogt AI")

# --- Ana İçerik ---
if app_mode == "🧠 Yazılı Sohbet":
    st.subheader("Yazılı Sohbet")

    if chat_history:
        st.markdown("### Geçmiş Mesajlar")
        for sender, message in chat_history:
            if sender == "Sen":
                st.markdown(f"**{sender}:** {message}")
            else:
                st.markdown(f"<div style='background-color: #f7f9fa; padding: 8px; border-radius: 8px;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)

    user_input = st.text_input("Yeni Mesajınız", key="chat_input")

    if st.button("Gönder"):
        if user_input:
            response = chatbot_response(user_input, knowledge)

            if isinstance(response, str) and response.strip() != "":
                st.success(f"Hanogt AI: {response}")
                chat_history.append(("Sen", user_input))
                chat_history.append(("Hanogt AI", response))
            else:
                wiki_result = learn_from_web(user_input)
                if wiki_result:
                    st.success(f"Wikipedia'dan öğrendim: {wiki_result}")
                    chat_history.append(("Sen", user_input))
                    chat_history.append(("Hanogt AI (Wikipedia'dan)", wiki_result))
                else:
                    st.error("Üzgünüm, bir cevap bulamadım.")
            save_chat_history(chat_history)

elif app_mode == "🎤 Sesli Sohbet":
    st.subheader("Sesli Sohbet")

    if st.button("Konuşmaya Başla"):
        user_text = listen_to_microphone()

        if user_text:
            st.write(f"Sen: {user_text}")
            response = chatbot_response(user_text, knowledge)

            if isinstance(response, str) and response.strip() != "":
                st.success(f"Hanogt AI: {response}")
                speak(response)
                chat_history.append(("Sen", user_text))
                chat_history.append(("Hanogt AI", response))
            else:
                wiki_result = learn_from_web(user_text)
                if wiki_result:
                    st.success(f"Wikipedia'dan öğrendim: {wiki_result}")
                    speak(wiki_result)
                    chat_history.append(("Sen", user_text))
                    chat_history.append(("Hanogt AI (Wikipedia'dan)", wiki_result))
                else:
                    st.error("Bilgi bulamadım.")
            save_chat_history(chat_history)

elif app_mode == "✨ Yaratıcı Mod":
    st.subheader("Yaratıcı Mod")

    creative_prompt = st.text_input("Bir fikir üret:", key="creative_input")

    if creative_prompt:
        creative_text = creative_response(creative_prompt)
        st.success(creative_text)

        new_word = advanced_word_generator(creative_prompt)
        st.info(f"Yeni bir kelime: **{new_word}**")

elif app_mode == "🖼️ Görsel Üretici":
    st.subheader("Görsel Üretici")

    image_prompt = st.text_input("Çizilecek şey:", key="image_input")

    if st.button("Görsel Üret"):
        if image_prompt:
            image = generate_fake_image(image_prompt)
            st.image(image, caption=f"Hanogt AI - {image_prompt}", use_container_width=True)
        else:
            st.error("Lütfen bir açıklama girin!")