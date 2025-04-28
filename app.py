# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
import random
from knowledge_base import load_knowledge, chatbot_response
import os
import json
from PIL import Image, ImageDraw
from io import BytesIO

# Sayfa yapılandırması
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:", layout="wide")

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

# --- Bilgileri Yükle ---
knowledge = load_knowledge()
chat_history = load_chat_history()

# --- Sidebar Menü ---
st.sidebar.title("Hanogt AI")
st.sidebar.markdown("### Akıllı Yardımcı & Yaratıcı Modlar")
app_mode = st.sidebar.radio("Mod Seçin:", ["Yazılı Sohbet", "Sesli Sohbet", "Yaratıcı Mod", "Görsel Üretici"])

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Hanogt AI")

# --- Ana Uygulama ---
if app_mode == "Yazılı Sohbet":
    st.title("🧠 Hanogt AI - Yazılı Sohbet")

    # Önce geçmiş konuşmalar göster
    if chat_history:
        st.subheader("Geçmiş Konuşmalar")
        for sender, message in chat_history:
            if sender == "Sen":
                st.markdown(f"**{sender}:** {message}", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 8px; border-radius: 8px;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)

    # Yeni mesaj kutusu
    st.subheader("Yeni Mesaj")
    user_input = st.text_input("Bir şeyler yaz...", key="chat_input")

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
                    st.error("Üzgünüm, cevap bulamadım.")

            save_chat_history(chat_history)

elif app_mode == "Sesli Sohbet":
    st.title("🎤 Hanogt AI - Sesli Sohbet")

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

elif app_mode == "Yaratıcı Mod":
    st.title("✨ Hanogt AI - Yaratıcı Mod")

    creative_prompt = st.text_input("Hayal gücünü serbest bırak:", key="creative_input")

    if creative_prompt:
        creative_text = creative_response(creative_prompt)
        st.success(creative_text)

        new_word = advanced_word_generator(creative_prompt)
        st.info(f"Yeni bir kelime icat ettim: **{new_word}**")

elif app_mode == "Görsel Üretici":
    st.title("🖼️ Hanogt AI - Görsel Üretici")

    image_prompt = st.text_input("Ne çizmeli?", key="image_input")

    if st.button("Görsel Üret"):
        if image_prompt:
            image = generate_fake_image(image_prompt)
            st.image(image, caption=f"Hanogt AI - {image_prompt}", use_container_width=True)
        else:
            st.error("Lütfen bir açıklama girin!")