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
import time

# --- Sayfa yapılandırması ---
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:", layout="wide")

# --- Yükleniyor Animasyonu ---
def show_loading_animation():
    placeholder = st.empty()

    with placeholder.container():
        st.markdown(
            """
            <div style="text-align: center; margin-top: 50px;">
                <img src="https://i.imgur.com/NySv35d.png" alt="Logo" width="150" style="animation: spin 2s linear infinite;">
            </div>

            <style>
            @keyframes spin {
                0% { transform: rotate(0deg);}
                100% { transform: rotate(360deg);}
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    time.sleep(2)
    placeholder.empty()  # Loading ekranını temizle
    st.session_state.page_loaded = True
    st.experimental_rerun()

# --- İlk Yüklemede Animasyonu Göster ---
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = False

if not st.session_state.page_loaded:
    show_loading_animation()
    st.stop()

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

def generate_fake_image(prompt):
    img = Image.new('RGB', (512, 512), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    d = ImageDraw.Draw(img)
    d.text((20, 250), prompt, fill=(255, 255, 255))
    return img

# --- Veri Yükleme ---
knowledge = load_knowledge()
chat_history = load_chat_history()

# --- Sayfa Başlığı ---
st.markdown("<h1 style='text-align: center;'>🧠 Hanogt AI</h1>", unsafe_allow_html=True)

# --- Mod Seçimi ---
st.markdown("### Mod Seçimi")
col1, col2, col3, col4 = st.columns(4)

with col1:
    yazili_buton = st.button("✏️ Yazılı Sohbet")
with col2:
    sesli_buton = st.button("🎤 Sesli Sohbet")
with col3:
    yaratıcı_buton = st.button("✨ Yaratıcı Mod")
with col4:
    gorsel_buton = st.button("🖼️ Görsel Üretici")

# --- Mod Takibi ---
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "Yazılı Sohbet"

if yazili_buton:
    st.session_state.app_mode = "Yazılı Sohbet"
elif sesli_buton:
    st.session_state.app_mode = "Sesli Sohbet"
elif yaratıcı_buton:
    st.session_state.app_mode = "Yaratıcı Mod"
elif gorsel_buton:
    st.session_state.app_mode = "Görsel Üretici"

app_mode = st.session_state.app_mode

# --- Modlara Göre İşlevler ---
if app_mode == "Yazılı Sohbet":
    st.subheader("Geçmiş Konuşmalar")

    if chat_history:
        for sender, message in chat_history:
            if sender == "Sen":
                st.markdown(f"**{sender}:** {message}", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 8px; border-radius: 8px;'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)

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
    st.subheader("Sesli Konuşma Başlat")

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
                    st.error("Bilgi bulunamadı.")
            save_chat_history(chat_history)

elif app_mode == "Yaratıcı Mod":
    st.subheader("Hayal Gücünü Serbest Bırak")

    creative_prompt = st.text_input("Bir hayal ya da fikir yazın:", key="creative_input")

    if creative_prompt:
        creative_text = creative_response(creative_prompt)
        st.success(creative_text)

        new_word = advanced_word_generator(creative_prompt)
        st.info(f"Yeni kelime: **{new_word}**")

elif app_mode == "Görsel Üretici":
    st.subheader("Görsel Üret")

    image_prompt = st.text_input("Ne çizelim?", key="image_input")

    if st.button("Görsel Üret"):
        if image_prompt:
            image = generate_fake_image(image_prompt)
            st.image(image, caption=f"Hanogt AI - {image_prompt}", use_container_width=True)
        else:
            st.error("Lütfen bir açıklama girin!")