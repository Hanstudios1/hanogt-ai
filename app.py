import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
from knowledge_base import load_knowledge, save_knowledge, chatbot_response
import os
import time

# Sayfa Ayarları
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:", layout="centered")

# CSS Yükleme
with open("static/loading.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Logo ve Başlık
st.markdown("""
<div class="avatar-container">
    <img src="app/static/avatar.png" class="avatar" alt="AI Avatar">
    <h1 style="color:#FF4B4B;">Hanogt AI</h1>
</div>
""", unsafe_allow_html=True)

# Tema Seçimi
theme = st.sidebar.selectbox("Tema Seçin", ["Light", "Dark"])

if theme == "Dark":
    st.markdown(
        """
        <style>
        body { background-color: #0e1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body { background-color: #ffffff; color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )

# Menü
st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Mod Seçin:", ["Sohbet Botu", "Sesli Sohbet"])

knowledge = load_knowledge()
chat_history = []

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
            search_query = query.replace(" ", "+") + "+C# site:learn.microsoft.com"
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

if app_mode == "Sohbet Botu":
    st.header("Yazılı Sohbet")

    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
        with st.spinner('Hanogt AI düşünüyor...'):
            time.sleep(1)  # loading efekti için

        result = chatbot_response(user_input, knowledge)

        if isinstance(result, str):
            st.success(f"Hanogt AI: {result}")
            chat_history.append(("Sen", user_input))
            chat_history.append(("Hanogt AI", result))
        elif isinstance(result, list):
            st.warning("Bu soruyu tam anlayamadım. Şunlardan mı bahsediyorsun?")
            for suggestion in result:
                st.info(f"- {suggestion}")
        else:
            st.error("Üzgünüm, hiç bilgi bulamadım.")

    # Chat Geçmişi
    if chat_history:
        st.subheader("Geçmiş Konuşmalar:")
        for sender, message in chat_history:
            st.write(f"**{sender}:** {message}")

elif app_mode == "Sesli Sohbet":
    st.header("Sesli Sohbet")

    if st.button("Konuşmaya Başla"):
        user_text = listen_to_microphone()

        if user_text:
            st.write(f"Sen: {user_text}")

            with st.spinner('Hanogt AI düşünüyor...'):
                time.sleep(1)

            result = chatbot_response(user_text, knowledge)

            if isinstance(result, str):
                st.success(f"Hanogt AI: {result}")
                speak(result)
                chat_history.append(("Sen", user_text))
                chat_history.append(("Hanogt AI", result))
            elif isinstance(result, list):
                st.warning("Şunlardan birini mi kastettin?")
                for suggestion in result:
                    st.info(f"- {suggestion}")
            else:
                st.error("Anlayamadım veya cevap bulamadım.")
        else:
            st.error("Sesi anlayamadım. Lütfen tekrar deneyin.")

    # Chat Geçmişi
    if chat_history:
        st.subheader("Geçmiş Konuşmalar:")
        for sender, message in chat_history:
            st.write(f"**{sender}:** {message}")