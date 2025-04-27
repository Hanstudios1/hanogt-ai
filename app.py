import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
import speech_recognition as sr
import pyttsx3
import random
from knowledge_base import load_knowledge, chatbot_response
import os
import time
import json
from PIL import Image
import base64
from io import BytesIO

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

# --- Görsel Üretici Fonksiyonu ---
def generate_fake_image(prompt):
    """Sahte bir görsel üretir (şu anda demo)."""
    img = Image.new('RGB', (512, 512), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))
    d = Image.ImageDraw.Draw(img)
    d.text((10,10), prompt, fill=(255,255,255))
    return img

# Eğer OpenAI DALL-E kullanmak istersek:
# def generate_image_dalle(prompt):
#     response = openai.Image.create(prompt=prompt, n=1, size="512x512")
#     image_url = response['data'][0]['url']
#     return image_url

# --- Streamlit Sayfa Ayarları ---
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:", layout="centered")

st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Mod Seçin:", ["Sohbet Botu", "Sesli Sohbet", "Yaratıcı Mod", "Görsel Üretici"])

# --- Bilgi Yükle ---
knowledge = load_knowledge()
chat_history = load_chat_history()

# --- Uygulamalar ---
if app_mode == "Sohbet Botu":
    st.header("Yazılı Sohbet")
    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
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
                st.error("Üzgünüm, bilgi bulamadım.")

        save_chat_history(chat_history)

elif app_mode == "Sesli Sohbet":
    st.header("Sesli Sohbet")

    if st.button("Konuşmaya Başla"):
        user_text = listen_to_microphone()

        if user_text:
            st.write(f"Sen: {user_text}")
            result = chatbot_response(user_text, knowledge)

            if isinstance(result, str) and result.strip() != "":
                st.success(f"Hanogt AI: {result}")
                speak(result)
                chat_history.append(("Sen", user_text))
                chat_history.append(("Hanogt AI", result))
            else:
                wiki_result = learn_from_web(user_text)
                if wiki_result:
                    st.success(f"Hanogt AI (Wikipedia'dan öğrendi): {wiki_result}")
                    speak(wiki_result)
                    chat_history.append(("Sen", user_text))
                    chat_history.append(("Hanogt AI (Wikipedia'dan)", wiki_result))
                else:
                    st.error("Bilgi bulamadım.")

            save_chat_history(chat_history)

elif app_mode == "Yaratıcı Mod":
    st.header("Yaratıcı Mod")
    prompt = st.text_input("Hayal gücünü serbest bırak:", key="creative_input")

    if prompt:
        creative_text = creative_response(prompt)
        st.success(creative_text)

        new_word = advanced_word_generator(prompt)
        st.info(f"Yeni bir kelime icat ettim: **{new_word}**")

elif app_mode == "Görsel Üretici":
    st.header("Görsel Üretici")
    image_prompt = st.text_input("Ne çizelim?", key="image_input")

    if st.button("Görsel Üret!"):
        if image_prompt:
            image = generate_fake_image(image_prompt)
            st.image(image, caption=f"Hanogt AI - {image_prompt}", use_column_width=True)
        else:
            st.error("Lütfen bir açıklama girin!")

# --- Geçmiş Konuşmalar ---
if chat_history and app_mode in ["Sohbet Botu", "Sesli Sohbet"]:
    st.subheader("Geçmiş Konuşmalar:")
    for sender, message in chat_history:
        st.write(f"**{sender}:** {message}")