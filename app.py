# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
from knowledge_base import load_knowledge, save_knowledge, chatbot_response

st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:")
st.title("Hanogt AI - Öğrenebilen Yapay Zeka")

st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Mod Seçin:", ["Sohbet Botu"])

knowledge = load_knowledge()

def learn_from_web(query):
    try:
        # Önce Wikipedia'dan bilgi arıyoruz
        summary = wikipedia.summary(query, sentences=2)
        return summary
    except:
        try:
            # Wikipedia'da bulamazsak, Microsoft Docs üzerinden C# için arıyoruz
            search_query = query.replace(" ", "+") + "+C# site:learn.microsoft.com"
            url = f"https://www.google.com/search?q={search_query}"

            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # Google sonuçlarından ilk linki alıyoruz
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
    st.header("Sohbet Botu (Öğrenebilen)")

    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
        response = chatbot_response(user_input, knowledge)

        if response:
            st.write("Hanogt AI:", response)
        else:
            st.warning("Bu konuda bilgim yok. Web'den araştırayım mı?")
            if st.button("Web'den Öğren"):
                web_info = learn_from_web(user_input)
                if web_info:
                    st.info(f"Web'den öğrendiğim bilgi: {web_info}")
                    if st.button("Bu bilgiyi kaydet"):
                        knowledge[user_input.lower()] = web_info
                        save_knowledge(knowledge)
                        st.success("Bilgi kaydedildi!")
                else:
                    st.error("Üzgünüm, internette de bulamadım. Bana öğretebilirsin.")
                    new_response = st.text_input("Bu soruya ne cevap vermeliyim?", key="teach_input")
                    if st.button("Öğret"):
                        if new_response:
                            knowledge[user_input.lower()] = new_response
                            save_knowledge(knowledge)
                            st.success("Teşekkürler! Bunu öğrendim.")
                        else:
                            st.error("Lütfen bir cevap girin.")