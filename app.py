import streamlit as st
import requests
from bs4 import BeautifulSoup
import wikipedia
from knowledge_base import load_knowledge, save_knowledge, chatbot_response

# Streamlit ayarları
st.set_page_config(page_title="Hanogt AI", page_icon=":robot_face:")
st.title("Hanogt AI - Öğrenebilen Yapay Zeka")

st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Mod Seçin:", ["Sohbet Botu"])

# Bilgi veritabanını yükle
knowledge = load_knowledge()

# Web'den bilgi alma fonksiyonu
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

# Ana uygulama modu
if app_mode == "Sohbet Botu":
    st.header("Sohbet Botu (Yazılı)")

    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
        result = chatbot_response(user_input, knowledge)

        if isinstance(result, str):
            st.write("Hanogt AI:", result)
        elif isinstance(result, list):
            st.warning("Bu soruyu tam anlayamadım. Şunları mı demek istedin?")
            for suggestion in result:
                st.info(f"- {suggestion}")
            st.info("İstersen bu konuda Web'den araştırabilirim.")

            if st.button("Web'den Öğren"):
                web_info = learn_from_web(user_input)
                if web_info:
                    st.info(f"Web'den öğrendiğim bilgi: {web_info}")
                    if st.button("Bu bilgiyi kaydet"):
                        knowledge[user_input.lower()] = web_info
                        save_knowledge(knowledge)
                        st.success("Bilgi kaydedildi!")
                else:
                    st.error("Üzgünüm, internette de bulamadım. Bana doğrudan öğretebilirsin.")
                    new_response = st.text_input("Bu soruya ne cevap vermeliyim?", key="teach_input")
                    if st.button("Öğret"):
                        if new_response:
                            knowledge[user_input.lower()] = new_response
                            save_knowledge(knowledge)
                            st.success("Teşekkürler! Bunu öğrendim.")
                        else:
                            st.error("Lütfen bir cevap girin.")
        else:
            st.error("Üzgünüm, hiç bilgi bulamadım. Bana öğretebilirsin.")