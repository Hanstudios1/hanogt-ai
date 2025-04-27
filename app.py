# app.py

import streamlit as st
from knowledge_base import load_knowledge, save_knowledge, chatbot_response, learn_from_web

st.title("Hanogt AI - Öğrenen Yapay Zeka")

knowledge = load_knowledge()

user_input = st.text_input("Sen:", key="chat_input")

if user_input:
    response = chatbot_response(user_input, knowledge)
    
    if response:
        st.write("Hanogt AI:", response)
    else:
        st.warning("Bu konuda bilgim yok. Web'den öğrenmek ister misin?")
        if st.button("Web'den Öğren"):
            web_info = learn_from_web(user_input)
            if web_info:
                st.info(f"Web'den Öğrendiğim Bilgi: {web_info}")
                if st.button("Bu bilgiyi kaydet"):
                    knowledge[user_input.lower()] = web_info
                    save_knowledge(knowledge)
                    st.success("Bilgi kaydedildi!")
            else:
                st.error("Web'den bilgi bulamadım. İstersen sen öğretebilirsin.")
                new_response = st.text_input("Bu soruya ne cevap vermeliyim?", key="teach_input")
                if st.button("Öğret"):
                    if new_response:
                        knowledge[user_input.lower()] = new_response
                        save_knowledge(knowledge)
                        st.success("Teşekkürler! Bunu öğrendim.")
                    else:
                        st.error("Lütfen bir cevap girin.")