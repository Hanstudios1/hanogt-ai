# app.py

import streamlit as st
import numpy as np
from knowledge_base import load_knowledge, save_knowledge, chatbot_response
from regression_model import RegressionModel

# Başlık
st.title("Hanogt AI - Süper Yapay Zeka")

# Yan Menü
st.sidebar.title("Hanogt AI Menü")
app_mode = st.sidebar.selectbox("Bir Mod Seçin:", ["Sohbet Botu", "Regresyon Modeli"])

knowledge = load_knowledge()

if app_mode == "Sohbet Botu":
    st.header("Sohbet Botu (Öğrenebilen)")

    user_input = st.text_input("Sen:", key="chat_input")

    if user_input:
        response = chatbot_response(user_input, knowledge)
        
        if response:
            st.write("Hanogt AI:", response)
        else:
            st.warning("Bu bilgiyi bilmiyorum.")
            new_response = st.text_input("Bu soruya ne cevap vermeliyim?", key="teach_input")
            if st.button("Öğret"):
                if new_response:
                    knowledge[user_input.lower()] = new_response
                    save_knowledge(knowledge)
                    st.success("Teşekkürler! Bunu öğrendim.")
                else:
                    st.error("Lütfen bir cevap girin.")

elif app_mode == "Regresyon Modeli":
    st.header("Makine Öğrenmesi: Regresyon Modeli")
    
    degree = st.sidebar.slider("Polinom Derecesi", 1, 5, 2)
    n_samples = st.sidebar.slider("Veri Sayısı", 20, 500, 100)
    noise = st.sidebar.slider("Gürültü (Noise)", 0, 50, 10)

    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10
    y = 3 * (X.squeeze()**2) + 2 * X.squeeze() + np.random.randn(n_samples) * noise

    model = RegressionModel(degree=degree)
    model.train(X, y)

    metrics = model.evaluate()

    st.subheader("Model Performansı")
    for metric, value in metrics.items():
        st.write(f"**{metric}**: {value:.4f}")

    st.subheader("Model Grafiği")
    model.plot()

    st.subheader("Yeni Tahmin Yap")
    input_value = st.number_input("X değeri girin:", min_value=0.0, max_value=20.0, value=5.0)
    if st.button("Tahmin Yap"):
        prediction = model.predict(np.array([[input_value]]))
        st.success(f"{input_value:.2f} için tahmin: {prediction[0]:.2f}")