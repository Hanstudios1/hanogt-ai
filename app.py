import streamlit as st
import google.generativeai as genai
import os
import io
import uuid
import time
from duckduckgo_search import DDGS
import requests
import re
import datetime
from PIL import Image
import numpy as np
import logging

# --- İsteğe Bağlı Kütüphaneler (Platforma özel kurulum gerektirebilir) ---
try:
    import pyttsx3
    import speech_recognition as sr
    TTS_SR_AVAILABLE = True
except ImportError:
    TTS_SR_AVAILABLE = False
    logging.warning("pyttsx3 veya speech_recognition modülleri bulunamadı. Sesli özellikler devre dışı bırakıldı.")

# --- Global Değişkenler ve Ayarlar ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Anahtarı Kontrolü
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") if st.secrets else os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY bulunamadı. Lütfen Streamlit Secrets'ı veya ortam değişkenlerini kontrol edin.")
    logger.error("GOOGLE_API_KEY bulunamadı. Uygulama durduruluyor.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google API Anahtarı başarıyla yapılandırıldı.")
except Exception as e:
    logger.error(f"Genel API Yapılandırma Hatası: {e}")
    st.error(f"API anahtarı yapılandırılamadı: {e}. Lütfen anahtarınızı kontrol edin.")
    st.stop()

# Gemini Model Parametreleri
GLOBAL_MODEL_NAME = 'gemini-1.5-flash-latest'
GLOBAL_TEMPERATURE = 0.7
GLOBAL_TOP_P = 0.95
GLOBAL_TOP_K = 40
GLOBAL_MAX_OUTPUT_TOKENS = 4096

# --- Yardımcı Fonksiyonlar ---

def initialize_session_state():
    """Uygulama oturum durumunu başlatır."""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_avatar" not in st.session_state:
        st.session_state.user_avatar = None # Bytes formatında sakla
    if "models_initialized" not in st.session_state:
        st.session_state.models_initialized = False
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat_0"
        if "chat_0" not in st.session_state.all_chats:
            st.session_state.all_chats["chat_0"] = []
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "Yazılı Sohbet"
    if "current_mode_index" not in st.session_state:
        st.session_state.current_mode_index = 0

    load_chat_history()
    initialize_gemini_model()

def initialize_gemini_model():
    """Gemini modelini başlatır ve oturum durumuna kaydeder."""
    if st.session_state.get("gemini_model") is None: # Sadece henüz başlatılmamışsa başlat
        try:
            st.session_state.gemini_model = genai.GenerativeModel(
                model_name=GLOBAL_MODEL_NAME,
                generation_config=genai.GenerationConfig(
                    temperature=GLOBAL_TEMPERATURE,
                    top_p=GLOBAL_TOP_P,
                    top_k=GLOBAL_TOP_K,
                    max_output_tokens=GLOBAL_MAX_OUTPUT_TOKENS,
                )
            )
            st.session_state.models_initialized = True
            st.toast("Gemini Modeli başarıyla başlatıldı!", icon="✅")
            logger.info(f"Gemini Modeli başlatıldı: {GLOBAL_MODEL_NAME}")
        except Exception as e:
            st.error(f"Gemini modelini başlatırken bir hata oluştu: {e}. Lütfen API anahtarınızın doğru ve aktif olduğundan emin olun.")
            st.session_state.models_initialized = False
            logger.error(f"Gemini modeli başlatma hatası: {e}")

def add_to_chat_history(chat_id, role, content):
    """Sohbet geçmişine mesaj ekler."""
    if chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[chat_id] = []
    
    if isinstance(content, Image.Image):
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='PNG')
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [img_byte_arr.getvalue()]})
    elif isinstance(content, bytes):
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    else:
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    
    logger.info(f"Sohbet geçmişine eklendi: Chat ID: {chat_id}, Rol: {role}, İçerik Türü: {type(content)}")

def load_chat_history():
    """Sohbet geçmişini yükler."""
    # Bu fonksiyon şimdi initialize_session_state içinde çağrılıyor.
    # Burada sadece mevcut sohbetin varlığını garanti ediyoruz.
    if st.session_state.active_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []

def clear_active_chat():
    """Aktif sohbetin içeriğini temizler."""
    if st.session_state.active_chat_id in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.toast("Aktif sohbet temizlendi!", icon="🧹")
        logger.info(f"Aktif sohbet ({st.session_state.active_chat_id}) temizlendi.")
    st.rerun()

def text_to_speech(text):
    """Metni konuşmaya çevirir ve sesi oynatır."""
    if not TTS_SR_AVAILABLE:
        st.warning("Metin okuma özelliği kullanılamıyor (pyttsx3 yüklü değil).")
        return False
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        found_turkish_voice = False
        for voice in voices:
            if "turkish" in voice.name.lower() or "tr-tr" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                found_turkish_voice = True
                break
        if not found_turkish_voice:
            st.warning("Türkçe ses bulunamadı, varsayılan ses kullanılacak.")

        engine.say(text)
        engine.runAndWait()
        logger.info("Metinden sese çevirme başarılı.")
        return True
    except Exception as e:
        st.error(f"Metni sese çevirirken hata oluştu: {e}")
        logger.error(f"Metinden sese çevirme hatası: {e}")
        return False

def record_audio():
    """Kullanıcıdan ses girişi alır."""
    if not TTS_SR_AVAILABLE:
        st.warning("Ses tanıma özelliği kullanılamıyor (speech_recognition yüklü değil).")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Dinleniyor...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning("Ses algılanamadı, lütfen tekrar deneyin.")
            return ""
        except Exception as e:
            st.error(f"Ses kaydı sırasında bir hata oluştu: {e}")
            return ""
            
    try:
        text = r.recognize_google(audio, language="tr-TR")
        st.write(f"Sen dedin: {text}")
        logger.info(f"Tanınan ses: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Ne dediğini anlayamadım.")
        return ""
    except sr.RequestError as e:
        st.error(f"Ses tanıma servisine ulaşılamıyor; {e}")
        return ""
    except Exception as e:
        st.error(f"Ses tanıma sırasında beklenmeyen bir hata oluştu: {e}")
        return ""

@st.cache_data(ttl=3600)
def duckduckgo_search(query):
    """DuckDuckGo kullanarak web araması yapar."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results
    except Exception as e:
        st.error(f"DuckDuckGo araması yapılırken hata oluştu: {e}")
        return []

@st.cache_data(ttl=3600)
def wikipedia_search(query):
    """Wikipedia'da arama yapar."""
    try:
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
        response.raise_for_status()
        data = response.json()
        if data and "query" in data and "search" in data["query"]:
            return data["query"]["search"]
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Wikipedia araması yapılırken ağ hatası oluştu: {e}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Wikipedia yanıtı çözümlenirken hata oluştu: {e}")
        return []
    except Exception as e:
        st.error(f"Wikipedia araması yapılırken genel bir hata oluştu: {e}")
        return []

def generate_image(prompt):
    """Görsel oluşturma (örnek - placeholder)."""
    st.warning("Görsel oluşturma özelliği şu anda bir placeholder'dır ve gerçek bir API'ye bağlı değildir.")
    placeholder_image_url = "https://via.placeholder.com/400x300.png?text=Görsel+Oluşturuldu"
    st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", f"'{prompt}' için bir görsel oluşturuldu (örnek).")

def process_image_input(uploaded_file):
    """Yüklenen görseli işler ve metne dönüştürür."""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görsel", use_column_width=True)
            add_to_chat_history(st.session_state.active_chat_id, "user", image)
            
            if st.session_state.gemini_model:
                vision_chat_session = st.session_state.gemini_model.start_chat(history=[])
                response = vision_chat_session.send_message([image, "Bu görselde ne görüyorsun?"])
                response_text = response.text
                st.markdown(response_text)
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
            else:
                st.error("Gemini modeli başlatılmamış.")
        except Exception as e:
            st.error(f"Görsel işlenirken bir hata oluştu: {e}")

# --- UI Bileşenleri ---

def display_welcome_and_profile_setup():
    """Hoş geldiniz mesajı ve profil oluşturma/düzenleme."""
    st.title("Hanogt AI")
    st.markdown("<h4 style='text-align: center; color: gray;'>Yeni Kişisel Yapay Zeka Asistanınız!</h4>", unsafe_allow_html=True)
    st.write("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Profiliniz")
        if st.session_state.user_avatar:
            try:
                profile_image = Image.open(io.BytesIO(st.session_state.user_avatar))
                st.image(profile_image, caption=st.session_state.user_name if st.session_state.user_name else "Kullanıcı", width=150)
            except Exception as e:
                st.warning(f"Profil resmi yüklenemedi: {e}")
                st.image("https://via.placeholder.com/150?text=Profil", width=150) # Hata olursa varsayılan
        else:
            st.image("https://via.placeholder.com/150?text=Profil", width=150)

    with col2:
        if st.session_state.user_name == "":
            st.subheader("Size Nasıl Hitap Etmeliyim?")
            new_name = st.text_input("Adınız:", key="initial_name_input")
            uploaded_avatar = st.file_uploader("Profil Resmi Yükle (isteğe bağlı)", type=["png", "jpg", "jpeg"], key="initial_avatar_upload")

            if st.button("Kaydet", key="initial_save_button"):
                if new_name:
                    st.session_state.user_name = new_name
                if uploaded_avatar:
                    st.session_state.user_avatar = uploaded_avatar.read()
                st.rerun()
        else:
            st.subheader(f"Merhaba, {st.session_state.user_name}!")
            st.write("Ayarlar & Kişiselleştirme bölümünden profilinizi düzenleyebilirsiniz.")

    st.write("---")

def display_settings_and_personalization():
    """Ayarlar ve Kişiselleştirme bölümünü gösterir."""
    st.markdown("## Ayarlar & Kişiselleştirme")

    new_name = st.text_input("Adınızı Değiştir:", value=st.session_state.user_name, key="settings_name_input")
    uploaded_avatar = st.file_uploader("Profil Resmini Değiştir (isteğe bağlı)", type=["png", "jpg", "jpeg"], key="settings_avatar_upload")

    if st.button("Profil Bilgilerini Güncelle", key="update_profile_button"):
        st.session_state.user_name = new_name
        if uploaded_avatar:
            st.session_state.user_avatar = uploaded_avatar.read()
        st.toast("Profil güncellendi!", icon="✅")
        st.rerun()

    st.markdown("---")
    st.markdown("### Sohbet Yönetimi")
    if st.button("🧹 Aktif Sohbet Geçmişini Temizle", key="clear_active_chat_button"):
        clear_active_chat()

    st.write("---")

def display_about_section():
    """'Hakkımızda' bölümünü gösterir."""
    st.markdown("## ℹ️ Hakkımızda")
    st.markdown("""
        **Hanogt AI** HanStudios'un Sahibi Oğuz Han Guluzade Tarafından 2025 Yılında Yapılmıştır,
        Açık Kaynak Kodludur, Gemini Tarafından Eğitilmiştir Ve Bütün Telif Hakları Saklıdır.
    """)
    st.write("---")

def display_main_chat_interface():
    """Ana sohbet arayüzünü gösterir."""
    
    # Ana mod seçim butonları
    # Modların üzerine ayarlar butonu geldiği için burada tekrar title eklemiyorum
    
    # Ayrı kolonlarda Ayarlar ve Hakkımızda butonları
    settings_col, about_col = st.columns(2)
    with settings_col:
        if st.button("⚙️ Ayarlar & Kişiselleştirme", key="btn_settings_personalization"):
            st.session_state.current_page = "settings_personalization"
            st.rerun()
    with about_col:
        if st.button("ℹ️ Hakkımızda", key="btn_about"):
            st.session_state.current_page = "about_page"
            st.rerun()
    
    st.markdown("---") # Ayarlar/Hakkımızda butonları ile mod seçim arasına çizgi
    st.markdown("## Uygulama Modu")


    st.session_state.chat_mode = st.radio(
        "Mod Seçimi",
        ["💬 Yazılı Sohbet", "🖼️ Görsel Oluşturucu", "🎤 Sesli Sohbet (Dosya Yükle)", "✨ Yaratıcı Stüdyo"],
        horizontal=True,
        index=st.session_state.get("current_mode_index", 0),
        key="main_mode_radio"
    )
    st.session_state.current_mode_index = ["💬 Yazılı Sohbet", "🖼️ Görsel Oluşturucu", "🎤 Sesli Sohbet (Dosya Yükle)", "✨ Yaratıcı Stüdyo"].index(st.session_state.chat_mode)

    if st.session_state.chat_mode == "💬 Yazılı Sohbet":
        handle_text_chat()
    elif st.session_state.chat_mode == "🖼️ Görsel Oluşturucu":
        handle_image_generation()
    elif st.session_state.chat_mode == "🎤 Sesli Sohbet (Dosya Yükle)":
        handle_voice_chat()
    elif st.session_state.chat_mode == "✨ Yaratıcı Stüdyo":
        handle_creative_studio()

def handle_text_chat():
    """Yazılı sohbet modunu yönetir."""
    chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    for message_index, message in enumerate(chat_messages):
        avatar_src = None
        if message["role"] == "user" and st.session_state.user_avatar:
            avatar_src = Image.open(io.BytesIO(st.session_state.user_avatar))

        with st.chat_message(message["role"], avatar=avatar_src):
            content_part = message["parts"][0]
            if isinstance(content_part, str):
                st.markdown(content_part)
            elif isinstance(content_part, bytes):
                try:
                    image = Image.open(io.BytesIO(content_part))
                    st.image(image, caption="Yüklenen Görsel", use_column_width=True)
                except Exception as e:
                    st.warning(f"Görsel yüklenemedi: {e}")

            col_btn1, col_btn2 = st.columns([0.05, 1])
            with col_btn1:
                if st.button("▶️", key=f"tts_btn_{st.session_state.active_chat_id}_{message_index}"):
                    if isinstance(content_part, str):
                        text_to_speech(content_part)
                    else:
                        st.warning("Bu içerik konuşmaya çevrilemez (metin değil).")
            with col_btn2:
                if st.button("👍", key=f"fb_btn_{st.session_state.active_chat_id}_{message_index}"):
                    st.toast("Geri bildirim için teşekkürler!", icon="🙏")

    prompt = st.chat_input("Mesajınızı yazın veya bir komut girin: Örn: 'Merhaba', 'web ara: Streamlit', 'yaratıcı metin: uzaylılar'...")

    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        
        if prompt.lower().startswith("web ara:"):
            query = prompt[len("web ara:"):].strip()
            results = duckduckgo_search(query)
            if results:
                response_text = "Web Arama Sonuçları:\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n{r['body']}\n{r['href']}\n\n"
            else:
                response_text = "Aradığınız terimle ilgili sonuç bulunamadı."
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("wiki ara:"):
            query = prompt[len("wiki ara:"):].strip()
            results = wikipedia_search(query)
            if results:
                response_text = "Wikipedia Arama Sonuçları:\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n"
            else:
                response_text = "Aradığınız terimle ilgili sonuç bulunamadı."
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("görsel oluştur:"):
            image_prompt = prompt[len("görsel oluştur:"):].strip()
            generate_image(image_prompt)
        else:
            if st.session_state.gemini_model:
                with st.spinner("Yanıt oluşturuluyor..."):
                    try:
                        processed_history = []
                        for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                            if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                try:
                                    processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                except Exception:
                                    continue # Geçersiz görseli atla
                            else:
                                processed_history.append(msg)

                        st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=processed_history)
                        response = st.session_state.chat_session.send_message(prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty() 
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                    except Exception as e:
                        st.error(f"Yanıt alınırken beklenmeyen bir hata oluştu: {e}")
            else:
                st.warning("Gemini modeli başlatılmamış.")
        
        st.rerun()

def handle_image_generation():
    """Görsel oluşturma modunu yönetir."""
    st.subheader("Görsel Oluşturucu")
    image_prompt = st.text_input("Oluşturmak istediğiniz görseli tanımlayın:", key="image_prompt_input")
    if st.button("Görsel Oluştur", key="generate_image_button"):
        if image_prompt:
            generate_image(image_prompt)
        else:
            st.warning("Lütfen bir görsel açıklaması girin.")

def handle_voice_chat():
    """Sesli sohbet modunu yönetir."""
    st.subheader("Sesli Sohbet")
    
    if not TTS_SR_AVAILABLE:
        st.info("Sesli sohbet özellikleri kullanılamıyor. Gerekli kütüphanelerin (pyttsx3, SpeechRecognition) kurulu olduğundan emin olun.")
    else:
        uploaded_audio_file = st.file_uploader("Ses dosyası yükle (MP3, WAV)", type=["mp3", "wav"], key="audio_uploader")
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
            st.warning("Ses dosyasından metin transkripsiyonu özelliği şu anda bir placeholder'dır.")

        st.markdown("---")
        st.subheader("Canlı Ses Girişi")
        if st.button("Mikrofonu Başlat", key="start_mic_button"):
            recognized_text = record_audio()
            if recognized_text:
                add_to_chat_history(st.session_state.active_chat_id, "user", recognized_text)

                if st.session_state.gemini_model:
                    with st.spinner("Yanıt oluşturuluyor..."):
                        try:
                            processed_history = []
                            for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                                if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                    try:
                                        processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                    except Exception:
                                        continue
                                else:
                                    processed_history.append(msg)

                            st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=processed_history)
                            response = st.session_state.chat_session.send_message(recognized_text, stream=True)
                            response_text = ""
                            response_placeholder = st.empty()
                            for chunk in response:
                                response_text += chunk.text
                                with response_placeholder.container():
                                    st.markdown(response_text)
                            
                            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                            text_to_speech(response_text)
                            st.rerun()

                        except Exception as e:
                            st.error(f"Yanıt alınırken beklenmeyen bir hata oluştu: {e}")
                else:
                    st.warning("Gemini modeli başlatılmamış.")

def handle_creative_studio():
    """Yaratıcı stüdyo modunu yönetir."""
    st.subheader("Yaratıcı Stüdyo")
    st.write("Bu bölüm, yaratıcı metin üretimi gibi gelişmiş özellikler için tasarlanmıştır.")
    
    creative_prompt = st.text_area("Yaratıcı metin isteğinizi girin:", height=150, key="creative_prompt_input")
    if st.button("Metin Oluştur", key="generate_creative_text_button"):
        if creative_prompt:
            if st.session_state.gemini_model:
                with st.spinner("Yaratıcı metin oluşturuluyor..."):
                    try:
                        creative_chat_session = st.session_state.gemini_model.start_chat(history=[])
                        response = creative_chat_session.send_message(f"Yaratıcı metin oluştur: {creative_prompt}", stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", f"Yaratıcı Metin Oluşturuldu: {response_text}")
                    except Exception as e:
                        st.error(f"Yaratıcı metin oluşturulurken bir hata oluştu: {e}")
            else:
                st.warning("Gemini modeli başlatılmamış.")
        else:
            st.warning("Lütfen bir yaratıcı metin isteği girin.")

# --- Ana Uygulama Mantığı ---

def main():
    """Ana Streamlit uygulamasını çalıştırır."""
    st.set_page_config(
        page_title="Hanogt AI Asistan",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed" # Sidebar'ı tamamen kaldır
    )

    initialize_session_state()

    # Sayfa yönlendirme (Basit bir router)
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main_chat" # Varsayılan olarak ana sohbet

    if st.session_state.user_name == "" and st.session_state.current_page == "main_chat":
        display_welcome_and_profile_setup()
    else:
        # Menü butonları ve ana başlık her zaman görünür olacak
        st.title("Hanogt AI")
        st.markdown("<h4 style='text-align: center; color: gray;'>Yeni Kişisel Yapay Zeka Asistanınız!</h4>", unsafe_allow_html=True)
        st.write("---") # Başlık ile diğer içerik arasına çizgi


        if st.session_state.current_page == "main_chat":
            display_main_chat_interface()
        elif st.session_state.current_page == "settings_personalization":
            display_settings_and_personalization()
            # Ana menüye dön butonu
            if st.button("⬅️ Ana Menüye Dön", key="back_to_main_from_settings"):
                st.session_state.current_page = "main_chat"
                st.rerun()
        elif st.session_state.current_page == "about_page":
            display_about_section()
            # Ana menüye dön butonu
            if st.button("⬅️ Ana Menüye Dön", key="back_to_main_from_about"):
                st.session_state.current_page = "main_chat"
                st.rerun()

    # Footer (Ana menü dışında da görünür olabilir, Streamlit'in footer'ı yok)
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; font-size: 12px; color: gray;">
            Kullanıcı: {st.session_state.user_name if st.session_state.user_name else "Misafir"} <br>
            Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {datetime.datetime.now().year} <br>
            AI: Aktif ({GLOBAL_MODEL_NAME}) | Log: Aktif
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
