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

# --- Dil Ayarları ---
# Basit bir dil sözlüğü. Daha karmaşık uygulamalar için ayrı bir JSON/YAML dosyası kullanılabilir.
LANGUAGES = {
    "TR": {"name": "Türkçe", "emoji": "🇹🇷"},
    "EN": {"name": "English", "emoji": "🇬🇧"},
    "FR": {"name": "Français", "emoji": "🇫🇷"},
    "ES": {"name": "Español", "emoji": "🇪🇸"},
    "DE": {"name": "Deutsch", "emoji": "🇩🇪"}, # Örnek Avrupa ülkesi
    "RU": {"name": "Русский", "emoji": "🇷🇺"},
    "SA": {"name": "العربية", "emoji": "🇸🇦"}, # Suudi Arabistan
    "AZ": {"name": "Azərbaycan dili", "emoji": "🇦🇿"},
    "JP": {"name": "日本語", "emoji": "🇯🇵"},
    "KR": {"name": "한국어", "emoji": "🇰🇷"},
}

# --- Yardımcı Fonksiyonlar ---

def get_text(key):
    """Seçili dile göre metin döndürür."""
    # Bu basit bir örnek. Gerçek uygulamada çok daha kapsamlı bir sözlük veya dış dosya kullanılır.
    texts = {
        "TR": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Yeni Kişisel Yapay Zeka Asistanınız!",
            "profile_title": "Size Nasıl Hitap Etmeliyim?",
            "profile_name_label": "Adınız:",
            "profile_upload_label": "Profil Resmi Yükle (isteğe bağlı)",
            "profile_save_button": "Kaydet",
            "profile_greeting": "Merhaba, {name}!",
            "profile_edit_info": "Ayarlar & Kişiselleştirme bölümünden profilinizi düzenleyebilirsiniz.",
            "ai_features_title": "Hanogt AI Özellikleri:",
            "feature_general_chat": "Genel sohbet",
            "feature_web_search": "Web araması (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Bilgi tabanı yanıtları",
            "feature_creative_text": "Yaratıcı metin üretimi",
            "feature_image_generation": "Basit görsel oluşturma (örnek)",
            "feature_text_to_speech": "Metin okuma (TTS)",
            "feature_feedback": "Geri bildirim mekanizması",
            "settings_button": "⚙️ Ayarlar & Kişiselleştirme",
            "about_button": "ℹ️ Hakkımızda",
            "app_mode_title": "Uygulama Modu",
            "chat_mode_text": "💬 Yazılı Sohbet",
            "chat_mode_image": "🖼️ Görsel Oluşturucu",
            "chat_mode_voice": "🎤 Sesli Sohbet (Dosya Yükle)",
            "chat_mode_creative": "✨ Yaratıcı Stüdyo",
            "chat_input_placeholder": "Mesajınızı yazın veya bir komut girin: Örn: 'Merhaba', 'web ara: Streamlit', 'yaratıcı metin: uzaylılar'...",
            "generating_response": "Yanıt oluşturuluyor...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Geri bildirim için teşekkürler!",
            "image_gen_title": "Görsel Oluşturucu",
            "image_gen_input_label": "Oluşturmak istediğiniz görseli tanımlayın:",
            "image_gen_button": "Görsel Oluştur",
            "image_gen_warning_placeholder": "Görsel oluşturma özelliği şu anda bir placeholder'dır ve gerçek bir API'ye bağlı değildir.",
            "image_gen_warning_prompt_missing": "Lütfen bir görsel açıklaması girin.",
            "voice_chat_title": "Sesli Sohbet",
            "voice_upload_label": "Ses dosyası yükle (MP3, WAV)",
            "voice_upload_warning": "Ses dosyasından metin transkripsiyonu özelliği şu anda bir placeholder'dır.",
            "voice_live_input_title": "Canlı Ses Girişi",
            "voice_mic_button": "Mikrofonu Başlat",
            "voice_not_available": "Sesli sohbet özellikleri kullanılamıyor. Gerekli kütüphanelerin (pyttsx3, SpeechRecognition) kurulu olduğundan emin olun.",
            "voice_listening": "Dinleniyor...",
            "voice_heard": "Sen dedin: {text}",
            "voice_no_audio": "Ses algılanamadı, lütfen tekrar deneyin.",
            "voice_unknown": "Ne dediğini anlayamadım.",
            "voice_api_error": "Ses tanıma servisine ulaşılamıyor; {error}",
            "creative_studio_title": "Yaratıcı Stüdyo",
            "creative_studio_info": "Bu bölüm, yaratıcı metin üretimi gibi gelişmiş özellikler için tasarlanmıştır.",
            "creative_studio_input_label": "Yaratıcı metin isteğinizi girin:",
            "creative_studio_button": "Metin Oluştur",
            "creative_studio_warning_prompt_missing": "Lütfen bir yaratıcı metin isteği girin.",
            "settings_personalization_title": "Ayarlar & Kişiselleştirme",
            "settings_name_change_label": "Adınızı Değiştir:",
            "settings_avatar_change_label": "Profil Resmini Değiştir (isteğe bağlı)",
            "settings_update_profile_button": "Profil Bilgilerini Güncelle",
            "settings_profile_updated_toast": "Profil güncellendi!",
            "settings_chat_management_title": "Sohbet Yönetimi",
            "settings_clear_chat_button": "🧹 Aktif Sohbet Geçmişini Temizle",
            "about_us_title": "ℹ️ Hakkımızda",
            "about_us_text": "Hanogt AI HanStudios'un Sahibi Oğuz Han Guluzade Tarafından 2025 Yılında Yapılmıştır, Açık Kaynak Kodludur, Gemini Tarafından Eğitilmiştir Ve Bütün Telif Hakları Saklıdır.",
            "footer_user": "Kullanıcı: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Aktif ({model_name}) | Log: Aktif",
            "model_init_success": "Gemini Modeli başarıyla başlatıldı!",
            "model_init_error": "Gemini modelini başlatırken bir hata oluştu: {error}. Lütfen API anahtarınızın doğru ve aktif olduğundan emin olun.",
            "gemini_model_not_initialized": "Gemini modeli başlatılmamış. Lütfen API anahtarınızı kontrol edin.",
            "image_load_error": "Görsel yüklenemedi: {error}",
            "image_not_convertible": "Bu içerik konuşmaya çevrilemez (metin değil).",
            "duckduckgo_error": "DuckDuckGo araması yapılırken hata oluştu: {error}",
            "wikipedia_network_error": "Wikipedia araması yapılırken ağ hatası oluştu: {error}",
            "wikipedia_json_error": "Wikipedia yanıtı çözümlenirken hata oluştu: {error}",
            "wikipedia_general_error": "Wikipedia araması yapılırken genel bir hata oluştu: {error}",
            "unexpected_response_error": "Yanıt alınırken beklenmeyen bir hata oluştu: {error}",
            "source_error": "Kaynak: Hata ({error})",
            "chat_cleared_toast": "Aktif sohbet temizlendi!",
            "profile_image_load_error": "Profil resmi yüklenemedi: {error}",
            "web_search_results": "Web Arama Sonuçları:",
            "web_search_no_results": "Aradığınız terimle ilgili sonuç bulunamadı.",
            "wikipedia_search_results": "Wikipedia Arama Sonuçları:",
            "wikipedia_search_no_results": "Aradığınız terimle ilgili sonuç bulunamadı.",
            "image_generated_example": "'{prompt}' için bir görsel oluşturuldu (örnek).",
            "image_upload_caption": "Yüklenen Görsel",
            "image_processing_error": "Görsel işlenirken bir hata oluştu: {error}",
            "image_vision_query": "Bu görselde ne görüyorsun?",
            "loading_audio_file": "Ses dosyası yükleniyor...",
            "tts_sr_not_available": "Sesli sohbet ve metin okuma özellikleri şu anda kullanılamıyor. Gerekli kütüphaneler yüklenmemiş veya uyumlu değil.",
            "mic_listen_timeout": "Ses algılama zaman aşımına uğradı.",
            "unexpected_audio_record_error": "Ses kaydı sırasında beklenmeyen bir hata oluştu: {error}",
            "gemini_response_error": "Yanıt alınırken beklenmeyen bir hata oluştu: {error}",
            "creative_text_generated": "Yaratıcı Metin Oluşturuldu: {text}",
            "turkish_voice_not_found": "Türkçe ses bulunamadı, varsayılan ses kullanılacak. İşletim sisteminizin ses ayarlarını kontrol ediniz."
        },
        "EN": {
            "welcome_title": "Hanogt AI",
            "welcome_subtitle": "Your New Personal AI Assistant!",
            "profile_title": "How Should I Address You?",
            "profile_name_label": "Your Name:",
            "profile_upload_label": "Upload Profile Picture (optional)",
            "profile_save_button": "Save",
            "profile_greeting": "Hello, {name}!",
            "profile_edit_info": "You can edit your profile in the Settings & Personalization section.",
            "ai_features_title": "Hanogt AI Features:",
            "feature_general_chat": "General chat",
            "feature_web_search": "Web search (DuckDuckGo, Wikipedia)",
            "feature_knowledge_base": "Knowledge base responses",
            "feature_creative_text": "Creative text generation",
            "feature_image_generation": "Simple image generation (placeholder)",
            "feature_text_to_speech": "Text-to-speech (TTS)",
            "feature_feedback": "Feedback mechanism",
            "settings_button": "⚙️ Settings & Personalization",
            "about_button": "ℹ️ About Us",
            "app_mode_title": "Application Mode",
            "chat_mode_text": "💬 Text Chat",
            "chat_mode_image": "🖼️ Image Generator",
            "chat_mode_voice": "🎤 Voice Chat (Upload File)",
            "chat_mode_creative": "✨ Creative Studio",
            "chat_input_placeholder": "Type your message or enter a command: E.g., 'Hello', 'web search: Streamlit', 'creative text: aliens'...",
            "generating_response": "Generating response...",
            "tts_button": "▶️",
            "feedback_button": "👍",
            "feedback_toast": "Thanks for your feedback!",
            "image_gen_title": "Image Generator",
            "image_gen_input_label": "Describe the image you want to create:",
            "image_gen_button": "Generate Image",
            "image_gen_warning_placeholder": "Image generation feature is currently a placeholder and not connected to a real API.",
            "image_gen_warning_prompt_missing": "Please enter an image description.",
            "voice_chat_title": "Voice Chat",
            "voice_upload_label": "Upload audio file (MP3, WAV)",
            "voice_upload_warning": "Audio file transcription feature is currently a placeholder.",
            "voice_live_input_title": "Live Voice Input",
            "voice_mic_button": "Start Microphone",
            "voice_not_available": "Voice chat features are currently unavailable. Make sure required libraries (pyttsx3, SpeechRecognition) are installed and compatible.",
            "voice_listening": "Listening...",
            "voice_heard": "You said: {text}",
            "voice_no_audio": "No audio detected, please try again.",
            "voice_unknown": "Could not understand what you said.",
            "voice_api_error": "Could not reach speech recognition service; {error}",
            "creative_studio_title": "Creative Studio",
            "creative_studio_info": "This section is designed for advanced features like creative text generation.",
            "creative_studio_input_label": "Enter your creative text request:",
            "creative_studio_button": "Generate Text",
            "creative_studio_warning_prompt_missing": "Please enter a creative text request.",
            "settings_personalization_title": "Settings & Personalization",
            "settings_name_change_label": "Change Your Name:",
            "settings_avatar_change_label": "Change Profile Picture (optional)",
            "settings_update_profile_button": "Update Profile Info",
            "settings_profile_updated_toast": "Profile updated!",
            "settings_chat_management_title": "Chat Management",
            "settings_clear_chat_button": "🧹 Clear Active Chat History",
            "about_us_title": "ℹ️ About Us",
            "about_us_text": "Hanogt AI was created by Oğuz Han Guluzade, owner of HanStudios, in 2025. It is open-source, trained by Gemini, and all copyrights are reserved.",
            "footer_user": "User: {user_name}",
            "footer_version": "Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {year}",
            "footer_ai_status": "AI: Active ({model_name}) | Log: Active",
            "model_init_success": "Gemini Model successfully initialized!",
            "model_init_error": "An error occurred while initializing the Gemini model: {error}. Please ensure your API key is correct and active.",
            "gemini_model_not_initialized": "Gemini model not initialized. Please check your API key.",
            "image_load_error": "Could not load image: {error}",
            "image_not_convertible": "This content cannot be converted to speech (not text).",
            "duckduckgo_error": "An error occurred while performing DuckDuckGo search: {error}",
            "wikipedia_network_error": "Network error occurred while performing Wikipedia search: {error}",
            "wikipedia_json_error": "Error occurred while parsing Wikipedia response: {error}",
            "wikipedia_general_error": "A general error occurred while performing Wikipedia search: {error}",
            "unexpected_response_error": "An unexpected error occurred while getting a response: {error}",
            "source_error": "Source: Error ({error})",
            "chat_cleared_toast": "Active chat cleared!",
            "profile_image_load_error": "Could not load profile image: {error}",
            "web_search_results": "Web Search Results:",
            "web_search_no_results": "No results found for your search term.",
            "wikipedia_search_results": "Wikipedia Search Results:",
            "wikipedia_search_no_results": "No results found for your search term.",
            "image_generated_example": "An image for '{prompt}' was generated (example).",
            "image_upload_caption": "Uploaded Image",
            "image_processing_error": "An error occurred while processing the image: {error}",
            "image_vision_query": "What do you see in this image?",
            "loading_audio_file": "Loading audio file...",
            "tts_sr_not_available": "Voice chat and text-to-speech features are currently unavailable. Make sure required libraries are installed and compatible.",
            "mic_listen_timeout": "Audio detection timed out.",
            "unexpected_audio_record_error": "An unexpected error occurred during audio recording: {error}",
            "gemini_response_error": "An unexpected error occurred while getting a response: {error}",
            "creative_text_generated": "Creative Text Generated: {text}",
            "turkish_voice_not_found": "Turkish voice not found, default voice will be used. Please check your operating system's sound settings."
        }
        # Diğer diller buraya eklenebilir
    }
    return texts.get(st.session_state.current_language, texts["TR"]).get(key, "TEXT_MISSING")

def initialize_session_state():
    """Uygulama oturum durumunu başlatır."""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "user_avatar" not in st.session_state:
        st.session_state.user_avatar = None
    if "models_initialized" not in st.session_state:
        st.session_state.models_initialized = False
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "chat_0"
        if "chat_0" not in st.session_state.all_chats:
            st.session_state.all_chats["chat_0"] = []
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "💬 Yazılı Sohbet" # Updated to include emoji
    if "current_mode_index" not in st.session_state:
        st.session_state.current_mode_index = 0
    if "show_settings" not in st.session_state: # Ayarlar bölümünü gösterme kontrolü
        st.session_state.show_settings = False
    if "show_about" not in st.session_state: # Hakkında bölümünü gösterme kontrolü
        st.session_state.show_about = False
    if "current_language" not in st.session_state:
        st.session_state.current_language = "TR" # Varsayılan dil Türkçe

    load_chat_history()
    initialize_gemini_model()

def initialize_gemini_model():
    """Gemini modelini başlatır ve oturum durumuna kaydeder."""
    if st.session_state.get("gemini_model") is None:
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
            st.toast(get_text("model_init_success"), icon="✅")
        except Exception as e:
            st.error(get_text("model_init_error").format(error=e))
            st.session_state.models_initialized = False

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
    
def load_chat_history():
    """Sohbet geçmişini yükler."""
    if st.session_state.active_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []

def clear_active_chat():
    """Aktif sohbetin içeriğini temizler."""
    if st.session_state.active_chat_id in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        if "chat_session" in st.session_state:
            del st.session_state.chat_session
        st.toast(get_text("chat_cleared_toast"), icon="🧹")
    st.rerun()

def text_to_speech(text):
    """Metni konuşmaya çevirir ve sesi oynatır."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
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
            st.warning(get_text("turkish_voice_not_found"))

        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        st.error(get_text("unexpected_response_error").format(error=e))
        return False

def record_audio():
    """Kullanıcıdan ses girişi alır."""
    if not TTS_SR_AVAILABLE:
        st.warning(get_text("tts_sr_not_available"))
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write(get_text("voice_listening"))
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            st.warning(get_text("voice_no_audio"))
            return ""
        except Exception as e:
            st.error(get_text("unexpected_audio_record_error").format(error=e))
            return ""
            
    try:
        text = r.recognize_google(audio, language="tr-TR") # Always use TR for recognition
        st.write(get_text("voice_heard").format(text=text))
        return text
    except sr.UnknownValueError:
        st.warning(get_text("voice_unknown"))
        return ""
    except sr.RequestError as e:
        st.error(get_text("voice_api_error").format(error=e))
        return ""
    except Exception as e:
        st.error(get_text("unexpected_audio_record_error").format(error=e))
        return ""

@st.cache_data(ttl=3600)
def duckduckgo_search(query):
    """DuckDuckGo kullanarak web araması yapar."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results
    except Exception as e:
        st.error(get_text("duckduckgo_error").format(error=e))
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
        st.error(get_text("wikipedia_network_error").format(error=e))
        return []
    except json.JSONDecodeError as e:
        st.error(get_text("wikipedia_json_error").format(error=e))
        return []
    except Exception as e:
        st.error(get_text("wikipedia_general_error").format(error=e))
        return []

def generate_image(prompt):
    """Görsel oluşturma (örnek - placeholder)."""
    st.warning(get_text("image_gen_warning_placeholder"))
    placeholder_image_url = "https://via.placeholder.com/400x300.png?text=Görsel+Oluşturuldu"
    st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", get_text("image_generated_example").format(prompt=prompt))

def process_image_input(uploaded_file):
    """Yüklenen görseli işler ve metne dönüştürür."""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=get_text("image_upload_caption"), use_column_width=True)
            add_to_chat_history(st.session_state.active_chat_id, "user", image)
            
            if st.session_state.gemini_model:
                vision_chat_session = st.session_state.gemini_model.start_chat(history=[])
                response = vision_chat_session.send_message([image, get_text("image_vision_query")])
                response_text = response.text
                st.markdown(response_text)
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
            else:
                st.error(get_text("gemini_model_not_initialized"))
        except Exception as e:
            st.error(get_text("image_processing_error").format(error=e))

# --- UI Bileşenleri ---

def display_welcome_and_profile_setup():
    """Hoş geldiniz mesajı ve profil oluşturma/düzenleme."""
    st.markdown("<h1 style='text-align: center;'>Hanogt AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center; color: gray;'>{get_text('welcome_subtitle')}</h4>", unsafe_allow_html=True)
    st.write("---")

    col_features, col_profile = st.columns([1, 1])

    with col_features:
        st.subheader(get_text("ai_features_title"))
        st.markdown(f"""
            * {get_text('feature_general_chat')}
            * {get_text('feature_web_search')}
            * {get_text('feature_knowledge_base')}
            * {get_text('feature_creative_text')}
            * {get_text('feature_image_generation')}
            * {get_text('feature_text_to_speech')}
            * {get_text('feature_feedback')}
        """)

    with col_profile:
        st.markdown(f"<h3 style='text-align: center;'>{get_text('welcome_title')}</h3>", unsafe_allow_html=True)
        st.subheader(get_text("profile_title"))
        
        # Profil resmi gösterimi
        if st.session_state.user_avatar:
            try:
                profile_image = Image.open(io.BytesIO(st.session_state.user_avatar))
                st.image(profile_image, caption=st.session_state.user_name if st.session_state.user_name else "Kullanıcı", width=150)
            except Exception as e:
                st.warning(get_text("profile_image_load_error").format(error=e))
                st.image("https://via.placeholder.com/150?text=Profil", width=150)
        else:
            st.image("https://via.placeholder.com/150?text=Profil", width=150)
        
        new_name = st.text_input(get_text("profile_name_label"), key="initial_name_input")
        uploaded_avatar = st.file_uploader(get_text("profile_upload_label"), type=["png", "jpg", "jpeg"], key="initial_avatar_upload")

        if st.button(get_text("profile_save_button"), key="initial_save_button"):
            if new_name:
                st.session_state.user_name = new_name
            if uploaded_avatar:
                st.session_state.user_avatar = uploaded_avatar.read()
            st.rerun()

    st.write("---")

def display_settings_and_personalization():
    """Ayarlar ve Kişiselleştirme bölümünü gösterir."""
    st.markdown(f"## {get_text('settings_personalization_title')}")

    new_name = st.text_input(get_text("settings_name_change_label"), value=st.session_state.user_name, key="settings_name_input")
    uploaded_avatar = st.file_uploader(get_text("settings_avatar_change_label"), type=["png", "jpg", "jpeg"], key="settings_avatar_upload")

    if st.button(get_text("settings_update_profile_button"), key="update_profile_button"):
        st.session_state.user_name = new_name
        if uploaded_avatar:
            st.session_state.user_avatar = uploaded_avatar.read()
        st.toast(get_text("settings_profile_updated_toast"), icon="✅")
        st.rerun()

    st.markdown("---")
    st.markdown(f"### {get_text('settings_chat_management_title')}")
    if st.button(get_text("settings_clear_chat_button"), key="clear_active_chat_button"):
        clear_active_chat()

    st.write("---")

def display_about_section():
    """'Hakkımızda' bölümünü gösterir."""
    st.markdown(f"## {get_text('about_us_title')}")
    st.markdown(get_text("about_us_text"))
    st.write("---")

def display_main_chat_interface():
    """Ana sohbet arayüzünü gösterir."""
    
    # Ayarlar ve Hakkımızda butonları
    col_settings, col_about = st.columns(2)
    with col_settings:
        if st.button(get_text("settings_button"), key="toggle_settings"):
            st.session_state.show_settings = not st.session_state.show_settings
            st.session_state.show_about = False # Diğerini kapat
    with col_about:
        if st.button(get_text("about_button"), key="toggle_about"):
            st.session_state.show_about = not st.session_state.show_about
            st.session_state.show_settings = False # Diğerini kapat

    if st.session_state.show_settings:
        display_settings_and_personalization()
    if st.session_state.show_about:
        display_about_section()

    st.markdown("---")
    st.markdown(f"## {get_text('app_mode_title')}")

    mode_options = [
        get_text("chat_mode_text"),
        get_text("chat_mode_image"),
        get_text("chat_mode_voice"),
        get_text("chat_mode_creative")
    ]
    st.session_state.chat_mode = st.radio(
        "Mod Seçimi",
        mode_options,
        horizontal=True,
        index=mode_options.index(st.session_state.chat_mode) if st.session_state.chat_mode in mode_options else 0,
        key="main_mode_radio"
    )
    
    # Mevcut modu string olarak saklayarak hatasız bir şekilde index'e dönüştürüyoruz
    current_mode_string = st.session_state.chat_mode 

    if current_mode_string == get_text("chat_mode_text"):
        handle_text_chat()
    elif current_mode_string == get_text("chat_mode_image"):
        handle_image_generation()
    elif current_mode_string == get_text("chat_mode_voice"):
        handle_voice_chat()
    elif current_mode_string == get_text("chat_mode_creative"):
        handle_creative_studio()

def handle_text_chat():
    """Yazılı sohbet modunu yönetir."""
    chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    for message_index, message in enumerate(chat_messages):
        avatar_src = None
        if message["role"] == "user" and st.session_state.user_avatar:
            try:
                avatar_src = Image.open(io.BytesIO(st.session_state.user_avatar))
            except Exception as e:
                logger.warning(f"Failed to load user avatar for chat message: {e}") # Log it, don't show to user every time
                avatar_src = None # Fallback

        with st.chat_message(message["role"], avatar=avatar_src):
            content_part = message["parts"][0]
            if isinstance(content_part, str):
                st.markdown(content_part)
            elif isinstance(content_part, bytes):
                try:
                    image = Image.open(io.BytesIO(content_part))
                    st.image(image, caption=get_text("image_upload_caption"), use_column_width=True)
                except Exception as e:
                    st.warning(get_text("image_load_error").format(error=e))

            col_btn1, col_btn2 = st.columns([0.05, 1])
            with col_btn1:
                if st.button(get_text("tts_button"), key=f"tts_btn_{st.session_state.active_chat_id}_{message_index}"):
                    if isinstance(content_part, str):
                        text_to_speech(content_part)
                    else:
                        st.warning(get_text("image_not_convertible"))
            with col_btn2:
                if st.button(get_text("feedback_button"), key=f"fb_btn_{st.session_state.active_chat_id}_{message_index}"):
                    st.toast(get_text("feedback_toast"), icon="🙏")

    prompt = st.chat_input(get_text("chat_input_placeholder"))

    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        
        if prompt.lower().startswith("web ara:"):
            query = prompt[len("web ara:"):].strip()
            results = duckduckgo_search(query)
            if results:
                response_text = get_text("web_search_results") + "\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n{r['body']}\n{r['href']}\n\n"
            else:
                response_text = get_text("web_search_no_results")
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("wiki ara:"):
            query = prompt[len("wiki ara:"):].strip()
            results = wikipedia_search(query)
            if results:
                response_text = get_text("wikipedia_search_results") + "\n"
                for i, r in enumerate(results):
                    response_text += f"{i+1}. **{r['title']}**\n"
            else:
                response_text = get_text("wikipedia_search_no_results")
            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
        elif prompt.lower().startswith("görsel oluştur:"):
            image_prompt = prompt[len("görsel oluştur:"):].strip()
            generate_image(image_prompt)
        else:
            if st.session_state.gemini_model:
                with st.spinner(get_text("generating_response")):
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
                        response = st.session_state.chat_session.send_message(prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty() 
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        
        st.rerun()

def handle_image_generation():
    """Görsel oluşturma modunu yönetir."""
    st.subheader(get_text("image_gen_title"))
    image_prompt = st.text_input(get_text("image_gen_input_label"), key="image_prompt_input")
    if st.button(get_text("image_gen_button"), key="generate_image_button"):
        if image_prompt:
            generate_image(image_prompt)
        else:
            st.warning(get_text("image_gen_warning_prompt_missing"))

def handle_voice_chat():
    """Sesli sohbet modunu yönetir."""
    st.subheader(get_text("voice_chat_title"))
    
    if not TTS_SR_AVAILABLE:
        st.info(get_text("voice_not_available"))
    else:
        uploaded_audio_file = st.file_uploader(get_text("voice_upload_label"), type=["mp3", "wav"], key="audio_uploader")
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
            st.warning(get_text("voice_upload_warning"))

        st.markdown("---")
        st.subheader(get_text("voice_live_input_title"))
        if st.button(get_text("voice_mic_button"), key="start_mic_button"):
            recognized_text = record_audio()
            if recognized_text:
                add_to_chat_history(st.session_state.active_chat_id, "user", recognized_text)

                if st.session_state.gemini_model:
                    with st.spinner(get_text("generating_response")):
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
                            st.error(get_text("gemini_response_error").format(error=e))
                else:
                    st.warning(get_text("gemini_model_not_initialized"))

def handle_creative_studio():
    """Yaratıcı stüdyo modunu yönetir."""
    st.subheader(get_text("creative_studio_title"))
    st.write(get_text("creative_studio_info"))
    
    creative_prompt = st.text_area(get_text("creative_studio_input_label"), height=150, key="creative_prompt_input")
    if st.button(get_text("creative_studio_button"), key="generate_creative_text_button"):
        if creative_prompt:
            if st.session_state.gemini_model:
                with st.spinner(get_text("generating_response")):
                    try:
                        creative_chat_session = st.session_state.gemini_model.start_chat(history=[])
                        response = creative_chat_session.send_message(f"Yaratıcı metin oluştur: {creative_prompt}", stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", get_text("creative_text_generated").format(text=response_text))
                    except Exception as e:
                        st.error(get_text("unexpected_response_error").format(error=e))
            else:
                st.warning(get_text("gemini_model_not_initialized"))
        else:
            st.warning(get_text("creative_studio_warning_prompt_missing"))


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

    # Sağ üstteki menü ve sol üstteki menüyü gizlemek için CSS enjeksiyonu (Streamlit üzerinde sınırlı etki)
    # Bu kısımlar doğrudan Python kodundan kontrol edilemez, kullanıcı Streamlit Cloud'daysa bu kodun etkisi olmaz.
    st.markdown("""
        <style>
            /* Streamlit header'ı gizle - sağ üstteki menüleri içerir */
            header.st-emotion-cache-zq5bqg.ezrtsby0 {
                display: none;
            }
            /* Sol üstteki menü açma butonunu gizle */
            .st-emotion-cache-1avcm0k.e1tzin5v2 { /* Bu selektör değişebilir */
                display: none;
            }
            /* Uygulama başlığını ortala */
            h1 {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)


    # Dil Seçici Butonu (Sol üst köşede)
    col_lang, _ = st.columns([0.1, 0.9])
    with col_lang:
        current_lang_display = f"{LANGUAGES[st.session_state.current_language]['emoji']} {st.session_state.current_language}"
        lang_options = [f"{v['emoji']} {k}" for k, v in LANGUAGES.items()]
        
        selected_lang_index = lang_options.index(current_lang_display) if current_lang_display in lang_options else 0
        
        selected_lang_display = st.selectbox(
            label="", # Label gizli
            options=lang_options,
            index=selected_lang_index,
            key="language_selector",
            help="Uygulama dilini seçin"
        )
        
        # Seçilen kısaltmayı al
        new_lang_code = selected_lang_display.split(" ")[1] 
        if new_lang_code != st.session_state.current_language:
            st.session_state.current_language = new_lang_code
            st.rerun()

    # Profil bilgisi girilmediyse, başlangıç ekranını göster
    if st.session_state.user_name == "":
        display_welcome_and_profile_setup()
    else:
        # Uygulamanın ana başlığı
        st.markdown("<h1 style='text-align: center;'>Hanogt AI</h1>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; color: gray;'>{get_text('welcome_subtitle')}</h4>", unsafe_allow_html=True)
        st.write("---") # Başlık ile diğer içerik arasına çizgi

        display_main_chat_interface()

    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style="text-align: center; font-size: 12px; color: gray;">
            {get_text('footer_user').format(user_name=st.session_state.user_name if st.session_state.user_name else "Misafir")} <br>
            {get_text('footer_version').format(year=datetime.datetime.now().year)} <br>
            {get_text('footer_ai_status').format(model_name=GLOBAL_MODEL_NAME)}
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
