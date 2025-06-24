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
import logging # Loglama için

# --- İsteğe Bağlı Kütüphaneler (Platforma özel kurulum gerektirebilir) ---
# pyttsx3 ve speech_recognition genellikle Streamlit Cloud'da ek yapılandırma gerektirir.
# Yerel geliştirme için uygundur.
try:
    import pyttsx3
    import speech_recognition as sr
    TTS_SR_AVAILABLE = True
except ImportError:
    TTS_SR_AVAILABLE = False
    logging.warning("pyttsx3 veya speech_recognition modülleri bulunamadı. Sesli özellikler devre dışı bırakıldı.")
    st.warning("Sesli sohbet ve metin okuma özellikleri şu anda kullanılamıyor. Gerekli kütüphaneler yüklenmemiş veya uyumlu değil.")

# --- Global Değişkenler ve Ayarlar ---
# Loglama yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Anahtarı Kontrolü
# Streamlit Secrets (Cloud) veya .env (Yerel) kullanımı
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


# Gemini Model Parametreleri (Global olarak tanımlandı)
GLOBAL_MODEL_NAME = 'gemini-1.5-flash-latest'
GLOBAL_TEMPERATURE = 0.7
GLOBAL_TOP_P = 0.95
GLOBAL_TOP_K = 40
GLOBAL_MAX_OUTPUT_TOKENS = 4096

# --- Yardımcı Fonksiyonlar (Helper Functions) ---

def initialize_gemini_model():
    """Gemini modelini başlatır ve oturum durumuna kaydeder."""
    if "gemini_model" not in st.session_state:
        st.session_state.gemini_model = None # Başlangıçta None olarak ayarla
    
    if st.session_state.gemini_model is None:
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
    
    # Eğer içerik bir PIL Image nesnesi ise, byte'a dönüştürerek kaydet
    if isinstance(content, Image.Image):
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='PNG') # PNG daha iyi şeffaflık/kalite sunar, JPEG de olabilir
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [img_byte_arr.getvalue()]})
    elif isinstance(content, bytes): # Eğer zaten bytes ise doğrudan ekle
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    else: # Metin ise
        st.session_state.all_chats[chat_id].append({"role": role, "parts": [content]})
    
    # Sohbet geçmişini güncelledikten sonra logla
    logger.info(f"Sohbet geçmişine eklendi: Chat ID: {chat_id}, Rol: {role}, İçerik Türü: {type(content)}")


def load_chat_history():
    """Sohbet geçmişini yükler (uygulama başlatıldığında veya yeniden yüklendiğinde)."""
    if "all_chats" not in st.session_state:
        st.session_state.all_chats = {}
        logger.info("all_chats oturum durumu başlatıldı.")
    if "active_chat_id" not in st.session_state:
        # Tek sohbet modu için varsayılan olarak 'chat_0'
        st.session_state.active_chat_id = "chat_0"
        logger.info("active_chat_id varsayılan olarak 'chat_0' olarak ayarlandı.")
    
    # Mevcut aktif sohbetin boş olduğundan emin ol
    if st.session_state.active_chat_id not in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        logger.info(f"Aktif sohbet ID'si {st.session_state.active_chat_id} için boş bir liste oluşturuldu.")

def clear_active_chat():
    """Aktif sohbetin içeriğini temizler."""
    if st.session_state.active_chat_id in st.session_state.all_chats:
        st.session_state.all_chats[st.session_state.active_chat_id] = []
        # Chat session'ı da temizle, aksi takdirde eski geçmişi hatırlayabilir
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
                logger.info(f"Türkçe ses bulundu ve ayarlandı: {voice.name}")
                break
        if not found_turkish_voice:
            logger.warning("Türkçe ses bulunamadı, varsayılan ses kullanılacak.")
            st.warning("Türkçe ses bulunamadı, varsayılan ses kullanılacak. İşletim sisteminizin ses ayarlarını kontrol ediniz.")

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
        logger.info("Mikrofondan dinleniyor...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            logger.info("Ses kaydı tamamlandı.")
        except sr.WaitTimeoutError:
            st.warning("Ses algılanamadı, lütfen tekrar deneyin.")
            logger.warning("Ses algılama zaman aşımına uğradı.")
            return ""
        except Exception as e:
            st.error(f"Ses kaydı sırasında bir hata oluştu: {e}")
            logger.error(f"Ses kaydı hatası: {e}")
            return ""
            
    try:
        text = r.recognize_google(audio, language="tr-TR")
        st.write(f"Sen dedin: {text}")
        logger.info(f"Tanınan ses: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Ne dediğini anlayamadım.")
        logger.warning("Google Speech Recognition anlaşılamayan ses.")
        return ""
    except sr.RequestError as e:
        st.error(f"Ses tanıma servisine ulaşılamıyor; {e}")
        logger.error(f"Google Speech Recognition API hatası: {e}")
        return ""
    except Exception as e:
        st.error(f"Ses tanıma sırasında beklenmeyen bir hata oluştu: {e}")
        logger.error(f"Genel ses tanıma hatası: {e}")
        return ""

@st.cache_data(ttl=3600) # Bir saat önbelleğe al
def duckduckgo_search(query):
    """DuckDuckGo kullanarak web araması yapar."""
    logger.info(f"DuckDuckGo araması başlatılıyor: {query}")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            logger.info(f"DuckDuckGo araması tamamlandı, {len(results)} sonuç bulundu.")
            return results
    except Exception as e:
        st.error(f"DuckDuckGo araması yapılırken hata oluştu: {e}")
        logger.error(f"DuckDuckGo arama hatası: {e}")
        return []

@st.cache_data(ttl=3600) # Bir saat önbelleğe al
def wikipedia_search(query):
    """Wikipedia'da arama yapar."""
    logger.info(f"Wikipedia araması başlatılıyor: {query}")
    try:
        response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json")
        response.raise_for_status() # HTTP hatalarını yakala
        data = response.json()
        if data and "query" in data and "search" in data["query"]:
            logger.info(f"Wikipedia araması tamamlandı, {len(data['query']['search'])} sonuç bulundu.")
            return data["query"]["search"]
        logger.info("Wikipedia araması sonuç bulamadı.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Wikipedia araması yapılırken ağ hatası oluştu: {e}")
        logger.error(f"Wikipedia ağ hatası: {e}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Wikipedia yanıtı çözümlenirken hata oluştu: {e}")
        logger.error(f"Wikipedia JSON çözümleme hatası: {e}")
        return []
    except Exception as e:
        st.error(f"Wikipedia araması yapılırken genel bir hata oluştu: {e}")
        logger.error(f"Wikipedia genel arama hatası: {e}")
        return []

def generate_image(prompt):
    """Görsel oluşturma (örnek - placeholder)."""
    st.warning("Görsel oluşturma özelliği şu anda bir placeholder'dır ve gerçek bir API'ye bağlı değildir.")
    placeholder_image_url = "https://via.placeholder.com/400x300.png?text=Görsel+Oluşturuldu"
    st.image(placeholder_image_url, caption=prompt)
    add_to_chat_history(st.session_state.active_chat_id, "model", f"'{prompt}' için bir görsel oluşturuldu (örnek).")
    logger.info(f"Görsel oluşturma placeholder'ı kullanıldı: {prompt}")

def process_image_input(uploaded_file):
    """Yüklenen görseli işler ve metne dönüştürür."""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Yüklenen Görsel", use_column_width=True)
            logger.info(f"Görsel yüklendi: {uploaded_file.name}")

            # Görseli sohbet geçmişine ekle
            add_to_chat_history(st.session_state.active_chat_id, "user", image) # Image nesnesini doğrudan ekle
            
            # Gemini modelinin çok modlu yeteneklerini kullanarak görseli anlama
            if st.session_state.gemini_model:
                # Yeni bir chat session başlat, sadece görsel açıklama için
                # Bu kısım stream=True kullanmaz, tek bir yanıt beklenir
                vision_chat_session = st.session_state.gemini_model.start_chat(history=[])
                
                # Modeli hem görseli hem de metni kullanarak sorgula
                response = vision_chat_session.send_message([image, "Bu görselde ne görüyorsun?"])
                
                response_text = response.text
                st.markdown(response_text)
                
                add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                logger.info("Görsel açıklaması Gemini modeli tarafından işlendi.")
            else:
                st.error("Gemini modeli başlatılmamış.")
                logger.error("Görsel işleme sırasında Gemini modeli başlatılmamış.")
        except Exception as e:
            st.error(f"Görsel işlenirken bir hata oluştu: {e}")
            logger.error(f"Görsel işleme hatası: {e}")

# --- UI Components (Kullanıcı Arayüzü Bileşenleri) ---

def display_about_section():
    """'Hakkında' bölümünü gösterir."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Hakkında")
    st.sidebar.markdown("""
        **Hanogt AI v5.1.5 Pro+ Enhanced (Refactored)**
        Streamlit ve Google Gemini Pro ile geliştirilmiştir.

        **Desteklenen Özellikler:**
        * **Genel sohbet**
        * **Web araması** (DuckDuckGo, Wikipedia)
        * **Bilgi tabanı** yanıtları
        * **Yaratıcı metin** üretimi
        * **Basit görsel** oluşturma (örnek)
        * **Metin okuma** (TTS)
        * **Geri bildirim** mekanizması
        (Supabase) © 2025
    """)

def display_settings_section():
    """Ayarlar bölümünü gösterir."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Ayarlar")

    if st.button("🧹 Sohbet Geçmişini Temizle", key="clear_chat_button"):
        clear_active_chat()

def display_main_chat_interface():
    """Ana sohbet arayüzünü gösterir."""
    st.markdown("## Uygulama Modu")

    st.session_state.chat_mode = st.radio(
        "Mod Seçimi",
        ["Yazılı Sohbet", "Görsel Oluşturucu", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Stüdyo"],
        horizontal=True,
        index=st.session_state.get("current_mode_index", 0),
        key="main_mode_radio"
    )
    st.session_state.current_mode_index = ["Yazılı Sohbet", "Görsel Oluşturucu", "Sesli Sohbet (Dosya Yükle)", "Yaratıcı Stüdyo"].index(st.session_state.chat_mode)


    if st.session_state.chat_mode == "Yazılı Sohbet":
        handle_text_chat()
    elif st.session_state.chat_mode == "Görsel Oluşturucu":
        handle_image_generation()
    elif st.session_state.chat_mode == "Sesli Sohbet (Dosya Yükle)":
        handle_voice_chat()
    elif st.session_state.chat_mode == "Yaratıcı Stüdyo":
        handle_creative_studio()

def handle_text_chat():
    """Yazılı sohbet modunu yönetir."""
    chat_messages = st.session_state.all_chats.get(st.session_state.active_chat_id, [])

    for message_index, message in enumerate(chat_messages):
        with st.chat_message(message["role"]):
            content_part = message["parts"][0]
            if isinstance(content_part, str):
                st.markdown(content_part)
            elif isinstance(content_part, bytes): # Görsel ise
                try:
                    image = Image.open(io.BytesIO(content_part))
                    st.image(image, caption="Yüklenen Görsel", use_column_width=True)
                except Exception as e:
                    st.warning(f"Görsel yüklenemedi: {e}")
                    logger.warning(f"Görsel yükleme hatası ({message_index}): {e}")

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
                    logger.info(f"Geri bildirim alındı. Mesaj Index: {message_index}")


    prompt = st.chat_input("Mesajınızı yazın veya bir komut girin: Örn: 'Merhaba', 'web ara: Streamlit', 'yaratıcı metin: uzaylılar'...")

    if prompt:
        add_to_chat_history(st.session_state.active_chat_id, "user", prompt)
        logger.info(f"Kullanıcı promptu: {prompt}")

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
            logger.info(f"Web araması tamamlandı. Sonuç: {response_text[:100]}...")
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
            logger.info(f"Wikipedia araması tamamlandı. Sonuç: {response_text[:100]}...")
        elif prompt.lower().startswith("görsel oluştur:"):
            image_prompt = prompt[len("görsel oluştur:"):].strip()
            generate_image(image_prompt)
        else:
            if st.session_state.gemini_model:
                with st.spinner("Yanıt oluşturuluyor..."):
                    try:
                        # Chat session'ı sadece ihtiyaç duyulduğunda başlat veya güncelleyin
                        if "chat_session" not in st.session_state or \
                           (st.session_state.chat_session.history != st.session_state.all_chats[st.session_state.active_chat_id] and \
                            st.session_state.all_chats[st.session_state.active_chat_id]): # Geçmiş boşsa yeniden başlatma
                            
                            # Geçmişte görsel olup olmadığını kontrol et (Gemini modeli sadece metin ve görseli birlikte alır, sadece bytes'ı almaz)
                            # `add_to_chat_history` Image nesnesini bytes'a çeviriyor, bu yüzden kontrol etmemiz gerekiyor.
                            processed_history = []
                            for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                                if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                    try:
                                        # Bytes'ı tekrar PIL Image'a dönüştürerek modele gönder
                                        processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                    except Exception as e:
                                        logger.warning(f"Geçmişteki görsel yüklenemedi, atlanıyor: {e}")
                                        # Görsel yüklenemezse veya geçersizse, bu mesajı atlayabiliriz
                                        continue 
                                else:
                                    processed_history.append(msg)

                            st.session_state.chat_session = st.session_state.gemini_model.start_chat(
                                history=processed_history
                            )
                            logger.info("Yeni Gemini sohbet oturumu başlatıldı veya güncellendi.")

                        # Kullanıcının son mesajını gönder (yukarıda zaten geçmişe eklendi)
                        response = st.session_state.chat_session.send_message(prompt, stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty() 
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                        logger.info("Gemini yanıtı başarıyla alındı ve eklendi.")

                    except Exception as e:
                        st.error(f"Yanıt alınırken beklenmeyen bir hata oluştu: {e}")
                        st.error(f"Kaynak: Hata ({e})")
                        logger.error(f"Gemini yanıtı hatası: {e}")
            else:
                st.warning("Gemini modeli başlatılmamış. Lütfen API anahtarınızı kontrol edin.")
                logger.warning("Gemini modeli başlatılmamış uyarısı.")
        
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
            logger.warning("Görsel oluşturma için prompt girilmedi.")

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
            logger.info(f"Ses dosyası yüklendi: {uploaded_audio_file.name}")

        st.markdown("---")
        st.subheader("Canlı Ses Girişi")
        if st.button("Mikrofonu Başlat", key="start_mic_button"):
            recognized_text = record_audio()
            if recognized_text:
                add_to_chat_history(st.session_state.active_chat_id, "user", recognized_text)
                logger.info(f"Canlı ses girişi tanındı: {recognized_text}")

                if st.session_state.gemini_model:
                    with st.spinner("Yanıt oluşturuluyor..."):
                        try:
                            # Sohbet oturumunu güncelle veya başlat
                            processed_history = []
                            for msg in st.session_state.all_chats[st.session_state.active_chat_id]:
                                if msg["role"] == "user" and isinstance(msg["parts"][0], bytes):
                                    try:
                                        processed_history.append({"role": msg["role"], "parts": [Image.open(io.BytesIO(msg["parts"][0]))]})
                                    except Exception as e:
                                        logger.warning(f"Geçmişteki görsel yüklenemedi (sesli sohbet), atlanıyor: {e}")
                                        continue 
                                else:
                                    processed_history.append(msg)

                            st.session_state.chat_session = st.session_state.gemini_model.start_chat(
                                history=processed_history
                            )
                            
                            response = st.session_state.chat_session.send_message(recognized_text, stream=True)
                            response_text = ""
                            response_placeholder = st.empty()
                            for chunk in response:
                                response_text += chunk.text
                                with response_placeholder.container():
                                    st.markdown(response_text)
                            
                            add_to_chat_history(st.session_state.active_chat_id, "model", response_text)
                            text_to_speech(response_text)
                            logger.info("Canlı ses girişi için Gemini yanıtı oluşturuldu ve okundu.")
                            st.rerun()

                        except Exception as e:
                            st.error(f"Yanıt alınırken beklenmeyen bir hata oluştu: {e}")
                            st.error(f"Kaynak: Hata ({e})")
                            logger.error(f"Canlı ses yanıtı hatası: {e}")
                else:
                    st.warning("Gemini modeli başlatılmamış.")
                    logger.warning("Canlı ses girişi sırasında Gemini modeli başlatılmamış.")

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
                        # Yaratıcı metin için yeni bir sohbet oturumu başlat, geçmişi sıfırla
                        creative_chat_session = st.session_state.gemini_model.start_chat(history=[])
                        response = creative_chat_session.send_message(f"Yaratıcı metin oluştur: {creative_prompt}", stream=True)
                        
                        response_text = ""
                        response_placeholder = st.empty()
                        for chunk in response:
                            response_text += chunk.text
                            with response_placeholder.container():
                                st.markdown(response_text)
                        
                        add_to_chat_history(st.session_state.active_chat_id, "model", f"Yaratıcı Metin Oluşturuldu: {response_text}")
                        logger.info("Yaratıcı metin başarıyla oluşturuldu ve eklendi.")

                    except Exception as e:
                        st.error(f"Yaratıcı metin oluşturulurken bir hata oluştu: {e}")
                        st.error(f"Kaynak: Hata ({e})")
                        logger.error(f"Yaratıcı metin oluşturma hatası: {e}")
            else:
                st.warning("Gemini modeli başlatılmamış.")
                logger.warning("Yaratıcı stüdyo sırasında Gemini modeli başlatılmamış.")
        else:
            st.warning("Lütfen bir yaratıcı metin isteği girin.")
            logger.warning("Yaratıcı metin oluşturma için prompt girilmedi.")


# --- Main Application Logic (Ana Uygulama Mantığı) ---

def main():
    """Ana Streamlit uygulamasını çalıştırır."""
    st.set_page_config(
        page_title="Hanogt AI Asistan",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Oturum durumu değişkenlerini başlat
    if "models_initialized" not in st.session_state:
        st.session_state.models_initialized = False
        logger.info("models_initialized oturum durumu başlatıldı.")
    
    load_chat_history()
    initialize_gemini_model() # Modeli burada başlat

    display_about_section()
    display_settings_section()
    display_main_chat_interface()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
        <div style="text-align: center; font-size: 12px; color: gray;">
            Kullanıcı: Oğuz Han Gülüzade <br>
            Hanogt AI v5.1.5 Pro+ Enhanced (Refactored) © {datetime.datetime.now().year} <br>
            AI: Aktif ({GLOBAL_MODEL_NAME}) | Log: Aktif
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

